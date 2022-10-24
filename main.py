import os
import glob
import random
import hydra
import time
import numpy as np
from skimage.measure import compare_psnr as PSNR
from scipy.io import loadmat

# PyTorch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# OWN
import model
from dataloader import BSD500_Dataset
from utils import MCSURE, get_epsilon

torch.autograd.set_detect_anomaly(True)

class ImageDenoiser:
    def __init__(self, cfg):
        self.cfg = cfg.parameters
        self.steplr = cfg.steplr
        self.plateau = cfg.plateau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self):
        
        # MODEL
        model_name = self.cfg.model_name
        if model_name not in dir(model):
            model_name = "UNet"
        self.model = getattr(model, model_name)(cfg=self.cfg).to(self.device)
        if self.cfg.net_verbose:
            summary(self.model, (1, 256, 256))

        print('-' * 40)
        print(f"[*] Model: {self.cfg.model_name}")
        print(f"[*] Device: {self.device}")
        print(f"[*] Path: {self.cfg.data_dir}")
        print(f"[*] Loss: {self.cfg.loss}")
        print(f"[*] Range: [0-{self.cfg.range}]")

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_init)


        # LOSSES 
        self.mse = nn.MSELoss().to(self.device)
        self.mcsure = MCSURE(self.cfg.range).to(self.device)
        
        # Data range:
        if self.cfg.range == 255.:
            self.scaling = 1.
        elif self.cfg.range == 1.:
            self.scaling = 255.
        else:
            print("[!] Input data range should be: [0-1] or [0-255]")
            exit() 

        # SCHEDULER
        if self.cfg.scheduler == "step":
            self.lr_sch = lr_scheduler.StepLR(self.optimizer, **self.steplr)
        else:
            self.lr_sch = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.plateau)

        # INITIALIZE: dirs, ckpts, data loaders
        self._make_dir()
        self._load_ckpt()
        self._load_data()
        self.writer = SummaryWriter(log_dir=self.cfg.logs_path)
        self.model = nn.DataParallel(self.model)

    def _perturbed(self, noisy, sigma):
        self.model.train()
        eps = get_epsilon(sigma, self.cfg.range)
        norm_vec = torch.from_numpy(np.random.normal(0, 1., noisy.size()))
        norm_vec = norm_vec.to(self.device, dtype=torch.float)
        perturbed = noisy + torch.einsum("b, bchw->bchw", [eps, norm_vec])
        
        out_p = self.model(perturbed)
        return out_p, norm_vec

    def train(self):
        self.model.train()
        val_loss, val_psnr = self.evaluate(self.step - 1, self.start_epoch)
        print("[*] Preliminary check: Epoch: {} Step: {} Val Loss: {:.5f} PSNR: {:.3f}".format(
            self.start_epoch,
            (self.step-1),
            val_loss["rnd"],
            val_psnr["rnd"]
            ))
        print('-' * 100)
        
        best_psnr = 0.
        # Resume training from stopped epoch
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            step_mse, step_sure = 0, 0
            start_time = time.time()

            for idx, (noisy, clean, sigma) in enumerate(self.train_loader, start=1):
                # Input/Target 
                noisy = noisy.to(self.device, dtype=torch.float) / self.scaling 
                clean = clean.to(self.device, dtype=torch.float) / self.scaling 
                sigma = sigma.to(self.device, dtype=torch.float) / self.scaling 
                
                # BackProp
                self.optimizer.zero_grad()
                out = self.model(noisy)
                mse  = self.mse(out, clean)

                # MC-SURE
                out_p, norm_vec = self._perturbed(noisy, sigma)
                sure = self.mcsure(noisy, out, out_p, sigma, norm_vec)
                
                if self.cfg.loss == "mse":
                    mse.backward()
                elif self.cfg.loss == "sure":
                    sure.backward()
                    
                self.optimizer.step()

                # STATS
                step_mse += mse.item()
                step_sure += sure.item()

            val_mse, val_psnr = self.evaluate(self.step, epoch)

            self.writer.add_scalar("Train/MSE", step_mse/idx, self.step)
            self.writer.add_scalar("Train/SURE", step_sure/idx, self.step)
            self.writer.add_scalar("Valid_std_0-55/MSE", val_mse['rnd'], self.step)
            self.writer.add_scalar("Valid_std_0-55/SURE", val_psnr['rnd'], self.step)

            self.writer.add_scalar("Valid_std_25/MSE", val_mse['25'], self.step)
            self.writer.add_scalar("Valid_std_25/SURE", val_psnr['25'], self.step)

            self.writer.add_scalar("Valid_std_50/MSE", val_mse['50'], self.step)
            self.writer.add_scalar("Valid_std_50/SURE", val_psnr['50'], self.step)

            self.writer.add_scalar("Stats/LR", self.optimizer.param_groups[0]['lr'], self.step)


            print("[{}/{}/{}] T-Loss [MSE/SURE]: [{:.5f}/{:.5f}] Val. PSNR: [{:.2f}/{:.2f}/{:.2f}] Sigma: [{:.1f}-{:.1f}] LR: {} Time: {:.1f} Out: [{:.3f}-{:.3f}]".format(
                epoch, self.step, idx,
                (step_mse/self.cfg.verbose_step), 
                (step_sure/self.cfg.verbose_step),
                val_psnr["rnd"],
                val_psnr["25"],
                val_psnr["50"],
                torch.min(sigma).item()*self.scaling,
                torch.max(sigma).item()*self.scaling,
                self.optimizer.param_groups[0]['lr'],
                (time.time()-start_time),
                torch.min(out).item(),
                torch.max(out).item()
                ))
            
            if best_psnr < val_psnr["25"]:
                ckpt_name = f"best_psnr.pth"
                self._save_ckpt(epoch, ckpt_name)
                
            if epoch % 500 == 0:
                ckpt_name = f"epoch_{epoch}_loss_{self.cfg.loss}_{self.cfg.range}_psnr_{val_psnr}.pth"
                self._save_ckpt(epoch, ckpt_name)
            
            
            self.step += 1
            if self.cfg.scheduler == "step":
                self.lr_sch.step()
            elif self.cfg.scheduler == "plateau":
                self.lr_sch.step(metrics=val_loss) 
            
            step_loss, start_time = 0, time.time()
            self.model.train()


    @torch.no_grad()
    def evaluate(self, step, epoch):
        self.model.eval()
        
        epoch_psnr = {"rnd": 0, "25": 0, "50": 0}
        epoch_loss = {"rnd": 0, "25": 0, "50": 0}
        running_psnr = 0.
        running_loss = 0.

        B, _, H, W = np.shape(self.clean_test)
        step = 20

        for k, std in enumerate(epoch_psnr.keys()):
            clean_test = self.clean_test
            noisy_test = self.noisy_test[k]

            running_psnr, running_loss = 0., 0.
            for idx in range(0, B, step):
                noisy = noisy_test[idx:(idx+step)].to(self.device, dtype=torch.float)
                clean = clean_test[idx:(idx+step)].to(self.device, dtype=torch.float)

                output = self.model(noisy)

                if idx == 1:
                    save_image(output, 'step_%d_output.png'%step, nrow=4)
                    save_image(clean, 'step_%d_clean.png'%step, nrow=4)
                
                loss = self.mse(output, clean)
                running_loss += loss.item()

                clean = clean.cpu().detach().numpy()
                output = np.clip(output.cpu().detach().numpy(), 0., self.cfg.range)

                # ------------ PSNR ------------
                for m in range(step):
                    running_psnr += PSNR(clean[m], output[m], data_range=int(self.cfg.range))

            epoch_loss[std] = running_loss / B
            epoch_psnr[std] = running_psnr / B

        #ckpt_name = f"epoch_{epoch}_val_loss_{epoch_loss['rnd']:.3}_psnr_{epoch_psnr['rnd']:.3}.pth"
        #self._save_ckpt(epoch, ckpt_name)
        self.model.train()
        
        return epoch_loss, epoch_psnr


    def _load_ckpt(self):
        self.step = 1
        self.start_epoch = 0
        if os.path.exists(self.cfg.resume):
            resume_path = self.cfg.resume
        else:
            ckpts = [[f, int(f.split("_")[1])] for f in os.listdir(self.cfg.ckpt_path) if f.endswith(".pth")]
            ckpts.sort(key=lambda x: x[1], reverse=True)
            resume_path = None if len(ckpts) == 0 else os.path.join(self.cfg.ckpt_path, ckpts[0][0])

        if resume_path and os.path.exists(resume_path):
            print("[&] LOADING CKPT {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu')
            self.step = checkpoint['step'] + 1
            self.model.load_state_dict(checkpoint['model'])

            if not self.cfg.optim_reset:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # To support previously trained models w\o epoch parameter
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']

            print("[*] CKPT Loaded '{}' (Start Epoch: {} Step {})".format(resume_path, self.start_epoch,
                                                                          checkpoint['step']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("[!] NO checkpoint found at '{}'".format(resume_path))


    def _save_ckpt(self, epoch, save_file):
        state = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': epoch
        }
        torch.save(state, os.path.join(self.cfg.ckpt_path, save_file))
        del state


    def _load_data(self):

        img_list = glob.glob(os.path.join(self.cfg.data_dir, "BSD500", "*.jpg"))
        train_dataset = BSD500_Dataset(self.cfg, img_list)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True    # easier to estimate PSNR, loss, etc. 
        )

        tmp = loadmat(os.path.join(self.cfg.data_dir, "TestClean_180x180.mat"))['clean']
        self.clean_test = torch.from_numpy(np.expand_dims(np.asarray(tmp, dtype=np.float32), axis=1)) / self.scaling

        tmp = loadmat(os.path.join(self.cfg.data_dir, "TestNoisy_rnd_180x180.mat"))['noisy']
        noisy_test = np.expand_dims(np.asarray(tmp, dtype=np.float32), axis=1)
        tmp  = loadmat(os.path.join(self.cfg.data_dir, "TestNoisy_25_180x180.mat"))['noisy']
        noisy25_test = np.expand_dims(np.asarray(tmp, dtype=np.float32), axis=1)
        tmp = loadmat(os.path.join(self.cfg.data_dir, "TestNoisy_50_180x180.mat"))['noisy']
        noisy50_test = np.expand_dims(np.asarray(tmp,  dtype=np.float32), axis=1)

        self.noisy_test = torch.from_numpy(np.stack((noisy_test, noisy25_test, noisy50_test), axis=0)) / self.scaling

        print(f"[*] Trainset: {self.cfg.batch_size} x {len(self.train_loader)}")
        print(f"[*] Testset: {self.noisy_test.shape}")
        del tmp
        pass

    def _make_dir(self):
        # Output path: ckpts, imgs, etc.
        if not os.path.exists(self.cfg.ckpt_path):
            os.mkdir(self.cfg.ckpt_path)
            print("[*] Checkpoints Directory created!")
        else:
            print("[*] Checkpoints Directory already exist!")

        if not os.path.exists(self.cfg.logs_path):
            os.mkdir(self.cfg.logs_path)
            print("[*] Logs Directory created!")
        else:
            print("[*] Logs Directory already exist!")


@hydra.main(config_path="./default.yaml")
def main(cfg):
    seed = cfg.parameters.random_seed
    print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    app = ImageDenoiser(cfg)
    app.build()

    if cfg.parameters.inference:
        app.test()
    else:
        app.train()


if __name__ == "__main__":
    main()
