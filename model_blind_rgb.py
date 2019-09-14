from __future__ import print_function
from __future__ import division
import time
from utils import *
 

def dncnn(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 19 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training, name='bn%d' % layers))
    with tf.variable_scope('block20'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input-output


class denoiser(object):
    def __init__(self, sess, sigma, cost_str, ckpt_dir, sample_dir, log_dir):
        self.sess = sess
        self.sigma = sigma
        
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
    
        # build model
        #placeholders for clean and noisy image batches
        self.GT = tf.placeholder(tf.float32, [None, None, None, 3], name='gt_true_image')
        self.X = tf.placeholder(tf.float32, [None, None, None, 3], name='gt_fake_image')
        self.Y = tf.placeholder(tf.float32, [None, None, None, 3], name='noisy_image')
        self.Z = tf.placeholder(tf.float32, [None, None, None, 3], name='less_noisy')
        
        self.is_training = tf.placeholder(tf.bool, name='is_training') #for batchnorm

        #vector containing sigma values for each image
        self.sigma_tf = tf.placeholder(tf.float32, [None], name='sigma_vector')
        self.sigma_tf_less = tf.placeholder(tf.float32, [None], name='sigma_vector')
        
        #tuned epsilon values
        self.eps_tf = 1.6*self.sigma_tf*0.0001  # for gt range [0-1]
        #self.eps_tf = tf.maximum(.001 * tf.reduce_max(tf.reshape(self.Y, shape=[tf.shape(self.Y)[0], -1]), axis=-1), .00001)  # for gt range [0-255] and remove normalization for sigma
        self.eps_tf = self.eps_tf[:, tf.newaxis, tf.newaxis, tf.newaxis] #for vectorized calculation of the sure
        
        #forward propagation
        with tf.variable_scope('DnCNN'):
            self.Y_ = dncnn(self.Y, is_training=self.is_training)

        #forward propagation of the perturbed input
        self.b_prime = tf.random_normal(shape=tf.shape(self.Y), stddev=1.0)
        self.Yptb = self.Y + tf.multiply(self.b_prime, self.eps_tf)
        with tf.variable_scope('DnCNN', reuse=True):
            self.Yptb_ = dncnn(self.Yptb, is_training=self.is_training)
        
        self.Ht = tf.to_float(tf.shape(self.Y)[1]) #height of the image
        self.Wt = tf.to_float(tf.shape(self.Y)[2]) #width  of the image

        batch = tf.to_float(tf.shape(self.Y)[0])    #size of the minibatch
        
        self.var_Y = tf.square(self.sigma_tf/255.0, name = 'var_vector')
        self.var_Y = self.var_Y[:, tf.newaxis, tf.newaxis, tf.newaxis] #for vectorized calculation of the sure

        self.var_Z = tf.square(self.sigma_tf_less/255.0, name = 'var_vector_less') #tensor of variance values
        self.var_Z = self.var_Z[:, tf.newaxis, tf.newaxis, tf.newaxis] #for vectorized calculation of the sure

        self.divergence = tf.multiply((1.0/self.eps_tf), tf.multiply(self.b_prime, (self.Yptb_-self.Y_)))
        self.divergence_sum_Y = tf.reduce_sum(tf.multiply(self.var_Y, self.divergence))
        self.divergence_sum_Z = tf.reduce_sum(tf.multiply(self.var_Z, self.divergence))

        ############################################-COSTS-############################################
        # MSE
        self.mse = (1.0 / batch) * tf.nn.l2_loss(self.Y_ - self.GT)
        # N2N
        self.n2n = (1.0 / batch) * tf.nn.l2_loss(self.Y_ - self.X)

        # SURE
        self.var_sum_Y = self.Ht * self.Wt * 3 * tf.reduce_sum(self.var_Y)/2.0
        self.sure     = (1.0 / batch)*(tf.nn.l2_loss(self.Y - self.Y_) - self.var_sum_Y + self.divergence_sum_Y)

        # NEW SURE
        self.var_sum_Z = self.Ht * self.Wt * 3 * tf.reduce_sum(self.var_Z)/2.0
        self.new_sure = (1.0 / batch)*(tf.nn.l2_loss(self.Z - self.Y_) - self.var_sum_Z + self.divergence_sum_Z)

        
        #which cost function to use for training
        if cost_str=='SURE':
            self.cost = self.sure
            print('[*] Optimizing SURE!')
        elif cost_str=='N2N':
            self.cost = self.n2n
            print('[*] Optimizing Noise2Noise!')
        elif cost_str=='MSE':
            self.cost = self.mse
            print('[*] Optimizing MSE!')
        elif cost_str == 'e-SURE':
            self.cost = self.new_sure
            print('[*] Optimizing e-SURE!')
        else:
            print("UNKNOWN COST")

        
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf.placeholder(tf.float32, name='eva_psnr')

        #optimizer
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        #for batchnorm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.cost) ##CHANGED

        #checkpoint saver
        self.saver = tf.train.Saver(max_to_keep=15)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    #function to evaluate the performance after each epoch
    def evaluate(self, test_files, iter_num, summary_merged, summary_writer):
        mse_sum_ev = 0
        sigma_img = np.reshape(np.asarray(np.squeeze(self.sigma), dtype=np.float32), [1, ])
        print('Sigma image shape: ', np.shape(sigma_img))
    
        
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
            clean_image  = load_images_RGB(test_files[idx]).astype(np.float32) / 255.0
            noisy_image  = clean_image + np.random.normal(0, self.sigma/255.0, np.shape(clean_image)).astype('float32')

            # not important, just needed to denoise an image
            noisy_image2 = clean_image + np.random.normal(0, self.sigma/255.0, np.shape(clean_image)).astype('float32') 
            less_image = noisy_image2 
            
            output_clean_image, mse_ev = self.sess.run([self.Y_, self.mse], feed_dict={self.GT: clean_image, 
                                                                                    self.X: clean_image,
                                                                                    self.Y: noisy_image,
                                                                                    self.Z: less_image,
                                                                                    self.sigma_tf: sigma_img,
                                                                                    self.is_training: False})
            
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')

            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            if idx<10:
                print("img%d PSNR: %.2f" % (idx+1, psnr))
            psnr_sum += psnr
            if idx < 5:
                save_images_RGB(os.path.join(self.sample_dir, 'eval%d_%d.png' % (idx + 1, iter_num)), groundtruth, noisyimage, outputimage)
            
            # statistics
            mse_sum_ev += mse_ev
        
        avg_psnr = psnr_sum / len(test_files)
        mse_ave_ev      = mse_sum_ev      / len(test_files)


        # statistics
        print("Test set length: ", len(test_files))
        print("Iter: %d MSE: %.5f \n" % (iter_num, mse_ave_ev))
        print("--- Evaluation on BSD68 Dataset --- Average PSNR %.2f ---" % avg_psnr)
        
        
        psnr_summary = self.sess.run(summary_merged, feed_dict={self.eva_psnr:avg_psnr})
        summary_writer.add_summary(psnr_summary, iter_num)


    def train(self, data_path, eval_files, batch_size, epoch, lr, gt_type):

        # CLEAN ground-truth trainset
        gt_data      = np.load(os.path.join(data_path, 'gt_rgb_clean_patches.npy'), mmap_mode='r') 

        # IMPERFECT ground-truth trainset
        gt_name      = 'gt_rgb_noisy_' + str(gt_type) + '_patches.npy'
        gt_noisy     = np.load(os.path.join(data_path, gt_name), mmap_mode='r')    

        # Noise std. of imperfect ground-truth images
        sigmapath_imp   = 'sigma_rgb_vector_imperfect_' + str(gt_type) + '.npy'
        sigma_imperfect = np.load(os.path.join(data_path, sigmapath_imp), mmap_mode='r')        #.astype('float32')/255.


        # NOISY patches trainset
        noisy_name   ='rgb_noisy_' + str(gt_type) + '_patches.npy'
        noisy        = np.load(os.path.join(data_path, noisy_name), mmap_mode='r')    

        # Noise std. of noisy trainset images
        sigmapath_ny = 'sigma_rgb_vector_blind_' + str(gt_type) + '_' + str(int(self.sigma)) + '.npy'
        sigma_vector = np.load(os.path.join(data_path, sigmapath_ny), mmap_mode='r')        #.astype('float32')/255.


        print('[*] Dataset path: %s Sigma path: %s' % (datapath, sigmapath))
        print('[*] GT  clean range: [%.2f %.2f]' % (np.amin(gt_data), np.amax(gt_data)))
        print('[*] GT  noisy range: [%.2f %.2f]' % (np.amin(gt_noisy), np.amax(gt_noisy)))
        print('[*] Noisy img range: [%.2f %.2f]' % (np.amin(noisy), np.amax(noisy)))
        print('[*] Sigma vec range: [%.2f %.2f]' % (np.amin(sigma_vector), np.amax(sigma_vector)))
        print('[*] Imperfect GT sigma vec range: [%.2f %.2f]' % (np.amin(sigma_imperfect), np.amax(sigma_imperfect)))
        print('[!] BUT network is trained and tested on normalized images between [0-1]')

        numData = np.shape(gt_data)[0]
        numBatch = int(numData / batch_size)

        # load pretrained model
        load_model_status, global_step = self.load(self.ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = (global_step) // numBatch
            start_step = (global_step) % numBatch
            print("[*] Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("[*] Not find pretrained model!")

        # make summary
        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('N2N', self.n2n)
        tf.summary.scalar('SURE', self.sure)
        tf.summary.scalar('e-SURE', self.new_sure)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter(self.log_dir+"/", self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        
        
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(eval_files, iter_num, summary_merged=summary_psnr, summary_writer=writer)
        
        tf.get_default_graph().finalize() # making sure that the graph is fixed at this point
        
        #training loop
        for epoch in xrange(start_epoch, epoch):
            print("Model: %s" % (self.ckpt_dir))
            print("Learning rate: {}".format(lr[epoch]))
            
            rand_inds=np.random.choice(numData, numData,replace=False)

            for batch_id in xrange(0, numBatch):
                # No RAM required
                batch_rand_inds = rand_inds[batch_id * batch_size:(batch_id + 1) * batch_size]
                
                batch_images            = np.array(gt_data[batch_rand_inds]).astype(np.float32) / 255.0     # clean gt
                batch_images_noise      = np.array(gt_noisy[batch_rand_inds]).astype(np.float32) / 255.0    # imperfect gt
                batch_sigma_less        = np.array(sigma_imperfect[batch_rand_inds]).astype(np.float32)     # imperfect gt Gaussian std.

                batch_images_corrupt    = np.array(noisy[batch_rand_inds]).astype(np.float32) / 255.0       # noisy
                batch_sigma             = np.array(sigma_vector[batch_rand_inds]).astype(np.float32)        # noisy Gaussian std. 

                
                feed_dict = {self.GT: batch_images,
                             self.X: batch_images_noise,
                             self.Y: batch_images_corrupt,
                             self.Z: batch_images_noise,
                             self.sigma_tf: batch_sigma,
                             self.sigma_tf_less: batch_sigma_less,
                             self.lr: lr[epoch],
                             self.is_training: True}

                self.sess.run(self.train_op, feed_dict=feed_dict)
                
                if (iter_num)%100==0:
                    feed_dict2 = {self.GT: batch_images,
                                 self.X: batch_images_noise,
                                 self.Y: batch_images_corrupt,
                                 self.Z: batch_images_noise,
                                 self.sigma_tf: batch_sigma,
                                 self.sigma_tf_less: batch_sigma_less,
                                 self.lr: lr[epoch],
                                 self.is_training: False}

                    mse, n2n, sure, new_sure, summary = self.sess.run([self.mse, self.n2n, self.sure, self.new_sure, merged],feed_dict=feed_dict2)
                    print("Epoch: [%2d] [%4d/%4d] Time: %4.4f MSE: %.6f, n2n: %.6f, sure: %.6f, new_sure: %.6f"
                          % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, mse, n2n, sure, new_sure))
                    writer.add_summary(summary, iter_num)
                
                if (iter_num+1)%973==0:
                    self.evaluate(eval_files, iter_num+1, summary_merged=summary_psnr, summary_writer=writer)
                    
                
                iter_num += 1
            self.save(iter_num, self.ckpt_dir)
            print('\n')
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='CDnCNN-tensorflow'):
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
            
    def test(self, test_files, noisy_files, save_dir):
        """Test CDnCNN"""
        # init variables
        tf.initialize_all_variables().run()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        load_model_status, global_step = self.load(self.ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        
        numData = len(test_files)
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(numData):
            clean_image = load_images_RGB(test_files[idx]).astype(np.float32) / 255.0
            noisy_image = np.load(noisy_files[idx]).astype(np.float32)

            print('cl shape: {}'.format(np.shape(clean_image)))
            print(np.amax(clean_image))
            print('ny shape: {}'.format(np.shape(noisy_image)))
            print(np.amin(noisy_image))

            output_clean_image = self.sess.run(self.Y_, feed_dict={self.X: clean_image, self.Y: noisy_image, self.is_training: False})

            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')

            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_sum += psnr
            
            img_name, img_ext = os.path.splitext(os.path.basename(test_files[idx]))
            print("%s PSNR: %.2f" % (os.path.basename(test_files[idx]), psnr))
            
            save_images_RGB(os.path.join(save_dir, 'noisy%s.png' % img_name), noisyimage)
            save_images_RGB(os.path.join(save_dir, 'denoised%s.png' % img_name), outputimage)

        avg_psnr = psnr_sum / len(test_files)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
