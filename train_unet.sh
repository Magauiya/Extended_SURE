export CUDA_VISIBLE_DEVICES=0
python -W ignore main.py parameters.batch_size=20 parameters.min_std=.1 parameters.max_std=55. parameters.range=255. parameters.loss=mse

export CUDA_VISIBLE_DEVICES=1
python -W ignore main.py parameters.batch_size=20 parameters.min_std=.1 parameters.max_std=55. parameters.range=255. parameters.loss=sure

export CUDA_VISIBLE_DEVICES=0,1
python -W ignore main.py parameters.batch_size=40 parameters.min_std=.1 parameters.max_std=55. parameters.range=255. parameters.loss=sure parameters.num_epochs=3000 steplr.step_size=2000 steplr.gamma=0.1