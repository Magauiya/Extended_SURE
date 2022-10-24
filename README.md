# PyTorch Implementation

### To train with MC-SURE loss use the following command:
```
python -W ignore main.py parameters.batch_size=20 parameters.min_std=.1 parameters.max_std=55. parameters.range=255. parameters.loss=sure
```

### To train with MSE loss and to compare with MC-SURE, use below command:
```
python -W ignore main.py parameters.batch_size=20 parameters.min_std=.1 parameters.max_std=55. parameters.range=255. parameters.loss=sure
``` 

### TO DO:
- Implement Extended MC-SURE (wip)
- Prepare tests for MC-SURE and Extended MC-SURE losses
