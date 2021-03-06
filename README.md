# Zero DCE
From the [Li-Chonyi published paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) and [Li-Chongyi - author repository](https://github.com/Li-Chongyi/Zero-DCE), [Tuvovan implemented](https://github.com/tuvovan/Zero_DCE_TF), I tried to learn and re-implement Zero-DCE.

<b>I use Keras library from tensorflow 1.14 to develop this model </b>  


# Requirements

:+1: ```Ubuntu 18.04```

:+1: ```python=3.6```

:+1: ```tensorflow 1.14```

:+1: ```numpy>=1.16.4```

:+1: ```Opencv```

:+1: ```skimage```

:+1: ```pillow```

:+1: ```pyaml```


# Dataset

1. Train dataset from [Li-Chongyi /
Zero-DCE](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view)
2. Test dataset from [Li-Chongyi /
Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code/data/test_data)


## TFrecord
In this project, training data need to be converted to TFrecord format:

There are some Flag from file ```create_tfrecord.py``` need to be consideration:

1. ```train_directory``` in line 21 of ```create_tfrecord.py```: it should be where you store raw training data ```./data/train_data```

2. ```output_directory``` in line 26 of ```create_tfrecord.py```: it's the folder you want to store tfrecord files: ```./data/tfrecord```

3. run ```python3 create_tfrecord.py```

## Train 
<b>Becarefull with the callback function, because I log too many images to tensorboard, so the tensorboard log may very huge, if you do not have large memory hardisk you need to deactive tensorboard callback function.</b>

Need to check the config file in ```./config.yaml```.

```buildoutcfg
DATA_DIR: ./data/tfrecord
LOG_DIR: ./logs
OUTPUTS: ./outputs
INPUT_HEIGHT: 256
INPUT_WIDTH: 256
N_CHANNEL: 3
SOLVER:
  EPOCHS: 200
  INIT_LR: 0.0001
  MOMENTUM: 0.9
  LR_SCHEDULE: cosine
  OPTIMS: sgd
  TRAINING: True
  REGULARIZATION: 0.0001
  BATCH_SIZE: 48
```
To train the model run ```CUDA_VISIBLE_DEVICES=0 python3 train.py```

1. Training log and tensorboard log are stored in ``./logs``

2. weight h5 and *.ckpt are stored in ```./outputs```

## Result

### Model architecture
![Zero-DCE](https://github.com/dattv/Zero_DCE_TF14/blob/main/nets/model.png)

### Conversion history

![conversion-history](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/loss_epochs.png)

### Compare between lowlight and enhanced images

![compare0](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare0.png)
![compare1](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare1.png)
![compare2](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare2.png)
![compare3](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare3.png)
![compare4](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare4.png)
![compare5](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare5.png)
![compare6](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare6.png)
![compare7](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare7.png)
![compare8](https://github.com/dattv/Zero_DCE_TF14/blob/main/test_results/compare8.png)

# Future development
1. ADD QAT (quantization aware training) feature for optimize 8bit Uint8

2. Update channel first format
