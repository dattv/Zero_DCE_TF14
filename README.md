# Zero DCE
From the [published paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) and [Li-Chongyi - author repository](https://github.com/Li-Chongyi/Zero-DCE), [Tuvovan implemented](https://github.com/tuvovan/Zero_DCE_TF), I tried to learn and re-implement Zero-DCE.  

(:point_right:<b>This project is under development</b>:point_left:)

# Setup

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
