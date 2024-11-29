# MNIST Digit Classification with CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. This projects aim to reach 99.4% accuracy on the test set with less than 20k parameters and within 20 epochs.

## Overview

This project implements a custom CNN architecture with:
- Multiple convolutional layers with batch normalization and dropout
- Max pooling for feature reduction
- Transition layer (1x1 convolution)
- Final fully connected layer for classification

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- torchsummary

## Key Highlights

- Reached receptive field of 27 at the 5th convolutional layer
- Made use of different image transformation techniques, Learning Rate scheduler
- Max accuracy achieved: 99.56% on the test set


## CNN Architecture Details
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 16, 28, 28]           2,320
       BatchNorm2d-4           [-1, 16, 28, 28]              32
           Dropout-5           [-1, 16, 28, 28]               0
         MaxPool2d-6           [-1, 16, 14, 14]               0
            Conv2d-7           [-1, 16, 14, 14]           2,320
       BatchNorm2d-8           [-1, 16, 14, 14]              32
            Conv2d-9           [-1, 16, 14, 14]           2,320
      BatchNorm2d-10           [-1, 16, 14, 14]              32
          Dropout-11           [-1, 16, 14, 14]               0
        MaxPool2d-12             [-1, 16, 7, 7]               0
           Conv2d-13             [-1, 32, 7, 7]           4,640
      BatchNorm2d-14             [-1, 32, 7, 7]              64
           Conv2d-15              [-1, 8, 7, 7]             264
           Linear-16                   [-1, 10]           3,930
================================================================
Total params: 16,146
Trainable params: 16,146
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.66
Params size (MB): 0.06
Estimated Total Size (MB): 0.72
----------------------------------------------------------------
```

## Layerwise Details

| Layer | Input Size | Output Size | Receptive Field | Jump In | Jump Out |
|-------|------------|-------------|-----------------|----------|-----------|
| Conv0 | 28x28x1 | 28x28x16 | 3 | 1 | 1 |
| Conv1 | 28x28x16 | 28x28x16 | 5 | 1 | 1 |
| MaxPool1 | 28x28x16 | 14x14x16 | 5 | 1 | 2 |
| Conv2 | 14x14x16 | 14x14x16 | 11 | 2 | 2 |
| Conv3 | 14x14x16 | 14x14x16 | 15 | 2 | 2 |
| MaxPool2 | 14x14x16 | 7x7x16 | 19 | 2 | 4 |
| Conv4 | 7x7x16 | 7x7x32 | 27 | 4 | 4 |
| Transition | 7x7x32 | 7x7x8 | 27 | 4 | 4 |
| FC | 7x7x8 | 10 | - | - | - |

## Training Logs - Showing Test Accuracy and Loss
```
********* Epoch = 1 *********
loss=0.03642135113477707 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.80it/s]

Test set: Average loss: 0.0996, Accuracy: 9683/10000 (96.8300%)

********* Epoch = 2 *********
loss=0.0395449623465538 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.83it/s] 

Test set: Average loss: 0.0528, Accuracy: 9832/10000 (98.3200%)

********* Epoch = 3 *********
loss=0.021946275606751442 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.82it/s]

Test set: Average loss: 0.0440, Accuracy: 9858/10000 (98.5800%)

********* Epoch = 4 *********
loss=0.13119633495807648 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.81it/s] 

Test set: Average loss: 0.0368, Accuracy: 9874/10000 (98.7400%)

********* Epoch = 5 *********
loss=0.08169790357351303 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.83it/s] 

Test set: Average loss: 0.0204, Accuracy: 9935/10000 (99.3500%)

********* Epoch = 6 *********
loss=0.016586193814873695 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.83it/s]

Test set: Average loss: 0.0254, Accuracy: 9908/10000 (99.0800%)

********* Epoch = 7 *********
loss=0.03486835956573486 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.85it/s] 

Test set: Average loss: 0.0319, Accuracy: 9905/10000 (99.0500%)

********* Epoch = 8 *********
loss=0.06704837828874588 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.84it/s] 

Test set: Average loss: 0.0254, Accuracy: 9914/10000 (99.1400%)

********* Epoch = 9 *********
loss=0.019271304830908775 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.86it/s]

Test set: Average loss: 0.0144, Accuracy: 9948/10000 (99.4800%)

********* Epoch = 10 *********
loss=0.0075865816324949265 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.81it/s]

Test set: Average loss: 0.0142, Accuracy: 9949/10000 (99.4900%)

********* Epoch = 11 *********
loss=0.05111004784703255 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.75it/s] 

Test set: Average loss: 0.0134, Accuracy: 9951/10000 (99.5100%)

********* Epoch = 12 *********
loss=0.003569856286048889 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.78it/s]

Test set: Average loss: 0.0132, Accuracy: 9954/10000 (99.5400%)

********* Epoch = 13 *********
loss=0.03879081830382347 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.81it/s] 

Test set: Average loss: 0.0131, Accuracy: 9953/10000 (99.5300%)

********* Epoch = 14 *********
loss=0.007756310049444437 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.84it/s]

Test set: Average loss: 0.0130, Accuracy: 9955/10000 (99.5500%)

********* Epoch = 15 *********
loss=0.01726674474775791 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.86it/s] 

Test set: Average loss: 0.0134, Accuracy: 9950/10000 (99.5000%)

********* Epoch = 16 *********
loss=0.04849320277571678 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.84it/s] 

Test set: Average loss: 0.0141, Accuracy: 9950/10000 (99.5000%)

********* Epoch = 17 *********
loss=0.015940533950924873 batch_id=117: 100%|██████████| 118/118 [00:41<00:00,  2.84it/s]

Test set: Average loss: 0.0134, Accuracy: 9956/10000 (99.5600%)

********* Epoch = 18 *********
loss=0.033559706062078476 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.81it/s]

Test set: Average loss: 0.0130, Accuracy: 9955/10000 (99.5500%)

********* Epoch = 19 *********
loss=0.021474793553352356 batch_id=117: 100%|██████████| 118/118 [00:42<00:00,  2.78it/s]

Test set: Average loss: 0.0126, Accuracy: 9955/10000 (99.5500%)
```