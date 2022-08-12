# Detecting Concrete Crack Images for Classification

## 1. Overview
Ther purpose of this project is to classify an images of concrete wall whether it has cracks or not. This project was conducted by using the dataset from https://data.mendeley.com/datasets/5y9wdsg2zt/2 and was created using Spyder IDE. 

## 2. Methodology
### Data preprocessing

The data loaded is already in their classification file which is positive and negative crack folder. The data was splitted into the common split that is 70:30.

### Model
a. Data augmentation as RandomFlip and RandomRotate is applied on the training dataset.
b. The input layer is our augmentation layer where it will receive an image size of (100,100,3). As for the classification part of our model a global average pooling2D  are used and output layer of softmax activation function. 

![image](https://user-images.githubusercontent.com/108311968/184271257-a2c4dbde-165c-4cfc-80d7-7e77d93a0115.png)


The model was trained with epochs of 20 and batch size 32. Early stopping are applied to the model, and after reaches 0.9980 accuaracy and val_accuracy: 0.9986 the model was stopped at 3rd epochs.

![image](https://user-images.githubusercontent.com/108311968/184271325-abe8cc8a-576f-4d01-8dcd-abcf16fadf9a.png)
![image](https://user-images.githubusercontent.com/108311968/184271403-ebdd44de-a89f-4612-b33a-f31bb1c93c9e.png)

## 3. Result

Example of images was used and the classification are shown below

![result_concrete crack images model](https://user-images.githubusercontent.com/92585515/182049177-03c7b60e-293b-4a07-91ed-ecde6d04c27e.png)
