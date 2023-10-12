# Multilayer Perception
Introduction to artificial neural networks. Implementation of a multilayer perceptron that predicts whether a cancer is malignant or benign based on characteristics of cell nucleus. 

## Introduction
This project can be subdivided into three parts:
- Splitting the dataset into training and testing sets
- Model creation and training
- Prediction and evaluation

## Dataset
The features of the dataset describe the characteristics of a cell nucleus of breast mass extracted with fine-needle aspiration. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The dataset contains 568 instances with 30 numeric attributes.

### Splitting the dataset into training and validation sets
```
make split
```

## Training
The model  will be trained with the training set and the validation set will be used to evaluate the model.
The multilayer perceptron object is modular, you can use the --help option to see the different options available.
```
make train
```

## Prediction and evaluation
The model will be used to predict the class of the instances in the test set and the accuracy will be calculated.
```
make predict
```
