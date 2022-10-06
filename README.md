# AISecurity-Course Assignment
This repository is an assignment for the Artificial Intelligence Security course at Zhejiang University's School of Software Technology.I implemente an image classification model on the CIFAR10 dataset and try to modify some hyperparameters.

## Table of contents

* [Introduction](#introduction)
* [Installation and running](#installation-and-running)
* [Tuning a hyper-parameter](#tuning-a-hyper-parameter)
* [License](#license)
* [Acknowledgments](#acknowledgments)


## Introduction

In Experiment, I train model using Adam optimizer with learning rate 3e-4.



## Installation and running
#### Step 1: Clone the Code from Github

```
git clone https://github.com/Chao-Ye/AISecurity-Course.git
cd AISecurity-Course
```
#### Step 2: Install Requirements

Create a new virtual environment using conda and execute the following command to install the python dependencies.
```
conda create -n testenv python=3.8
source activate testenv
pip install -r requirement.txt 
```
#### Step3: Training Network
```
python train.py --use_dropout
```
We also support modifying runtime parameters, such as the training device(cpu/cuda) and the number of training epochs.Seeing train.py for details.


## Tuning a hyper-parameter

<img src="result_image/LeNet_result.jpg" />

As you can see, this picture shows my initial training results.It is not difficult to find that as the number of training epochs increases, the loss value is decreasing and the accuracy on the training set is increasing. However, there is a huge accuracy gap between the test set and the training set, and the accuracy on the test set shows a decreasing trend in the later stages.I analyzed that the model was overfitting on the training data, so I added some regularization means. Specifically, I used the dropout on the first two linear layers of LeNet.The result is as follows:

<img src="result_image/LeNet_dropout_result.jpg" />

The results show that the model has better generalizability after using the dropout.

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/mit-license) file for details

## Acknowledgments
* Song Jie Teacher
* All teaching assistants for the Artificial Intelligence Security course



