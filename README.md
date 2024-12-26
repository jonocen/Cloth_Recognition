# Cloth Recognition

This is a personal project that implements a Convolutional Neural Network (CNN) using PyTorch for the task of clothing item classification. The model is designed to recognize different types of clothing from the Fashion MNIST dataset.

## Project Overview

The goal of this project is to build a CNN that can accurately classify images of clothing items into one of ten categories. The Fashion MNIST dataset serves as the training and testing data for this model.

## Architecture

The CNN architecture consists of the following layers:

- **Input Layer**: Accepts grayscale images of size \(28 \times 28\) pixels.
- **Convolutional Layers**:
  - **Conv2D Layer 1**: 1 input channel, 4 output channels, kernel size \(4 \times 4\).
  - **Conv2D Layer 2**: 4 input channels, 4 output channels, kernel size \(2 \times 2\).
- **Pooling Layers**:
  - **MaxPool2D Layer 1**: Pooling size \(4 \times 4\).
  - **MaxPool2D Layer 2**: Pooling size \(2 \times 2\).
- **Flatten Layer**: Flattens the output from the convolutional and pooling layers into a 1D tensor.
- **Fully Connected Layers**:
  - Linear layer with 32 neurons.
  - Linear layer with 64 neurons (two layers).
  - Output layer with 10 neurons (one for each clothing category).
- **Activation Function**: ReLU (Rectified Linear Unit) is used after each fully connected layer (except the output layer) to introduce non-linearity.

## Dataset

The model is trained and evaluated using the **Fashion MNIST** dataset, which consists of 70,000 grayscale images of clothing items categorized into 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Installation

To run this project, you need to have Python and the following libraries installed:

- PyTorch
- torchvision
- numpy

You can install the required libraries using pip:

```bash
pip install torch torchvision 
