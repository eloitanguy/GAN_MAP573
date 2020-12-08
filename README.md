# Using linear GANs to generate MNIST images in PyTorch

A simple application of [Ian Goodfellow's original GAN article](https://arxiv.org/pdf/1406.2661.pdf) on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Repository overview

* *models.py* defines a linear Generator model and a linear Discriminator model.
* *train.py* initialises and trains these models on MNIST examples.
* *sample_generation.py* exports generated examples as a png image.

## Using this repository

Start by installing the required packages if you don't have them yet:

    pip install -r requirements.txt

You can run training by executing:
    
    python train.py

You can export generated images as a png file (after training) by running:

    python sample_generation.py