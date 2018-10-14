# Shoezam
Identify shoes by a picture.

# Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Run](#run)
- [Model](#model)
	- [Input](#input)
	- [Characteristics](#characteristics)
	- [Classifications](#classifications)
	- [Layers](#layers)

# Overview
Hack UMass project.  

Uses a convolutional neural network to determine if an image is a shoe or not.

Goals:
- [x] Shoe or not shoe (Using Tensor Flow)
- [ ] Type of shoe (Ex: Boot, sneaker, sandals)
- [ ] Branding of Shoe (Nike, Adidas)
- [x] Setup web server
- [x] Finish first working demo
- [x] Start writing our own algorithms to replace Tensor Flow
- [ ] Get Second working demo

# Setup
## GPU
For best performance run on a machine with a GPU.  

Install Nvidia's CUDA drivers, and their Nvidia Docker tool.

## Docker
Docker is used to run programs. Ensure Docker is installed.  

Next build the base Docker image:

```
make docker-build
```

# Run
The app runs in a Docker container.

Several useful Make targets are provided to run the app:

- `docker-notebook`: Start Jupyter notebook in a container
- `docker-web`: Start Flask web app in a container
- `docker-train`: Train the model in a container
- `docker-predict`: Test a few pre-selected images against the model

# Model
The convolution neural network is referred to as the "model".  

## Input
It takes images in the following form as input:

- 28x28 px
- Grayscale
- Inverted
- Pixel values normalized between [0, 1]
- In the shape: `(batch size dimension, 28, 28, 1)`
	- Where `batch size dimension` is the number of images the model should process

## Characteristics
The model extracts the following characteristics from input images:

- Image with edge detection
- Image with Gaussian blur

These characteristics are used as the base input for the model's layers.  
These characteristics are extracted by convoluting a kernel over input images.

## Classifications
The training data set is a 60,000 item large [fashion mnist](https://github.com/zalandoresearch/fashion-mnist).  

There are 10 label values present in this list. 3 of these values are shoe types. 
If an image is classified to be any of the 3 shoe types it will be marked as a shoe.

## Layers
The neural network is a series interconnected nodes and weights organized into layers.

The layers appear in the following order:

- Filters layer
	- Input shape: `(batch size, 28, 28, 1)`
	- Output shape: `(batch size, 26, 26, 2)`
	- Extracts characteristics from the input image
	- The output shape is 2 smaller in the 2nd and 3rd dimension b/c the kernels used 3x3 pixels and can not be convoluted on the very edge of the image
- Flatten layer
	- Input shape: `(batch size, 26, 26, 2)`
	- Output shape: `(batch size, 1352)`
		- `1352 = 26 * 26 * 2`
	- Flattens the result of the the filter layer in a single list of numbers
- RELU
	- Input shape: `(batch size, 1352)`
	- Output shape: `(batch size, 128)`
	- Transforms every value into the range of [0, 1] based on how close it is to the medium value
		- Closer to lowest value = mapped closer to 0
		- Closer to medium value = mapped closer to 0.5
		- Closer to highest value = mapped closer to 1
- Softmax
	- Input shape: `(batch size, 128)`
	- Output shape: `(batch size, 10)`
	- Maps the result of the RELU layer to 10 nodes, 1 node for each classification an image can be, where the value of each node ends up being the probability that classification is present
