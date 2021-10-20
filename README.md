# alignimation
Automated image registration. Registrationimation was too much of a mouthful.

This repo contains the code used for my blog post [Alignimation: Differentiable, Semantic Image Registration with Kornia](https://www.ethanrosenthal.com/2021/11/03/alignimation/). This code is not really meant to be used as a general purpose library; rather, this just just my collection of code that I happened to put into a package.

Maybe one day I'll turn this into a general purpose library for aligning things, but today is not that day.

## Installation

If you do want to use the code for some reason, you can install the dependencies with [poetry](https://python-poetry.org/) by cloning the repo and running `poetry install`. The code has been run with python 3.8. I don't know if it works for other versions.

If you don't feel like installing poetry, you can try installing the following libraries with `pip` and praying that the versions work together:

```
matplotlib
torch
torchvision
kornia
tqdm
facenet-pytorch
```

## Usage

You can load all images in a path with `alignimation.io.load_images`. The images will be loaded in sorted order of the date that they were taken.

The main function for creating the following gif was `alignimation.base.alignimate`.

![body](./static/body.gif)

To create the below gif tracking my face, I first ran `alignimation.face.main` to generate the facial keypoints and then ran `alignimation.base.alignimate`

![face](./static/face.gif)

## Tests

I didn't feel like writing any tests
