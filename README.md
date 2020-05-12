# dl-stencil

Template for a typical pytorch deep learning project based on [pycls](https://github.com/facebookresearch/pycls). Several modules (like tensorboard logging) also take from [Pytorch Ignite](https://github.com/pytorch/ignite).

Features (still WIP):

* Sensible config management using [yacs](https://github.com/rbgirshick/yacs)
* Tensorboard + json logging
* Early stopping
* LR policies like Exp, Cosine
* Should also largely work for multi-gpu setups, but am not that rich, so can't test it yet

Scroll through different branches for different training setups:
* master - Image classification on MNIST using Resnet50
* vae - Variational Autoencoder on MNIST
* lm (WIP) - Language model on IMDB using a simple LSTM
* gan (WIP) - Train a GAN model on TBD

