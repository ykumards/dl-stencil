# dl-stencil (WIP)

Template for a typical pytorch deep learning project based on [pycls](https://github.com/facebookresearch/pycls). Several modules (like tensorboard logging) also take from [Pytorch Ignite](https://github.com/pytorch/ignite).

The goal of this template is to have a functioning, minimalistic framework to run some quick experiments. I wouldn't recommend this for anything fancy like in production environments.

#### Features (still WIP):

* Sensible config management using [yacs](https://github.com/rbgirshick/yacs)
* Tensorboard + json logging
* Early stopping
* LR policies like Exp, Cosine
* Should also largely work for multi-gpu setups, but am not that rich, so can't test it yet

#### Branches are useful

Scroll through different branches for different training setups:
* master - Image classification on MNIST using Resnet50
* vae - Variational Autoencoder on MNIST
* lm (WIP) - Language model on IMDB using a simple LSTM
* gan (WIP) - Train a GAN model on TBD

After creating a new repo with this template, you might have to make the corresponding branch as the default branch using either [this](https://stackoverflow.com/questions/2763006/make-the-current-git-branch-a-master-branch) or [this](https://help.github.com/en/github/administering-a-repository/setting-the-default-branch).

#### Workflow, how to train, etc
- Place all the config files in `./configs` folder.
- Experiments are logged to the `./experiments` folder. Each config file can (optionally) have an experiment name. The corresponding experiments are logged under the experiment name.
- Further, within each experiment folder, the logs are separated into separate subfolders
  - `saved_models` - holds checkpoint files (model, optimizers, configs, etc)
  - `stdout.log` - dump of iteration logs in json format
  - `tb_logs` - logs for tensorboard. Point the tensorboard to this folder to view them.
- Start training using the `train_model.py` script in the `src` folder
Eg.,
```
python train_model.py --cfg ../configs/resnet18/resnet18_baseline.yml
```



