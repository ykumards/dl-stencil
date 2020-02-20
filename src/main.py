import os
from datetime import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torchvision.utils import save_image, make_grid

from ignite.engine import Engine, Events
from ignite.metrics import (Loss, RunningAverage)
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler

try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found")

from models.vae import VAE
from data import setup_cifar10_dataloaders as setup_dataloaders
import utils

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
warnings.simplefilter("ignore")


def main(args, smoke_test=False, use_visdom=False):
    if smoke_test:
        NUM_EPOCHS = 1
        use_visdom = False
    else:
        NUM_EPOCHS = args.num_epochs

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    visdom_env_name = args.experiment_name + timestamp
    utils.seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    if use_visdom:
        vis = visdom.Visdom(env=visdom_env_name)

        if not vis.check_connection():
            raise RuntimeError("Visdom server not running.",
                               "Please run python -m visdom.server")

        lr_window = utils.create_plot_window(
            vis, '# Epoch', 'Learning Rate', 'Learning Rate')

        train_avg_loss_window = utils.create_plot_window(
            vis, '# Epoch', 'Loss', 'Training Average Loss')
        train_avg_reconc_loss_window = utils.create_plot_window(
            vis, '# Epoch', 'reconc_loss', 'Training Average reconc_loss')
        train_avg_kld_window = utils.create_plot_window(
            vis, '# Epoch', 'KLD', 'Training Average KLDiv')
        # train_avg_mse_window = utils.create_plot_window(
        #     vis, '# Epoch', 'MSE', 'Training Average MSE')

        valid_avg_loss_window = utils.create_plot_window(
            vis, '# Epoch', 'Loss', 'Validation Average Loss')
        valid_avg_reconc_loss_window = utils.create_plot_window(
            vis, '# Epoch', 'reconc_loss', 'Validation Average reconc_loss')
        valid_avg_kld_window = utils.create_plot_window(
            vis, '# Epoch', 'KLD', 'Validation Average KLDiv')
        # valid_avg_mse_window = utils.create_plot_window(
        #     vis, '# Epoch', 'MSE', 'Validation Average MSE')

    train_loader, test_loader = setup_dataloaders(
        batch_size=args.batch_size
    )

    for batch in train_loader:
        x, y = batch
        break

    print('x.shape : ', x.shape)
    print('y.shape : ', y.shape)
    fixed_images = x.to(device)

    model = VAE(
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        is_mse_loss=args.is_mse_loss,
        device=device
    )
    print(
        "Encoder parameters:",
        sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    )
    print(
        "Decoder parameters:",
        sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    )
    print()

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    def lr_step(engine):
        lr_scheduler.step()
        if use_visdom:
            vis.line(X=np.array([engine.state.epoch]),
                     Y=np.array([lr_scheduler.get_lr()]),
                     update='append', win=lr_window)

    if args.is_mse_loss:
        reconc_loss = nn.MSELoss(reduction='sum')
    else:
        reconc_loss = nn.CrossEntropyLoss(reduction='sum')

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, _ = batch
        x = x.to(device)
        x_pred, mu, logvar = model(x)
        RECONC = reconc_loss(x_pred, x)
        KLD = utils.kld_loss(mu, logvar)
        loss = RECONC + KLD
        loss.backward()
        optimizer.step()

        return loss, RECONC.item(), KLD.item()

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, _ = batch
            x = x.to(device)
            x_pred, mu, logvar = model(x)
            return x_pred, x, mu, logvar

    trainer = Engine(process_function)
    train_evaluator = Engine(evaluate_function)
    evaluator = Engine(evaluate_function)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_step)

    if args.enable_time_profiling:
        profiler = BasicTimeProfiler()
        profiler.attach(trainer)

    pbar = ProgressBar(persist=False)
    # define trainer metrics
    train_metrics = {
        "total_loss": RunningAverage(output_transform=lambda x: x[0]),
        "reconc_loss": RunningAverage(output_transform=lambda x: x[1]),
        "kld": RunningAverage(output_transform=lambda x: x[2]),
    }
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)
    pbar.attach(trainer, list(train_metrics.keys()))

    # define evaluator metrics
    eval_metrics = {
        "reconc_loss": Loss(
            reconc_loss, output_transform=lambda x: [x[0], x[1]]),
        "kld": Loss(utils.kld_loss, output_transform=lambda x: [x[2], x[3]])
    }
    eval_metrics['total_loss'] = sum(eval_metrics.values())
    for eval_engine in [evaluator, train_evaluator]:
        for name, metric in eval_metrics.items():
            metric.attach(eval_engine, name)

    def print_logs(engine, dataloader, mode):
        eval_engine = evaluator if mode == "Training" else train_evaluator
        eval_engine.run(dataloader, max_epochs=1)
        metrics = evaluator.state.metrics
        avg_reconc_loss = metrics['reconc_loss']
        avg_kld = metrics['kld']
        avg_loss = avg_reconc_loss + avg_kld
        print_str = f"{mode} Results - Epoch {engine.state.epoch} - " +\
                    f" Avg loss: {avg_loss:.2f}" +\
                    f" Avg reconc_loss: {avg_reconc_loss:.2f}" +\
                    f" Avg kld: {avg_kld:.2f}"
        pbar.log_message(print_str)

        if use_visdom:
            if mode == 'Training':
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_loss]),
                         update='append', win=train_avg_loss_window)
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_reconc_loss]),
                         update='append', win=train_avg_reconc_loss_window)
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_kld]),
                         update='append', win=train_avg_kld_window)
                # vis.line(X=np.array([engine.state.epoch]),
                #          Y=np.array([avg_mse]),
                #          update='append', win=train_avg_mse_window)

            if mode == 'Validate':
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_loss]),
                         update='append', win=valid_avg_loss_window)
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_reconc_loss]),
                         update='append', win=valid_avg_reconc_loss_window)
                vis.line(X=np.array([engine.state.epoch]),
                         Y=np.array([avg_kld]),
                         update='append', win=valid_avg_kld_window)
                # vis.line(X=np.array([engine.state.epoch]),
                #          Y=np.array([avg_mse]),
                #          update='append', win=valid_avg_mse_window)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        print_logs,
        train_loader,
        'Training'
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        print_logs,
        test_loader,
        'Validate'
    )
    ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator)
    ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

    common.save_best_model_by_val_score(
        args.model_output_dir,
        evaluator,
        model=model,
        metric_name='total_loss',
        n_saved=3,
        trainer=trainer,
        tag="val",
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=args.plot_image_every))
    def compare_images(engine, save_img=True):
        epoch = engine.state.epoch
        reconstructed_images = model(fixed_images)[0]
        comparison = torch.cat([fixed_images, reconstructed_images])
        if save_img:
            save_image(
                comparison.detach().cpu(),
                args.output_img_dir + str(epoch) + '.png',
                nrow=8
            )
        comparison_image = make_grid(comparison.detach().cpu(), nrow=8)
        # TODO images are hazy
        # inv_normalize = transforms.Normalize(
        #     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #     std=[1/0.229, 1/0.224, 1/0.225]
        # )
        # inv_tensor = inv_normalize(tensor)
        unnorm = utils.UnNormalizeImage(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        comparison_image = unnorm(comparison_image)

        vis.image(
            comparison_image,
            opts=dict(caption=f"Reconstruciton after Epoch # {epoch}")
        )

    trainer.run(train_loader, max_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    args = utils.load_config('config/default.yml')
    main(args, smoke_test=False, use_visdom=True)
