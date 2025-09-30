import logging
import os
import sys
from functools import partial

import monai
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from seg_data import getSegmentationIRISDataset
from seg_model import getUNetForExtraction
from transforms_dict import getSegmentationPostProcessingForMaskOutput, getSegmentationPostProcessingForMask
from utils import compute_mean_dice, getAdamOptimizer, getReducePlateauScheduler, getDevice


def train(config, checkpoint_dir=None, data_dir=None):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    criterion = monai.losses.DiceLoss(sigmoid=True)
    post_im_trans = getSegmentationPostProcessingForMaskOutput()
    post_seg_trans = getSegmentationPostProcessingForMask()
    dataloaders, size = getSegmentationIRISDataset(batch=1, augment=True, n4=False)
    model = getUNetForExtraction()
    optimizer = getAdamOptimizer(model, config["lr"])
    scheduler = getReducePlateauScheduler(optimizer)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    num_epochs = 500

    for epoch in range(num_epochs):

        valid_loss = -1
        valid_metric = -1

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metric = 0.0

            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data["img"].to(device, non_blocking=True), data["seg"].to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    metric = compute_mean_dice(post_im_trans(outputs), post_seg_trans(labels))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_metric += metric.item() * inputs.size(0)

            epoch_loss = running_loss / size[phase]
            epoch_metric = running_metric / size[phase]

            if phase == 'valid':
                valid_loss = epoch_loss
                valid_metric = epoch_metric
                scheduler.step(epoch_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=valid_loss, accuracy=valid_metric)


def main(num_samples, max_num_epochs):
    data_dir = os.path.abspath("/home/valentini/dev/Mousenet/")
    config = {
        "lr": tune.loguniform(1e-6, 1e-2),
    }
    algo = TuneBOHB(metric="loss", mode="min")
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=algo,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    main(num_samples=2000, max_num_epochs=50)
