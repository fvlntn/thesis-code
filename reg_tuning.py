import logging
import logging
import os
import sys
from functools import partial

import numpy as np
import torch
from monai.transforms import AsDiscrete, MaskIntensity
from monai.utils import set_determinism
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from loss import get_deformable_registration_loss_from_weights, get_affine_registration_loss_from_weights
from reg_data import getRegistrationDataset
from reg_model import getRegistrationModel
from utils import getAdamOptimizer, getReducePlateauScheduler, getDevice
from utils import print_weights, compute_landmarks_distance_local


def train(config, checkpoint_dir=None, data_dir=None):

    lr=config["lr"]
    patience=config["patience"]
    weights = [1, 0, config["weight"]*0.1]
    print_weights(weights)
    atlas=True
    mask=False
    dataset="feminadaffine"
    batchsize=1
    max_epochs=500
    registration_type="local"
    ft=None
    ct=None
    pt=None
    channels=config["channels"]

    torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()


    model = getRegistrationModel(registration_type, img_size=128, pretrain_model=pt, channels=channels)
    optimizer = getAdamOptimizer(model, lr)
    scheduler = getReducePlateauScheduler(optimizer, patience=patience)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


    dataloaders, size = getRegistrationDataset(dataset=dataset,
                                               batch=batchsize,
                                               training=True,
                                               augment=True,
                                               eval_augment=False,
                                               atlas=atlas,
                                               mask=mask,
                                               )

    best_metric = np.inf
    best_epoch = -1


    for epoch in range(max_epochs):
        train_loss = 0
        train_metric = 0
        train_lbl_loss = 0
        train_img_loss = 0
        train_ddf_loss = 0

        valid_loss = 0
        valid_metric = 0
        valid_lbl_loss = 0
        valid_img_loss = 0
        valid_ddf_loss = 0

        for phase in ['train', 'valid']:
            if pt is None:
                if phase == 'train':
                    model.train()
                elif phase == 'valid':
                    model.eval()
            else:
                for param in model.globalnet.parameters():
                    param.requires_grad = False
                if phase == 'train':
                    model.globalnet.eval()
                    model.localnet.train()
                elif phase == 'valid':
                    model.globalnet.eval()
                    model.localnet.eval()

            running_loss = 0.0
            running_metric = 0.0
            running_img_loss = 0.0
            running_lbl_loss = 0.0
            running_ddf_loss = 0.0

            for i, data in enumerate(dataloaders[phase]):
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    if registration_type.lower() == 'affine' or registration_type.lower() == 'local':
                        ddf, pred_image, pred_label = model(data)
                    elif registration_type.lower() == 'deformable':
                        affine_ddf, ddf, pred_image, pred_label, affine_image, affine_label = model(data)

                    pred_image = pred_image.to(device, non_blocking=True)
                    pred_label = pred_label.to(device, non_blocking=True)
                    pred_mask = AsDiscrete(threshold=0.5)(pred_label)
                    pred_image_masked = MaskIntensity(mask_data=pred_mask)(pred_image)

                    fixed_image = data['fixed_image'].to(device, non_blocking=True)
                    fixed_label = data['fixed_label'].to(device, non_blocking=True)
                    fixed_mask = AsDiscrete(threshold=0.5)(fixed_label)
                    fixed_image_masked = MaskIntensity(mask_data=fixed_mask)(fixed_image)

                    if registration_type.lower() == 'affine':
                        img_loss, lbl_loss, ddf_loss = get_affine_registration_loss_from_weights(pred_image_masked, pred_label,
                                                                                                 fixed_image_masked, fixed_label,
                                                                                                 weights)
                        loss = img_loss + lbl_loss
                    elif registration_type.lower() == 'deformable' or registration_type.lower() == 'local':
                        img_loss, lbl_loss, ddf_loss = get_deformable_registration_loss_from_weights(pred_image, pred_label,
                                                                                                     fixed_image, fixed_label,
                                                                                                     ddf, weights)
                        loss = img_loss + lbl_loss + ddf_loss

                    if registration_type.lower() == 'local' and phase == 'valid':
                        metric = compute_landmarks_distance_local(ddf, data)
                    else:
                        metric = torch.zeros(1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * fixed_image.size(0)
                running_metric += metric.item() * fixed_image.size(0)
                running_img_loss += img_loss.item() * fixed_image.size(0)
                running_lbl_loss += lbl_loss.item() * fixed_image.size(0)
                running_ddf_loss += ddf_loss.item() * fixed_image.size(0)

            running_loss /= size[phase]
            running_metric /= size[phase]
            running_img_loss /= size[phase]
            running_lbl_loss /= size[phase]
            running_ddf_loss /= size[phase]

            if phase == 'train':
                train_loss = running_loss
                train_metric = running_metric
                train_img_loss = running_img_loss
                train_lbl_loss = running_lbl_loss
                train_ddf_loss = running_ddf_loss
            elif phase == 'valid':
                valid_loss = running_loss
                valid_metric = running_metric
                valid_img_loss = running_img_loss
                valid_lbl_loss = running_lbl_loss
                valid_ddf_loss = running_ddf_loss

            if phase == 'valid':
                scheduler.step(running_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=valid_loss, landmarks=valid_metric)

def main(num_samples, max_num_epochs):
    data_dir = os.path.abspath("/home/valentini/dev/Mousenet/")
    config = {
        "lr": tune.loguniform(1e-6, 1e-3),
        "patience": tune.choice([20]),
        "channels": tune.choice([8,16,32]),
        "weight": tune.randint(1,1000),
    }
    algo = TuneBOHB(metric="landmarks", mode="min")
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="landmarks",
        mode="min",
        max_t=max_num_epochs)
    reporter = CLIReporter(
        metric_columns=["loss", "landmarks", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 8, "gpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=algo,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("landmarks", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    main(num_samples=2000, max_num_epochs=100)
