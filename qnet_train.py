import argparse
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from qnet_data import getMaskDataset
from qnet_model import getQnet
from transforms_dict import getSegmentationPostProcessingForLabel, getSegmentationPostProcessingForMaskOutput
from utils import QnetLoss, check_model_name, compute_mean_dice, getAdamOptimizer, getReducePlateauScheduler
from utils import print_model_output, loadExistingModel, getDevice


def train(modelname, ft, ct, batch, num_epochs, lr, patience, augment):
    modelname = check_model_name(modelname)
    print_model_output(modelname)
    device = getDevice()

    criterion = QnetLoss()
    post_predmask_trans = getSegmentationPostProcessingForMaskOutput()
    post_mask_trans = getSegmentationPostProcessingForLabel()

    model = getQnet()
    optimizer = getAdamOptimizer(model, lr=lr)
    scheduler = getReducePlateauScheduler(optimizer, patience=patience)
    loadExistingModel(model, optimizer, ft, ct)

    dataloaders, size = getMaskDataset(batch, augment)

    best_loss = np.inf
    best_epoch = -1

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")

        train_loss = 0
        valid_loss = 0
        train_metric = 0
        valid_metric = 0

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metric = 0.0
            running_mask = 0.0
            running_dice = 0.0
            running_sens = 0.0
            running_spec = 0.0

            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')
                mask = data[0].to(device, non_blocking=True)
                dice = data[1].to(device, non_blocking=True)
                sens = data[2].to(device, non_blocking=True)
                spec = data[3].to(device, non_blocking=True)
                target = data[4].to(device, non_blocking=True)

                inputs = mask
                scores = []
                for j in range(list(dice.shape)[0]):
                    scores.append([dice[j], sens[j], spec[j]])
                scores = torch.FloatTensor(scores).to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output_mask, output_scores = model(inputs)
                    loss, maskloss, diceloss, sensloss, specloss = criterion(output_mask, target, output_scores, scores)
                    maskloss = maskloss * 7
                    metric = compute_mean_dice(post_predmask_trans(output_mask), post_mask_trans(target))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_metric += metric.item() * inputs.size(0)
                running_mask += maskloss.item() * inputs.size(0)
                running_dice += diceloss.item() * inputs.size(0)
                running_sens += sensloss.item() * inputs.size(0)
                running_spec += specloss.item() * inputs.size(0)

            running_loss /= size[phase]
            running_metric /= size[phase]
            running_mask /= size[phase]
            running_dice /= size[phase]
            running_sens /= size[phase]
            running_spec /= size[phase]

            if phase == 'train':
                train_loss = running_loss
                train_metric = running_metric
            elif phase == 'valid':
                valid_loss = running_loss
                valid_metric = running_metric

            print(
                "{}: loss: {:.4f} - dice: {:.4f} -- mask: {:.4f}, dice: {:.4f}, sens: {:.4f}, spec: {:.4f}".format(
                    phase, running_loss, running_metric, running_mask, running_dice, running_sens, running_spec,
                )
            )

            if phase == 'valid':
                scheduler.step(running_loss)
                if running_loss < best_loss:
                    best_loss = running_loss
                    best_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                        './models/' + modelname)

                print(
                    "best loss {:.4f} at epoch {}".format(
                        best_loss, best_epoch
                    )
                )

        writer.add_scalars('epoch_loss', {
            'train': train_loss,
            'valid': valid_loss,
        }, epoch + 1)
        writer.add_scalars('epoch_metric', {
            'train': train_metric,
            'valid': valid_metric,
        }, epoch + 1)

    print(f"train completed")
    writer.close()


def parseArguments():
    parser = argparse.ArgumentParser(description="Train model to enhance existing mouse brain masks")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.0001, help="Specify learning rate.")
    parser.add_argument("-p", "--patience", type=int, default=20, help="Specify patience for LR Plateau.")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Max epochs for training")
    parser.add_argument("-o", "--output", help="Model name for save")
    parser.add_argument("-a", "--augment", action='store_true', help="Use augmentation or not.")
    parser.add_argument("-ft", "--finetuning", help="Load existing model for finetuning.")
    parser.add_argument("-ct", "--continuetraining", help="Load existing model to continue training.")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    if args.finetuning is not None and args.continuetraining is None:
        print("=> Loading " + str(args.finetuning) + " for finetuning")

    if args.finetuning is None and args.continuetraining is not None:
        print("=> Loading " + str(args.continuetraining) + " to continue training")

    if args.output is None:
        print("Should specify model name output. See -h for instructions.")
        sys.exit(1)

    train(args.output, args.loadmodel, args.batchsize, args.epochs, args.learningrate, args.patience, args.augment)
