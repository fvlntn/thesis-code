import argparse
import logging
import sys

import monai
import torch
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

import utils_parser
from seg_data import getSegmentationDataset
from seg_model import getUNetForExtraction
from transforms_dict import getSegmentationPostProcessingForMaskOutput, getSegmentationPostProcessingForMask
from utils import compute_mean_dice, getReducePlateauScheduler, getAdamOptimizer, loadExistingModel
from utils import print_model_output, check_model_name, getDevice


def train(modelname, dataset, ft, ct, batchsize, num_epochs, lr, patience, N4=False, resample=False, labels=False):
    torch.multiprocessing.set_sharing_strategy('file_system')
    modelname = check_model_name(modelname)
    print_model_output(modelname)
    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    MaskOutputPostprocessing = getSegmentationPostProcessingForMaskOutput()
    MaskPostprocessing = getSegmentationPostProcessingForMask()

    model = getUNetForExtraction()
    optimizer = getAdamOptimizer(model, lr)
    scheduler = getReducePlateauScheduler(optimizer, patience=patience)
    loadExistingModel(model, optimizer, ft, ct)

    dataloaders, size = getSegmentationDataset(dataset=dataset,
                                               batch=batchsize,
                                               augment=True,
                                               eval_augment=False,
                                               training=True,
                                               n4=N4,
                                               resample=resample,
                                               labels=False,
                                               )

    best_metric = -1

    writer = SummaryWriter(comment='_'+modelname)

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

            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    inputs = data["img"].to(device, non_blocking=True)
                    labels = data["seg"].to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    metric = compute_mean_dice(MaskOutputPostprocessing(outputs), MaskPostprocessing(labels))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_metric += metric.item() * inputs.size(0)

            running_loss /= size[phase]
            running_metric /= size[phase]

            if phase == 'train':
                train_loss = running_loss
                train_metric = running_metric
            elif phase == 'valid':
                valid_loss = running_loss
                valid_metric = running_metric

            print(
                "{}: loss : {:.4f}, dice: {:.4f}".format(
                    phase, running_loss, running_metric,
                )
            )

            if phase == 'valid':
                scheduler.step(running_loss)
                if running_metric > best_metric:
                    best_metric = running_metric
                    best_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                        './models/' + str(modelname))

                    print(
                        "best dice {:.4f} at epoch {}".format(
                            best_metric, best_epoch
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
    print(f"train completed, "
          f"best_dice: {best_metric:.4f}  "
          f"at epoch: {best_epoch}")
    writer.close()


def parseArguments():
    parser = argparse.ArgumentParser(description="Train 3D mouse brain segmentation model.")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.003, help="Specify learning rate.")
    parser.add_argument("-p", "--patience", type=int, default=20, help="Specify patience for LR Plateau.")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Max epochs for training")
    parser.add_argument("-o", "--output", help="Model name for save")
    parser.add_argument("-d", "--dataset", help="Dataset name for training. IRIS/IRISResample/IRISCrop/CERMEP/Neat")
    parser.add_argument("-N4", "--N4", action='store_true', help="Use N4 corrected dataset if available")
    parser.add_argument("-r", "--resample", action='store_true', help="Use Resampled dataset if available")
    parser.add_argument("-l", "--labels", action='store_true', help="Multi-labels segmentation if available. NYI") # TODO
    parser.add_argument("-ft", "--finetuning", help="Load existing model for finetuning.")
    parser.add_argument("-ct", "--continuetraining", help="Load existing model to continue training.")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    utils_parser.handleModelOutputParser(args.output)
    utils_parser.handleDatasetParser(args.dataset)
    utils_parser.handleFinetuningParser(args.finetuning, args.continuetraining)

    train(args.output, args.dataset, args.finetuning, args.continuetraining, args.batchsize, args.epochs,
          args.learningrate, args.patience, args.N4, args.resample, args.labels)
