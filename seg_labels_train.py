import argparse
import logging
import sys

import monai
import torch
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

import utils_parser
from seg_data import getSegmentationDataset
from seg_model import getUNetForSegmentation, getUNETRForSegmentation
from transforms_dict import getSegmentationPostProcessingForLabelOutput, getSegmentationPostProcessingForLabel, getSegmentationPostProcessingForAllLabelsOutput
from utils import compute_mean_dice, getReducePlateauScheduler, getAdamOptimizer, loadExistingModel
from utils import print_model_output, check_model_name, getDevice


def train(modelname, dataset, ft, ct, batchsize, num_epochs, lr, patience, augment, N4=False):
    torch.multiprocessing.set_sharing_strategy('file_system')
    modelname = check_model_name(modelname)
    print_model_output(modelname)
    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    loss_function = monai.losses.GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)

    post_im_trans = getSegmentationPostProcessingForLabelOutput(axis=1)
    post_seg_trans = getSegmentationPostProcessingForLabel(axis=1)

    #model = getUNetForSegmentation()
    model = getUNETRForSegmentation()
    optimizer = getAdamOptimizer(model, lr)
    scheduler = getReducePlateauScheduler(optimizer, patience=patience)
    loadExistingModel(model, optimizer, ft, ct)

    dataloaders, size = getSegmentationDataset(dataset=dataset,
                                               batch=batchsize,
                                               augment=augment,
                                               training=True,
                                               n4=N4,
                                               eval_augment=False,
                                               labels=True,
                                               )

    best_metric = -1

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")

        train_loss = 0
        valid_loss = 0
        train_metric1 = 0
        valid_metric1 = 0
        train_metric2 = 0
        valid_metric2 = 0
        train_metric3 = 0
        valid_metric3 = 0
        train_metric4 = 0
        valid_metric4 = 0

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metric1 = 0.0
            running_metric2 = 0.0
            running_metric3 = 0.0
            running_metric4 = 0.0

            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    inputs, labels = data["img"].to(device, non_blocking=True), data["seg"].to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    metric = compute_mean_dice(post_im_trans(outputs).squeeze(), post_seg_trans(labels).squeeze())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_metric1 += metric[0].item() * inputs.size(0)
                running_metric2 += metric[1].item() * inputs.size(0)
                running_metric3 += metric[2].item() * inputs.size(0)
                running_metric4 += metric[3].item() * inputs.size(0)

            running_loss /= size[phase]
            running_metric1 /= size[phase]
            running_metric2 /= size[phase]
            running_metric3 /= size[phase]
            running_metric4 /= size[phase]

            if phase == 'train':
                train_loss = running_loss
                train_metric1 = running_metric1
                train_metric2 = running_metric2
                train_metric3 = running_metric3
                train_metric4 = running_metric4
            elif phase == 'valid':
                valid_loss = running_loss
                valid_metric1 = running_metric1
                valid_metric2 = running_metric2
                valid_metric3 = running_metric3
                valid_metric4 = running_metric4

            print(
                "{}: loss : {:.4f}, dice1: {:.4f}, dice2: {:.4f}, dice3: {:.4f}, dice4: {:.4f}".format(
                    phase, running_loss, running_metric1, running_metric2, running_metric3, running_metric4
                )
            )

            if phase == 'valid':
                scheduler.step(running_loss)
                if running_metric2 > best_metric:
                    best_metric = running_metric2
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
        writer.add_scalars('epoch_metric1', {
            'train': train_metric1,
            'valid': valid_metric1,
        }, epoch + 1)
        writer.add_scalars('epoch_metric2', {
            'train': train_metric2,
            'valid': valid_metric2,
        }, epoch + 1)
        writer.add_scalars('epoch_metric3', {
            'train': train_metric3,
            'valid': valid_metric3,
        }, epoch + 1)
        writer.add_scalars('epoch_metric4', {
            'train': train_metric4,
            'valid': valid_metric4,
        }, epoch + 1)

    print(f"train completed")
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train 3D mouse brain segmentation model.")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.003, help="Specify learning rate.")
    parser.add_argument("-p", "--patience", type=int, default=20, help="Specify patience for LR Plateau.")
    parser.add_argument("-a", "--augment", action='store_true', help="Use augmentation or not.")
    parser.add_argument("-b", "--batchsize", type=int, default=4, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Max epochs for training")
    parser.add_argument("-o", "--output", help="Model name for save")
    parser.add_argument("-d", "--dataset", help="Dataset name for training.")
    parser.add_argument("-N4", "--N4", action='store_true', help="Use N4 corrected dataset if available")
    parser.add_argument("-ft", "--finetuning", help="Load existing model for finetuning.")
    parser.add_argument("-ct", "--continuetraining", help="Load existing model to continue training.")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    utils_parser.handleModelOutputParser(args.output)
    utils_parser.handleDatasetParser(args.dataset)
    utils_parser.handleFinetuningParser(args.finetuning, args.continuetraining)

    train(args.output, args.dataset, args.finetuning, args.continuetraining, args.batchsize, args.epochs,
          args.learningrate, args.patience,
          args.augment, args.N4)