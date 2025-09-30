import argparse
import logging
import sys

import torch

from qnet_data import getMaskDataset
from qnet_model import getQnet
from transforms_dict import getSegmentationPostProcessingForMaskOutput, getSegmentationPostProcessingForLabel
from utils import getWorst, QnetLoss, compute_mean_dice, getDevice, loadExistingModel


def main(modelname, phase='test', number=5, verbose=True):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    criterion = QnetLoss()
    post_predmask_trans = getSegmentationPostProcessingForMaskOutput()
    post_mask_trans = getSegmentationPostProcessingForLabel()

    model = getQnet()
    loadExistingModel(model, None, ft=modelname)
    model.eval()

    dataloaders, size = getMaskDataset(batch=1)

    scores_list = []

    with torch.no_grad():
        running_loss = 0.0
        running_metric = 0.0
        for i, data in enumerate(dataloaders[phase]):
            print(i, end='\r')
            mask = data[0].to(device, non_blocking=True)
            dice = data[1].to(device, non_blocking=True)
            sens = data[2].to(device, non_blocking=True)
            spec = data[3].to(device, non_blocking=True)
            target = data[4].to(device, non_blocking=True)
            maskfilename = data[5][0].split('/Mask/')[1]
            truthfilename = data[6][0].split('/Mask/')[1]

            inputs = mask
            scores = []
            for j in range(list(dice.shape)[0]):
                scores.append([dice[j], sens[j], spec[j]])
            scores = torch.FloatTensor(scores).to(device, non_blocking=True)

            output_mask, output_scores = model(inputs)
            loss, maskloss, diceloss, sensloss, specloss = criterion(output_mask, target, output_scores, scores)
            metric = compute_mean_dice(post_predmask_trans(output_mask), post_mask_trans(target))

            scores_list.append([maskfilename, loss.item(), metric.item(),
                                maskloss.item(), diceloss.item(), sensloss.item(), specloss.item()])

            running_loss += loss.item() * inputs.size(0)
            running_metric += metric.item() * inputs.size(0)

            if verbose:
                print(
                    "{} - {} : loss : {:.4f}, dice: {:.4f}".format(
                        maskfilename, truthfilename, loss.item(), metric.item(),
                    )
                )

        epoch_loss = running_loss / size[phase]
        epoch_metric = running_metric / size[phase]

        print(
            "mean: loss : {:.8f}, dice: {:.8f}".format(
                epoch_loss, epoch_metric,
            )
        )
    getWorst(scores_list, number)


def parseArguments():
    parser = argparse.ArgumentParser(description="Evaluate 3D mouse brain segmentation model.")
    parser.add_argument("-m", "--modelpath", help="Specify model path.")
    parser.add_argument("-p", "--phase", help="Specify data phase (train/valid/test).")
    parser.add_argument("-n", "--number", default="5", help="Number of worst segmentations to output.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose or not.")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    if not args.modelpath:
        print("Should specify model path. See -h for instructions.")
        sys.exit(1)

    if not args.phase:
        print("Should specify set (train / val / test). See -h for instructions.")
        sys.exit(1)

    main(args.modelpath, args.phase, args.number, args.verbose)
