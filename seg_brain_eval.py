import argparse
import logging
import sys

import monai
import torch

from seg_data import getSegmentationDataset
from seg_model import getUNetForExtraction
from transforms_dict import getSegmentationPostProcessingForMask, getSegmentationPostProcessingForMaskOutput
from utils import getWorst, compute_mean_dice, getDevice, loadExistingModel


def eval(dataset, modelname, phase='test', number=5, verbose=True, augment=False, n4=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    criterion = monai.losses.DiceLoss(sigmoid=True)

    post_im_trans = getSegmentationPostProcessingForMaskOutput()
    post_seg_trans = getSegmentationPostProcessingForMask()

    model = getUNetForExtraction()
    loadExistingModel(model, None, ft=modelname)
    model.eval()

    dataloaders, size = getSegmentationDataset(dataset=dataset, batch=1, training=False, augment=augment, n4=n4)

    scores_list = []

    with torch.no_grad():
        running_loss = 0.0
        running_metric = 0.0
        for i, data in enumerate(dataloaders[phase]):
            inputs, labels = data["img"].to(device), data["seg"].to(device)
            filename = data["img_file"][0].split('/')[-1]  # TODO : use data['meta'] ...
            maskfilename = data["seg_file"][0].split('/')[-1]  # TODO: use data['meta'] ...

            outputs = model(inputs)
            loss = criterion(outputs, post_seg_trans(labels))
            metric = compute_mean_dice(post_im_trans(outputs), post_seg_trans(labels))

            scores_list.append([filename, loss.item(), metric.item()])

            running_loss += loss.item() * inputs.size(0)
            running_metric += metric.item() * inputs.size(0)

            if verbose:
                print(
                    "{} - {} : loss : {:.4f}, dice: {:.4f}".format(
                        filename, maskfilename, loss.item(), metric.item(),
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
    parser.add_argument("-a", "--augment", action='store_true', help="Use augmented transforms or not.")
    parser.add_argument("-n4", "--n4", action='store_true', help="Use N4 corrected dataset if available")
    parser.add_argument("-d", "--dataset", help="Specify dataset (either IRIS, NEAT-EX, NEAT-IN or CERMEP)")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    if not args.modelpath:
        print("Should specify model path. See -h for instructions.")
        sys.exit(1)

    if args.phase not in ['train', 'valid', 'test']:
        print("Should specify phase. See -h for instructions.")
        sys.exit(1)

    if args.dataset is None:
        print("Should specify dataset. See -h for instructions.")
        sys.exit(1)

    if 'iris' not in args.dataset.lower():
        args.n4 = False

    eval(args.dataset, args.modelpath, args.phase, args.number, args.verbose, args.augment, args.n4)
