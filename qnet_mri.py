import argparse
import sys

import numpy as np
import torch
from monai.transforms import LoadImage, AddChannel, Compose, Resize, EnsureType

from qnet_model import getQnet
from transforms_dict import getSegmentationPostProcessingForMaskOutput, getSegmentationInverseTransform
from utils import NiftiSaver, getDevice, loadExistingModel


def main(modelname, mriname, outname):
    device = getDevice()
    imtrans = Compose([
        AddChannel(),
        Resize((128, 128, 128), mode="trilinear"),
        EnsureType(),
        AddChannel()
    ])
    post_im_trans = getSegmentationPostProcessingForMaskOutput()

    model = getQnet()
    loadExistingModel(model, None, ft=modelname)
    model.eval()

    mri_loader = LoadImage(
        reader="NibabelReader",
        image_only=True,
        dtype=np.float32,
        as_closest_canonical=True,
    )

    input_mri = mri_loader(mriname)
    inverse_transform = getSegmentationInverseTransform(mriname)

    saver = NiftiSaver()

    with torch.no_grad():
        input_mri = imtrans(input_mri)
        input_mri = input_mri.to(device, non_blocking=True)

        output, _ = model(input_mri)
        output = post_im_trans(output)
        output = output.squeeze().cpu()
        output_resized = inverse_transform(output)
        output_resized[output_resized > 0] = 1
        output_resized[output_resized != 1] = 0
        saver.save(output_resized, outname)
        print('=> Mask file saved to ' + str(outname))


def parseArguments():
    parser = argparse.ArgumentParser(description="Use model to enhance existing mouse brain masks on Mask")
    parser.add_argument("-m", "--modelpath", help="Specify model path.")
    parser.add_argument("-i", "--input", help="MRI input")
    parser.add_argument("-o", "--output", help="MRI output")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    if not args.modelpath:
        print("Should specify model path. See -h for instructions.")
        sys.exit(1)

    if not args.input:
        print("Should specify input MRI. See -h for instructions.")
        sys.exit(1)

    if not args.output:
        print("Should specify output MRI. See -h for instructions.")
        sys.exit(1)

    main(args.modelpath, args.input, args.output)
