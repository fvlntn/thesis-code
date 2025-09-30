import argparse
import sys
import time

import torch
from monai.utils import set_determinism

from seg_model import getUNetForExtraction
from transforms_dict import getSegmentationEvalTransformsForMRI, getSegmentationPostProcessingForMaskOutput
from transforms_dict import getSegmentationInverseTransform, SaveTransformForMRI
from utils import getDevice, loadExistingModel

set_determinism(seed=0)

def main(modelname, mriname, outname, N4, resample, save_tmp, get_largest_component):
    
    device = getDevice()
    atlas_name = "dataset3/Atlas/P56_Atlas_128_norm_id.nii.gz"
    imtrans = getSegmentationEvalTransformsForMRI(N4=N4, atlas_name=atlas_name, resample=resample, outname=outname, save=save_tmp)
    inverse_transform = getSegmentationInverseTransform(mriname, atlas_name=atlas_name, resample=resample, outname=outname, save=save_tmp)
    post_im_trans = getSegmentationPostProcessingForMaskOutput()
    model = getUNetForExtraction()
    loadExistingModel(model, None, ft=modelname)
    model.eval()
    
    #start = time.process_time()
    with torch.no_grad():
        input_mri = imtrans(mriname)
        input_mri = input_mri.to(device, non_blocking=True)
        
        output = model(input_mri)
        output = output.squeeze().cpu()

        #if save_tmp:
        #    mri_save_transform = SaveTransformForMRI(outname + "_Resample_Mask.nii.gz", outname + "_Resample.nii.gz",
        #                                             getLargestComponent=get_largest_component,
        #                                             doSave=save_tmp)
        #    mri_save_transform(post_im_trans(output))

        output_resized = inverse_transform(output)
        mri_save_transform_inv = SaveTransformForMRI(outname + "_MRI_Mask.nii.gz", mriname, getLargestComponent=get_largest_component)
        mri_save_transform_inv(output_resized) 
        #print(time.process_time() - start)
    

    return output_resized


def parseArguments():
    parser = argparse.ArgumentParser(description="Use 3D mouse brain segmentation model on MRI.")
    parser.add_argument("-m", "--modelpath", help="Specify model path.")
    parser.add_argument("-i", "--input", help="MRI input")
    parser.add_argument("-o", "--output", help="MRI output")
    parser.add_argument("-N4", "--N4", action='store_true', help="Apply N4 to MRI before segmentation")
    parser.add_argument("-tmp", "--temp", action='store_true', help="Save temp files.")
    parser.add_argument("-g", "--get", action='store_true', help="Get largest component before saving.")
    parser.add_argument("-r", "--resample", action='store_true', help="Apply Resampling to MRI before segmentation")
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

    main(args.modelpath, args.input, args.output, args.N4, args.resample, args.temp, args.get)
