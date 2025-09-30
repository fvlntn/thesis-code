import argparse
import sys

import torch
from monai.networks.blocks import Warp
from monai.utils import set_determinism

from reg_model import getRegistrationModel
from transforms_dict import getRegistrationEvalInverseTransformForMRI, SaveTransformForMRI
from transforms_dict import getRegistrationEvalTransformsForMRI, getRegistrationEvalTransformsForAtlas
from utils import getDevice

set_determinism(seed=0)


def main(modelname, mriname, outname, N4, save_tmp, registration_type, newmodel=False, sym=False):
    device = getDevice()

    if "ddf" in modelname:
        use_ddf = True
    else:
        use_ddf = False
    model = getRegistrationModel(registration_type, img_size=128, use_ddf=use_ddf, sym=sym, newmodel=newmodel)
    if sym:
        sym_warp = Warp("bilinear", "border").to(device)
        
    checkpoint = torch.load('./models/' + str(modelname), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    atlas_name = "dataset3/Atlas/P56_Atlas_128_norm_id.nii.gz"

    mri_transform = getRegistrationEvalTransformsForMRI(atlas_name, N4, outname, False)
    mri_save_transform = SaveTransformForMRI(outname + "_Resample_Reg.nii.gz", atlas_name, False)
    mri_inverse_transform = getRegistrationEvalInverseTransformForMRI(mriname, outname + "_Reg.nii.gz", atlas_name, True)
    if sym:
        mri_inverse_transform_sym = getRegistrationEvalInverseTransformForMRI(mriname, outname + "_Reg_Inv.nii.gz", atlas_name, True)

    atlas_transform = getRegistrationEvalTransformsForAtlas()
    #print(mriname)
    mri = mri_transform(mriname)
    atlas = atlas_transform(atlas_name)

    #start = time.process_time()
    with torch.no_grad():
        data = {"fixed_image": atlas, "moving_image": mri}
        if registration_type.lower() == 'null' or registration_type.lower() == 'localzero':
            sym = False
            ddf, pred_image, _, _, _, _ = model(data)
            out = [mri, ddf]   
        elif registration_type.lower() == 'local':
            if sym:
                ddf, pred_image, _, _, ddf2, _ = model(data)   
                out = [mri, ddf, ddf2]
                predsym_image = sym_warp(data["fixed_image"].to(device), ddf2).to(device)
            else:
                ddf, pred_image, _, _, _, _ = model(data)
                out = [mri, ddf]   
        #else:            
        #    if registration_type.lower() == 'affine' or registration_type.lower() == 'local' or registration_type.lower() == 'null':
        #        if sym and registration_type.lower() == 'local':        
        #            ddf, pred_image, _, _, ddf2, _ = model(data)   
        #            out = [data["moving_image"], ddf, ddf2]
        #            predsym_image = sym_warp(data["fixed_image"].to(device), ddf2).to(device)
        #        else:
        #            ddf, pred_image, _, dvf, _, _ = model(data)
        #            out = [data["moving_image"], ddf]            
        #    else:            
        #        affine_ddf, global_ddf, pred_image, _, _, _ = model(data)
        #        out = [data["moving_image"], affine_ddf, global_ddf]
            
        print(pred_image.shape)
        #mri_save_transform(pred_image)
        mri_inverse_transform(pred_image)
        if sym:
            mri_inverse_transform_sym(predsym_image)
    #print(time.process_time() - start)
    if sym:
        return [pred_image, predsym_image], out
    else:
        return [pred_image], out


def parseArguments():
    parser = argparse.ArgumentParser(description="Perform registration on MRI with Allen atlas.")
    parser.add_argument("-m", "--modelpath", help="Specify model path.")
    parser.add_argument("-i", "--input", help="MRI input")
    parser.add_argument("-o", "--output", help="MRI output")
    parser.add_argument("-N4", "--N4", action='store_true', help="Apply N4 to MRI before registration")
    parser.add_argument("-tmp", "--temp", action='store_true', help="Save temp files.")
    parser.add_argument("-t", "--type", type=str, default='affine', help="Specify affine or deformable. Default: affine")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    if args.type.lower() != 'affine' and args.type.lower() != 'deformable' and args.type.lower() != "local" and args.type.lower() != "null":
        print("Should specify model type as affine or deformable")
        sys.exit(1)

    if not args.modelpath:
        print("Should specify model path. See -h for instructions.")
        sys.exit(1)

    if not args.input:
        print("Should specify input MRI. See -h for instructions.")
        sys.exit(1)

    if not args.output:
        print("Should specify output MRI. See -h for instructions.")
        sys.exit(1)

    main(args.modelpath, args.input, args.output, args.N4, args.temp, args.type)
