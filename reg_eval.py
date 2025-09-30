import argparse
import os
from glob import glob

import nibabel as nib
import nilearn
import nilearn.image
import torch
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

import utils_parser
from reg_data import getRegistrationDataset
from reg_model import getRegistrationModel
from utils import getWorst, getBest, get_registration_loss_from_weights, compute_mean_dice
from utils import save_nibabel, getDevice, loadExistingModel, getAffineHeader


def eval(dataset, modelname, registration_type, atlas, mask):
    phase = 'test'
    set_determinism(seed=0)

    atlas_mri_affine, atlas_mri_header = getAffineHeader("dataset3/Atlas/P56_Atlas_128_norm_id.nii.gz")
    atlas_mask_affine, atlas_mask_header = getAffineHeader("dataset3/Atlas/P56_Annotation_128_norm_id.nii.gz")

    dataloaders, size = getRegistrationDataset(dataset=dataset,
                                               batch=1,
                                               training=False,
                                               augment=False,
                                               eval_augment=False,
                                               atlas=atlas,
                                               mask=mask)

    device = getDevice()

    model = getRegistrationModel(registration_type)
    weights = loadExistingModel(model, None, ft=modelname, registration=True)
    model.eval()

    scores = []
    with torch.no_grad():
        running_loss = 0.0
        running_metric = 0.0
        for i, data in enumerate(dataloaders[phase]):
            print(i, end='\r')
            fixed_image = data["fixed_image"].to(device)
            fixed_label = data["fixed_label"].to(device)

            if registration_type.lower() == 'affine':
                ddf, pred_image, pred_label = model(data)
            else:
                affine_ddf, ddf, pred_image, pred_label, affine_image, affine_label = model(data)
            pred_label = AsDiscrete(threshold=0.5)(pred_label)

            img_loss, lbl_loss, ddf_loss = get_registration_loss_from_weights(pred_image, pred_label, fixed_image, fixed_label, ddf, weights)
            loss = img_loss + lbl_loss + ddf_loss
            metric = compute_mean_dice(pred_label, fixed_label)
            running_loss += loss.item() * fixed_image.size(0)
            running_metric += metric.item() * fixed_image.size(0)

            scores.append([i, loss, metric])
            pred_image = pred_image.squeeze().cpu()
            pred_label = pred_label.squeeze().cpu()
            mriname = "output/" + str(phase) + "_" + str(i) + "_mri.nii.gz"
            maskname = "output/" + str(phase) + "_" + str(i) + "_mask.nii.gz"
            pred_mri = nib.Nifti1Image(pred_image, atlas_mri_affine, atlas_mri_header)
            pred_mask = nib.Nifti1Image(pred_label, atlas_mask_affine, atlas_mask_header)
            save_nibabel(pred_mri, mriname)
            save_nibabel(pred_mask, maskname)

        epoch_loss = running_loss / size[phase]
        epoch_metric = running_metric / size[phase]

        print(
            "mean: loss : {:.8f}, dice: {:.8f}".format(
                epoch_loss, epoch_metric,
            )
        )

    getWorst(scores, 10)
    getBest(scores, 10)

    registration_mris = sorted(glob(os.path.join('registration_out', "*_mri.nii.gz")))
    mris_4d = nilearn.image.concat_imgs(registration_mris)
    mris_4d.to_filename('mris_4d/mris_4d_' + str(modelname) + '.nii.gz')


def parseArguments():
    parser = argparse.ArgumentParser(description="Evaluate 3D mouse brain segmentation model.")
    parser.add_argument("-model", "--modelpath", help="Specify model path.")
    parser.add_argument("-d", "--dataset", help="Specify dataset (IRIS)")
    parser.add_argument("-t", "--type", type=str, default='affine', help="Specifiy affine or deformable. Default: affine")
    parser.add_argument("-a", "--atlas", action='store_true', help="Perform to-atlas registration instead of paired registration")
    parser.add_argument("-m", "--mask", action='store_true', help="Skullstrip dataset if available")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    utils_parser.handleRegistrationTypeParser(args.type, False)
    utils_parser.handleModelOutputParser(args.modelpath)
    utils_parser.handleDatasetParser(args.dataset)

    eval(args.dataset, args.modelpath, args.type, args.atlas, args.mask)
