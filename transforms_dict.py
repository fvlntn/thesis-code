import numpy as np
import torch
from monai.transforms import Activations, ScaleIntensity, EnsureType, LoadImage
from monai.transforms import AsDiscrete
from monai.transforms import Compose, Resize, RandRotate90d, RandGaussianNoised, RandBiasFieldd
from monai.transforms import EnsureChannelFirst, EnsureChannelFirstd
from monai.transforms import LoadImaged, ScaleIntensityd, RandAffined, EnsureTyped
from monai.transforms import RandRotated, Resized, CropForegroundd, RandCropByPosNegLabeld, RandShiftIntensityd, SqueezeDim
from torch.nn import Softmax

from transforms import GetLargestComponent, GetLabelsAsOneHotd, GetMaxChannelWise
from transforms import MaskIntensityMultid
from transforms import N4MRI, LoadNibabel, NibabelToNumpy, ResampleMRIToAtlas, ToNibabel, SaveNibabel, InverseOneHot
from utils import getAffineHeader, try_load, getDevice


def getSegmentationToTensor(labels=False):
    if labels:
        tensor_transform = Compose(
            [
                GetLabelsAsOneHotd(
                    keys=["seg"],
                    get=True,
                    skip=False,
                ),
                EnsureTyped(
                    keys=["img", "seg"],
                ),
            ]
        )
    else:
        tensor_transform = Compose(
            [
                EnsureTyped(
                    keys=["img", "seg"],
                ),
            ]
        )
    return tensor_transform


def getRegistrationToTensor():
    tensor_transform = Compose(
        [
            EnsureTyped(
                keys=["moving_image", "moving_label", "fixed_image", "fixed_label"],
                allow_missing_keys=True,
            ),
        ]
    )
    return tensor_transform


# -- REGISTRATION TRAINING TRANSFORMS


def getRegistrationLoadingTransforms():
    load_transforms = Compose(
        [
            LoadImaged(
                as_closest_canonical=(True, True, True, True),
                keys=["moving_image", "moving_label", "fixed_image", "fixed_label", "moving_regions", "fixed_regions", "original_image"],
                allow_missing_keys=True,
            ),
        ]
    )
    return load_transforms


def getRegistrationClassicTransforms(mask=False, img_size=False):
    classic_transforms = Compose(
        [
            EnsureChannelFirstd(
                keys=["moving_image", "moving_label", "fixed_image", "fixed_label", "moving_regions", "fixed_regions", "original_image"],
                allow_missing_keys=True,
                #channel_dim="no_channel"
            ),
            ScaleIntensityd(
                keys=["moving_image", "fixed_image", "original_image"],
                minv=0.0, maxv=1.0,
                allow_missing_keys=True,
            ),
        ]
    )
    if img_size:
        classic_transforms = Compose(
            [
                classic_transforms,
                Resized(
                    keys=["moving_image", "moving_label", "fixed_image", "fixed_label", "moving_regions", "fixed_regions", "original_image"],
                    spatial_size=(img_size, img_size, img_size),
                    allow_missing_keys=True,
                ),
            ]
        )
    if mask:
        classic_transforms = Compose(
            [
                classic_transforms,
                MaskIntensityMultid(
                    keys=["moving_image", "fixed_image", "original_image"],
                    mask_key=("moving_label", "fixed_label", "moving_label"),
                    allow_missing_keys=True,
                ),
            ]
        )
    return classic_transforms


def getRegistrationAugmentTransforms(noise=False):

    #augment_transforms = Compose(
    #    [
    #        RandShiftIntensityd(
    #            keys=["original_image"],
    #            prob=0.0,
    #            offsets=0.0,
    #            allow_missing_keys=True,
    #        ),
    #    ]
    #)
    
    augment_transforms = Compose(
        [
            RandAffined(
                keys=["moving_image", "moving_label", "fixed_image", "fixed_label", "moving_regions", "fixed_regions", "original_image"],
                mode=('bilinear', 'nearest', 'bilinear', 'nearest', 'nearest', 'nearest', 'bilinear'),
                prob=0.5,
                rotate_range=(np.pi / 90, np.pi / 90, np.pi / 90),
                scale_range=(0.05, 0.05, 0.05),
                translate_range=(2, 2, 2),
                allow_missing_keys=True,
            ),
            RandShiftIntensityd(
                keys=["moving_image", "fixed_image", "original_image"],
                prob=0.5,
                offsets=0.2,
                allow_missing_keys=True,
            ),
        ]
    )
    
    if noise:
        augment_transforms = Compose(
            [
                augment_transforms,
                RandGaussianNoised(
                    keys=["moving_image"],
                    prob=1.0,
                    mean=0.0,
                    std=0.1,
                ),
                RandBiasFieldd(
                    keys=["moving_image"],
                    prob=1.0,
                    coeff_range=(0.0, 0.1),
                ),
            ]
        )
    return augment_transforms


def getRegistrationTrainingTransforms(mask=False, noise=False, img_size=False):
    train_transforms = Compose(
        [
            getRegistrationLoadingTransforms(),
            getRegistrationClassicTransforms(mask, img_size),
            getRegistrationAugmentTransforms(noise),
            getRegistrationToTensor()
        ]
    )
    return train_transforms


def getRegistrationValidationTransforms(mask=False, img_size=False):
    valid_transforms = Compose(
        [
            getRegistrationLoadingTransforms(),
            getRegistrationClassicTransforms(mask, img_size),
            getRegistrationToTensor()
        ]
    )
    return valid_transforms


# -- REGISTRATION EVALUATION TRANSFORMS


def getRegistrationEvalTransformsForMRI(atlas_name, N4, outname, save):
    mri_transform = Compose(
        [
            LoadNibabel(),
            N4MRI(N4),
            SaveNibabel(path=outname + "_MRI.nii.gz", save=save),
            ResampleMRIToAtlas(atlas=atlas_name, mask=False, resample=True),
            SaveNibabel(path=outname + "_Resample.nii.gz", save=save),
            NibabelToNumpy(),
            EnsureChannelFirstd(channel_dim="no_channel"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureChannelFirstd(channel_dim="no_channel"),
            EnsureType(),
        ]
    )
    return mri_transform


def getRegistrationEvalTransformsForMask(atlas_name, save):
    mask_transform = Compose(
        [
            LoadNibabel(),
            SaveNibabel(path="Mask.nii.gz", save=save),
            
            ResampleMRIToAtlas(atlas=atlas_name, mask=True, resample=True),
            #ResampleToMatch(),
            
            NibabelToNumpy(),
            EnsureChannelFirstd(channel_dim="no_channel"),
            EnsureChannelFirstd(channel_dim="no_channel"),
            SaveNibabel(path="Mask_Resample.nii.gz", save=save),
            EnsureType(),
        ]
    )
    return mask_transform


def getRegistrationEvalTransformsForAtlas():
    atlas_transform = Compose(
        [
            LoadImage(image_only=True, as_closest_canonical=True),
            EnsureChannelFirstd(channel_dim="no_channel"),
            EnsureChannelFirstd(channel_dim="no_channel"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(),
        ]
    )
    return atlas_transform


def getRegistrationEvalInverseTransformForMRI(mri_name, out_name, atlas_name, save=False, mask=False):
    atlas_affine, atlas_header = getAffineHeader(atlas_name)
    mri_inverse_transform = Compose(
        [
            ToNibabel(affine=atlas_affine, header=atlas_header),
            
            ResampleMRIToAtlas(atlas=mri_name, mask=mask, resample=True),
            #ResampleToMatch(),
            
            SaveNibabel(path=out_name, save=save)
        ]
    )
    return mri_inverse_transform


# -- SEGMENTATION TRAINING TRANSFORMS


def getSegmentationLoadingTransforms(crop=False):
    load_transforms = Compose([
        LoadImaged(
            keys=["img", "seg"],
            as_closest_canonical=(True, True),
        ),
        # CropMRId(
        #    keys=["img", "seg"],
        #    segkey="seg",
        #    crop=(crop, crop),
        # )
    ])
    return load_transforms


def getSegmentationClassicTransforms(resample=False, labels=False):

    classic_transforms = Compose(
        [
            ScaleIntensityd(
                keys=["img"],
                minv=0.0, maxv=1.0,
            ),            
            EnsureChannelFirstd(
                channel_dim="no_channel",
                keys=["img", "seg"],
            ),
        ]
    )
    if not resample:
        classic_transforms = Compose(
            [
                classic_transforms,
                Resized(
                    keys=["img", "seg"],
                    spatial_size=(128, 128, 128),
                ),
            ]
        )
    return classic_transforms


def getSegmentationAugmentTransforms(labels=False):
    common_augment_transforms = Compose(
        [
            RandRotate90d(
                keys=["img", "seg"],
                prob=0.5,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=["img", "seg"],
                prob=0.5,
                max_k=3,
                spatial_axes=(1, 2),
            ),
            RandRotate90d(
                keys=["img", "seg"],
                prob=0.5,
                max_k=3,
                spatial_axes=(2, 0),
            ),
            RandRotated(
                keys=["img", "seg"],
                range_x=np.pi / 4,
                range_y=np.pi / 4,
                range_z=np.pi / 4,
                prob=0.25,
            ),
            RandShiftIntensityd(
                keys=["img"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    augment_transforms = Compose(
        [
            common_augment_transforms
        ]
    )
    return augment_transforms


def getSegmentationTrainingTransforms(crop=False, resample=False, labels=False):
    train_transforms = Compose(
        [
            getSegmentationLoadingTransforms(crop),
            getSegmentationClassicTransforms(resample, labels),
            getSegmentationAugmentTransforms(labels),
            getSegmentationToTensor(labels),
        ]
    )
    return train_transforms


def getSegmentationValidationTransforms(crop=False, resample=False, labels=False):
    valid_transforms = Compose(
        [
            getSegmentationLoadingTransforms(crop),
            getSegmentationClassicTransforms(resample, labels),
            getSegmentationToTensor(labels),
        ]
    )
    return valid_transforms


# -- SEGMENTATION EVALUATION TRANSFORMS

def getSegmentationPostProcessingForMask():
    postprocessing_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(threshold=0.5),
        ]
    )
    return postprocessing_transforms


def getSegmentationPostProcessingForMaskOutput():
    postprocessing_transforms = Compose(
        [
            EnsureType(),
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ]
    )
    return postprocessing_transforms


def getSegmentationPostProcessingForLabel(axis=0):
    device = getDevice()
    postprocessing_transforms = Compose(
        [
            EnsureType(),
            SqueezeDim(dim=0),
            AsDiscrete(to_onehot=4),
            #Activations(other=Softmax(dim=axis)),
            #OneHot(),
            #GetMaxChannelWise(axis=axis),
            EnsureType(data_type="tensor", device=device),
        ]
    )
    return postprocessing_transforms


def getSegmentationPostProcessingForLabelOutput(axis=0):
    device = getDevice()
    postprocessing_transforms = Compose(
        [
            EnsureType(),
            Activations(other=Softmax(dim=axis)),
            GetMaxChannelWise(axis=axis),
            EnsureType(data_type="tensor", device=device),
        ]
    )
    return postprocessing_transforms


def getSegmentationPostProcessingForAllLabelsOutput(axis=0):
    device = getDevice()
    postprocessing_transforms = Compose(
        [
            EnsureType(),
            Activations(other=Softmax(dim=axis)),
            InverseOneHot(),
            EnsureType(data_type="tensor", device=device),
        ]
    )
    return postprocessing_transforms


def getSegmentationEvalTransformsForMRI(N4=False, atlas_name=None, resample=False, outname=None, save=False,
                                        no_resize=False):
    if resample:
        eval_imtrans = Compose(
            [
                LoadNibabel(),
                N4MRI(N4),
                SaveNibabel(path=outname + "_MRI.nii.gz", save=save),
                # NibabelToNumpy(),
                # Orientation(axcodes='RAS', as_closest_canonical=True, image_only=True),
                # ToNibabel(affine=try_load(outname + "_MRI.nii.gz").affine.copy(),
                #          header=try_load(outname + "_MRI.nii.gz").header.copy()),
                ResampleMRIToAtlas(atlas=atlas_name, mask=True, resample=resample),
                SaveNibabel(path=outname + "_Resample.nii.gz", save=save),
                NibabelToNumpy(),
                EnsureChannelFirstd(channel_dim="no_channel"),
                ScaleIntensity(minv=0.0, maxv=1.0),
                EnsureType(),
                EnsureChannelFirstd(channel_dim="no_channel")
            ]
        )
    else:
        if no_resize:
            eval_imtrans = Compose(
                [
                    LoadNibabel(),
                    N4MRI(N4),
                    SaveNibabel(path=outname + "_MRI.nii.gz", save=save),
                    NibabelToNumpy(),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    ScaleIntensity(minv=0.0, maxv=1.0),
                    ToNibabel(affine=None, header=None),
                    SaveNibabel(path=outname + "_Preprocess.nii.gz", save=save),
                    NibabelToNumpy(),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    EnsureType(data_type="tensor", dtype=torch.float),
                ]
            )
        else:
            eval_imtrans = Compose(
                [
                    LoadNibabel(),
                    N4MRI(N4),
                    SaveNibabel(path=outname + "_MRI.nii.gz", save=save),
                    NibabelToNumpy(),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    ScaleIntensity(minv=0.0, maxv=1.0),
                    Resize((128, 128, 128), mode="trilinear"),
                    ToNibabel(affine=None, header=None),
                    SaveNibabel(path=outname + "_Resample.nii.gz", save=save),
                    NibabelToNumpy(),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    EnsureChannelFirstd(channel_dim="no_channel"),
                    EnsureType(data_type="tensor", dtype=torch.float),
                ]
            )
    return eval_imtrans


def getSegmentationInverseTransform(mri_name, atlas_name=None, resample=False, outname=None, save=False):
    post_im_trans = getSegmentationPostProcessingForMaskOutput()
    mri = try_load(mri_name)
    affine, header = mri.affine, mri.header
    if not resample:
        inverse_transform = Compose(
            [
                EnsureChannelFirstd(channel_dim="no_channel"),
                Resize(try_load(mri_name).shape, mode="trilinear"),
                ToNibabel(affine=affine, header=header),
                SaveNibabel(path=outname + "_out.nii.gz", save=save),
                NibabelToNumpy(),
                post_im_trans,
            ]
        )
    else:
        atlas_affine, atlas_header = getAffineHeader(atlas_name)
        inverse_transform = Compose(
            [
                ToNibabel(affine=atlas_affine, header=atlas_header),
                ResampleMRIToAtlas(atlas=mri_name, mask=False, resample=True),
                SaveNibabel(path=outname + "_out.nii.gz", save=save),
                NibabelToNumpy(),
                post_im_trans,
            ]
        )
    return inverse_transform


def getSegmentationInverseTransformForLabels(mri_name, out_name=None, suffix=None, save=False, no_resize=False):
    mri = try_load(mri_name)
    affine, header = mri.affine, mri.header
    if suffix is None:
        out_suffix = ""
    else:
        out_suffix = "_" + str(suffix)
    if no_resize:
        inverse_transform = Compose(
            [
                ToNibabel(affine=affine, header=header),
                SaveNibabel(path=out_name + "_out" + str(out_suffix) + ".nii.gz", save=save),
                NibabelToNumpy(),
            ]
        )
    else:
        inverse_transform = Compose(
            [
                EnsureChannelFirstd(channel_dim="no_channel"),
                Resize(try_load(mri_name).shape, mode="nearest"),
                ToNibabel(affine=affine, header=header),
                SaveNibabel(path=out_name + "_out" + str(out_suffix) + ".nii.gz", save=save),
                NibabelToNumpy(),
            ]
        )

    return inverse_transform


# UTILS


def SaveTransformForMRI(out_name, atlas_name, getLargestComponent=True, doSave=True):
    atlas_affine, atlas_header = getAffineHeader(atlas_name)
    save_transform = Compose(
        [
            ToNibabel(affine=atlas_affine, header=atlas_header),
            GetLargestComponent(get=getLargestComponent),
            SaveNibabel(path=out_name, save=doSave)
        ]
    )
    return save_transform
