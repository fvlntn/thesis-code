import subprocess
from typing import Hashable, Mapping, Sequence, Union, Callable

import ants
import nibabel as nib
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MaskIntensity
from monai.transforms.transform import Transform, MapTransform
from monai.transforms.utils import is_positive
from monai.utils import ensure_tuple_rep
from nilearn.image import resample_img, resample_to_img, math_img
from scipy import ndimage

from utils import clear_tmp, make_tmp, try_load, save_nibabel, getMaskBox, cropMRI


class MaskIntensityMultid(MapTransform):
    backend = MaskIntensity.backend

    def __init__(
            self,
            keys: KeysCollection,
            mask_key: Union[Sequence[str], str],
            select_fn: Callable = is_positive,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MaskIntensity(mask_data=None, select_fn=select_fn)
        self.mask_key = ensure_tuple_rep(mask_key, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        for key, mask_key in self.key_iterator(d, self.mask_key):
            d[key] = self.converter(d[key], d[mask_key]) if self.mask_key is not None else self.converter(d[key])
        return d


class SaveNibabel(Transform):
    def __init__(
            self,
            path: str,
            save: bool,
    ) -> None:
        self.path = path
        self.save = save

    def __call__(self, data):
        if self.save and self.path is not None:
            save_nibabel(data, self.path)
        return data


class SaveNibabeld(MapTransform):
    def __init__(
            self,
            save: Union[Sequence[bool], bool],
            path: Union[Sequence[str], str],
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.path = ensure_tuple_rep(path, len(self.keys))
        self.save = ensure_tuple_rep(save, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        for key, path, save in self.key_iterator(d, self.path, self.save):
            if save and path is not None:
                save_nibabel(d[key], path)
        return d


class LoadNibabel(Transform):
    def __call__(self, data):
        return try_load(data)


class LoadNibabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = try_load(d[key])
        return d


class ToNibabel(Transform):
    def __init__(
            self,
            affine,
            header,
    ) -> None:
        self.affine = affine
        self.header = header

    def __call__(self, data):
        try:
            new_data = data.squeeze().cpu().detach().numpy()
            new_image = nib.Nifti1Image(new_data, self.affine, self.header)
        except Exception:
            new_data = data.squeeze()
            new_image = nib.Nifti1Image(new_data, self.affine, self.header)
        return new_image


def nibabel_to_numpy(mri):
    return mri.get_fdata()


class NibabelToNumpy(Transform):
    def __call__(self, img) -> np.ndarray:
        return nibabel_to_numpy(img)


class NibabelToNumpyd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = nibabel_to_numpy(d[key])
        return d


def apply_resample_to_atlas(mri, atlas, mask=False):
    mri = try_load(mri)
    atlas = try_load(atlas)
    if mask:
        mri = resample_to_img(mri, atlas, interpolation='nearest')
        ensure_binary_mask(mri)
    else:
        mri = resample_to_img(mri, atlas, interpolation='continuous')
    mri.set_data_dtype('float32')
    return mri


class ResampleMRIToAtlas(Transform):
    def __init__(
            self,
            atlas: str,
            mask: bool,
            resample: bool,
    ) -> None:
        self.atlas = try_load(atlas)
        self.mask = mask
        self.resample = resample

    def __call__(self, img):
        if self.resample:
            img = apply_resample_to_atlas(img, self.atlas, self.mask)
        return img


class ResampleMRIToAtlasd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            atlas: str,
            mask: Union[Sequence[bool], bool],
            resample: Union[Sequence[bool], bool],
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask = ensure_tuple_rep(mask, len(self.keys))
        self.resample = ensure_tuple_rep(resample, len(self.keys))
        self.atlas = try_load(atlas)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key, mask, resample in self.key_iterator(d, self.mask, self.resample):
            if resample:
                d[key] = apply_resample_to_atlas(d[key], self.atlas, mask)
        return d


def apply_change_center(mri):
    mri = try_load(mri)
    old_affine = mri.affine[:3, :3]
    new_origin = old_affine.dot(list(np.array(mri.shape) / 2)[:3])
    new_affine = mri.affine.copy()
    new_affine[:3, 3] = -1.0 * new_origin
    imres = nib.Nifti1Image(mri.get_fdata().copy(), new_affine, mri.header)
    imres.set_sform(imres.get_qform())
    return imres


class ChangeOriginToCenter(Transform):
    def __call__(self, img):
        img = try_load(img)
        imres = apply_change_center(img)
        return imres


class ChangeOriginToCenterd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            mri = try_load(d[key])
            d[key] = apply_change_center(mri)
        return d


def apply_resample(mri, size, voxel):
    mri = try_load(mri)
    target_shape = np.array((size, size, size))
    new_resolution = [voxel, ] * 3
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    new_affine[:3, 3] = target_shape * new_resolution / 2. * -1
    new_affine[3, 3] = 1.
    mri = resample_img(mri, target_affine=new_affine, target_shape=(size, size, size))
    mri.set_data_dtype('float32')
    return mri


class ResampleMRI(Transform):
    def __init__(
            self,
            size: float,
            voxel: float,
    ) -> None:
        self.size = size
        self.voxel = voxel

    def __call__(self, img):
        img = apply_resample(img, self.size, self.voxel)
        return img


class ResampleMRId(MapTransform):
    def __init__(
            self,
            size: float,
            voxel: float,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = size
        self.voxel = voxel

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = apply_resample(d[key], self.size, self.voxel)
        return d


def call_fsl(scrFilename, resFilename, size, interp="spline"):
    command = "flirt -in " + scrFilename + " -ref " + scrFilename + " -out " + resFilename + " -applyisoxfm " + str(
        size) + " -interp " + interp
    subprocess.call(command.split(" "))
    image_out = try_load(resFilename)
    return image_out


def apply_isoxfm(mri, size):
    mri = try_load(mri)
    make_tmp()
    save_nibabel(mri, "tmp/mri.nii.gz", verbose=False)
    mri_isoxfm = call_fsl("tmp/mri.nii.gz", "tmp/mri_isoxfm.nii.gz", size)
    clear_tmp()
    return mri_isoxfm


class IsotropicMRI(Transform):
    def __init__(
            self,
            size: float,
    ) -> None:
        self.size = size

    def __call__(self, img):
        img = try_load(img)
        img = apply_isoxfm(img, self.size)
        return img


class IsotropicMRId(MapTransform):
    def __init__(
            self,
            size: float,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = size

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = apply_isoxfm(d[key], self.size)
        return d


class Shape(Transform):
    def __call__(self, mri):
        print(mri.shape)
        return mri


class Shaped(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            print(d[key].shape)
        return d


class IsotropicMRId(MapTransform):
    def __init__(
            self,
            size: float,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.size = size

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = apply_isoxfm(d[key], self.size)
        return d


def ensure_binary_mask(mask):
    try:
        mask = math_img('img > 0.5', img=mask)
    except Exception:
        mask = (mask > 0.5).float()
    return mask


class BinaryMask(Transform):
    def __call__(self, mask):
        mask = ensure_binary_mask(mask)
        return mask


class BinaryMaskd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = ensure_binary_mask(d[key])
        return d


def max_channel_wise(labels, axis=0):
    labels = labels.detach().cpu().numpy()
    labels[np.where(labels == np.amax(labels, axis=axis))] = 1
    labels[labels != 1] = 0
    return labels


class GetMaxChannelWise(Transform):
    def __init__(
            self,
            axis: int,
    ) -> None:
        self.axis = axis

    def __call__(self, labels):
        labels = max_channel_wise(labels, self.axis)
        return labels


def inverse_one_hot(labels):
    labels = labels.detach().cpu().numpy()
    labels = np.argmax(labels, axis=1)  # assume shape is (B,C,H,W,D)
    return labels


class InverseOneHot(Transform):
    def __call__(self, labels):
        labels = inverse_one_hot(labels)
        return labels


def apply_n4_antspy(mri):
    ants_image = ants.image_read(mri)
    image_n4 = ants.n4_bias_field_correction(ants_image, shrink_factor=4, spline_param=[4, 4, 4])
    nifti_image_n4 = ants.utils.to_nibabel(image_n4)
    return nifti_image_n4


def apply_n4_ants(mri):
    make_tmp()
    mri = try_load(mri)
    save_nibabel(mri, "tmp/mri.nii.gz", verbose=False)
    mri = "tmp/mri.nii.gz"
    truncated = "tmp/Truncated_" + mri.split('/')[-1]
    subprocess.run(["ImageMath", "3", truncated, "TruncateImageIntensity", mri, "0.001", "0.999", "256"])
    corrected = "tmp/N4_" + mri.split('/')[-1]
    subprocess.run(["N4BiasFieldCorrection", "-d", "3", "-i", truncated, "-o", corrected])
    n4 = try_load(corrected)
    clear_tmp()
    return n4


class N4MRI(Transform):
    def __init__(
            self,
            N4: bool,
    ) -> None:
        self.N4 = N4

    def __call__(self, img):
        if self.N4:
            img = apply_n4_ants(img)
        return img


class N4MRId(MapTransform):
    def __init__(
            self,
            N4: Union[Sequence[bool], bool],
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.N4 = ensure_tuple_rep(N4, len(self.keys))

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key, N4 in self.key_iterator(d, self.N4):
            if N4:
                d[key] = apply_n4_ants(d[key])
        return d


def get_crop_img(img, seg):
    box = getMaskBox(seg)
    img = cropMRI(img, box)
    return img


class CropMRI(Transform):
    def __init__(
            self,
            crop: bool,
            segkey,
    ) -> None:
        self.segkey = segkey
        self.crop = crop

    def __call__(self, img):
        if self.crop:
            img = get_crop_img(img, self.segkey)
        return img


class CropMRId(MapTransform):
    def __init__(
            self,
            crop: Union[Sequence[bool], bool],
            segkey,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.crop = ensure_tuple_rep(crop, len(self.keys))
        self.segkey = segkey

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key, crop in self.key_iterator(d, self.crop):
            if crop:
                d[key] = get_crop_img(d[key], d[self.segkey])
        return d


def get_largest_component(seg):
    mask = try_load(seg)
    data = mask.get_fdata()
    label_im, nb_labels = ndimage.label(data)
    sizes = ndimage.sum(data, label_im, range(nb_labels + 1))
    data = sizes == max(sizes)
    binary_img = data[label_im]
    new_mask = nib.Nifti1Image(binary_img, mask.affine, mask.header)
    return new_mask


class GetLargestComponent(Transform):
    def __init__(
            self,
            get: bool,
    ) -> None:
        self.get = get

    def __call__(self, mask):
        if self.get:
            mask = get_largest_component(mask)
        return mask


class GetLargestComponentd(MapTransform):
    def __init__(
            self,
            get: Union[Sequence[bool], bool],
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.get = ensure_tuple_rep(get, len(self.keys))

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key, get in self.key_iterator(d, self.get):
            if get:
                d[key] = get_largest_component(d[key])
        return d


def get_labels_as_one_hot(seg, skip_background=False):
    if skip_background:
        labels = np.zeros([3] + list(seg.shape))
        for k in range(int(seg.max()) + 1):
            one_hot = np.zeros_like(seg)
            one_hot[seg == k] = 1
            labels[k - 1, :, :, :] = one_hot
    else:
        labels = np.zeros([4] + list(seg.shape))
        for k in range(int(seg.max()) + 1):
            one_hot = np.zeros_like(seg)
            one_hot[seg == k] = 1
            labels[k, :, :, :] = one_hot
    return labels


class GetLabelsAsOneHot(Transform):
    def __init__(
            self,
            get: bool,
            skip: bool,
    ) -> None:
        self.get = get
        self.skip = skip

    def __call__(self, mask):
        if self.get:
            mask = get_labels_as_one_hot(mask, self.skip)
        return mask


class GetLabelsAsOneHotd(MapTransform):
    def __init__(
            self,
            get: Union[Sequence[bool], bool],
            skip: bool,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.get = ensure_tuple_rep(get, len(self.keys))
        self.skip = skip

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key, get in self.key_iterator(d, self.get):
            if get:
                d[key] = get_labels_as_one_hot(d[key], self.skip)
        return d
