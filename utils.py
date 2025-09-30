import csv
import os
import os.path
import subprocess
from glob import glob
from typing import Dict, Optional, Union, Sequence, Callable

import torch.nn as nn 
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch
import random
from monai.data import DataLoader, CacheDataset
from monai.data.meta_tensor import MetaTensor
#from monai.data.nifti_writer import write_nifti
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, MaskIntensity
from monai.transforms import Compose, ScaleIntensity, EnsureType, RandGaussianNoise, RandBiasField
from monai.utils import GridSampleMode, GridSamplePadMode
from skimage.measure import regionprops
from monai.networks.blocks import Warp
from monai.networks.utils import meshgrid_ij
import torch_directml
from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,
    RegistrationResidualConvBlock,
    get_conv_block,
    get_deconv_block,
)


def compute_model_input(fixed_image, fixed_label, moving_image, moving_label):
    return {
        "fixed_image": fixed_image,
        "fixed_label": fixed_label,
        "moving_image": moving_image,
        "moving_label": moving_label,
    }


def get_noise_transforms():
    noise_smoothing_transform = RandGaussianNoise(
        prob=1.0,
        mean=0.0,
        std=0.1,
    )
    noise_biasfield_transform = RandBiasField(
        prob=1.0,
        coeff_range=(0.0, 0.1),
    )
    noise_rescaling_transform = ScaleIntensity(
        minv=0.0, maxv=1.0,
    )
    return noise_smoothing_transform, noise_biasfield_transform, noise_rescaling_transform


def get_noise_loss(ddf1, ddf2, weight):
    loss = torch.nn.MSELoss()
    noise_ddf_compare_loss = weight * loss(ddf1, ddf2)
    return noise_ddf_compare_loss


def weight_histograms_conv3d(writer, step, weights, layer_number):
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"layer_{layer_number}/kernel_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
  flattened_weights = weights.flatten()
  tag = f"layer_{layer_number}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

def check_layer(layer, layer_number, writer, step):
    if isinstance(layer, nn.Conv3d):
        layer_number += 1
        weights = layer.weight
        weight_histograms_conv3d(writer, step, weights, layer_number)
    elif isinstance(layer, nn.ConvTranspose3d):
        layer_number += 1
        weights = layer.weight
        weight_histograms_conv3d(writer, step, weights, layer_number)
    elif isinstance(layer, nn.Linear):
        layer_number += 1
        weights = layer.weight
        weight_histograms_linear(writer, step, weights, layer_number)

def weight_histograms(writer, step, model):
  blocks = [
    model.encode_convs,
    model.encode_pools,
    model.bottom_block,
    model.decode_deconvs,
    model.decode_convs,
    ]
  layer_number = 0
  for block in blocks:
    for layer in block: 
        layer_number += 1
        check_layer(layer, layer_number, writer, step)      
        if isinstance(layer, nn.Sequential):
            for layer2 in layer:
                check_layer(layer2, layer_number, writer, step)
                layer_number += 1
                if not isinstance(layer2, RegistrationResidualConvBlock) and not isinstance(layer2, nn.BatchNorm3d) and not isinstance(layer2, nn.ReLU):
                    for layer3 in layer2:
                        check_layer(layer3, layer_number, writer, step)
                        layer_number += 1                        
                        if not isinstance(layer3, RegistrationResidualConvBlock) and not isinstance(layer3, nn.BatchNorm3d) and not isinstance(layer3, nn.ReLU) and not isinstance(layer3, nn.Conv3d):
                            for layer4 in layer3:
                                check_layer(layer4, layer_number, writer, step)
                    
            
      
      
def get_symcompare_loss(u1, u2, warp_stn, img_size=128):
    device = getDevice()
    warp_stn = warp_stn.to(device)
    image_size = (img_size, img_size, img_size)
    mesh_points = [torch.arange(0, dim) for dim in image_size]
    grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
    X = grid.to(dtype=torch.float).to(device)
    u1X = X + u1[0,:,:,:,:]
    u1X_x = u1X[0,:,:,:].unsqueeze(0).unsqueeze(0)
    u1X_y = u1X[1,:,:,:].unsqueeze(0).unsqueeze(0)
    u1X_z = u1X[2,:,:,:].unsqueeze(0).unsqueeze(0)
    u2u1X_x = warp_stn(u1X_x, u2)
    u2u1X_y = warp_stn(u1X_y, u2)
    u2u1X_z = warp_stn(u1X_z, u2)
    u2u1X = torch.stack([u2u1X_x.squeeze(), u2u1X_y.squeeze(), u2u1X_z.squeeze()])
    loss = torch.nn.MSELoss()
    sym_inv_loss = loss(u2u1X, X)
    return sym_inv_loss


def get_sym_loss(u1, u2, weight, warp_stn, img_size=128):
    return 0.5 * weight * (get_symcompare_loss(u1, u2, warp_stn, img_size) + get_symcompare_loss(u2, u1, warp_stn, img_size))


class TBLogger:
    def __init__(self, sym, noise, jcb, fold):
        self.metrics = {
            "loss": 0.0,
            "metric": 0.0,
            "dice": 0.0,
            "img_loss": 0.0,
            "lbl_loss": 0.0,
            "ddf_loss": 0.0,
        }
        self.running_dicts = {}
        self.train_dicts = {}
        self.valid_dicts = {}

        self.best_loss = np.inf
        self.best_epoch = -1

        self.metric_batch = 0

        for metric in self.metrics:
            self.running_dicts["running_" + metric] = 0.0
            self.train_dicts["train_" + metric] = 0.0
            self.valid_dicts["valid_" + metric] = 0.0

        self.add_metrics_scenario(sym, noise, jcb, fold)

    def print_train_complete(self, writer):
        print(f"train completed, "
              f"best_loss: {self.best_loss:.4f}  "
              f"at epoch: {self.best_epoch}")
        writer.close()

    def check_new_best_loss(self, epoch, model, optimizer, weights, lr, modelname):        
        if self.running_dicts["running_img_loss"] < self.best_loss:    
        #if self.running_dicts["running_loss"] < self.best_loss:
            #self.best_loss = self.running_dicts["running_loss"]
            self.best_loss = self.running_dicts["running_img_loss"]
            self.best_epoch = epoch + 1
            self.save_model(epoch, model, optimizer, weights, lr, modelname)
            print(
                "best loss {:.4f} at epoch {}".format(
                    self.best_loss, self.best_epoch
                )
            )

    def save_model(self, epoch, model, optimizer, weights, lr, modelname):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'weights': weights,
            'epoch': epoch,
            'lr': lr,
        },
            './models/' + modelname
        )

    def add_metric(self, key):
        self.metrics[key] = 0.0
        self.update_train("train_" + key, 0.0)
        self.update_valid("valid_" + key, 0.0)
        self.update_running("running_" + key, 0.0)

    def add_metrics(self, keys):
        for key in keys:
            self.add_metric(key)

    def add_metrics_scenario(self, sym, noise, jcb, fold):
        if sym:
            self.add_metrics(["sym_img_loss", "sym_ddf_loss", "sym_inv_loss"])
        if jcb:
            self.add_metrics(["jcb_loss"])
        if fold:
            self.add_metrics(["fold_loss"])
        if noise:
            self.add_metrics(["noise_img_loss", "noise_ddf_loss", "noise_ddfcompare_loss"])

    def update(self, key, value):
        self.metrics[key] = value

    def updates(self, keys, values):
        for i, key in enumerate(keys):
            self.update(key, values[i])

    def add(self, key, value):
        self.running_dicts[key] += value * self.metric_batch

    def one_pass_complete(self, metric_batch):
        self.set_metric_batch(metric_batch)
        for key in self.metrics:
            self.add("running_" + key, self.metrics[key])

    def half_epoch_complete(self, sizelol, phase):
        self.divide_running(sizelol)
        self.swap_running_to(phase)
        self.return_metrics(phase)

    def reset_running(self):
        for metric in self.metrics:
            self.running_dicts["running_" + metric] = 0.0

    def reset_train(self):
        for metric in self.metrics:
            self.train_dicts["train_" + metric] = 0.0

    def reset_valid(self):
        for metric in self.metrics:
            self.valid_dicts["valid_" + metric] = 0.0

    def set_metric_batch(self, value):
        self.metric_batch = value

    def update_train(self, key, value):
        self.train_dicts[key] = value

    def update_valid(self, key, value):
        self.valid_dicts[key] = value

    def update_running(self, key, value):
        self.running_dicts[key] = value

    def return_running(self, metric):
        return self.running_dicts[metric]

    def divide_running(self, sizelol):
        for metric in self.metrics:
            self.running_dicts["running_" + metric] /= sizelol

    def swap_running_to(self, phase):
        for metric in self.metrics:
            if phase == 'train':
                self.train_dicts["train_" + metric] = self.running_dicts["running_" + metric]
            else:
                self.valid_dicts["valid_" + metric] = self.running_dicts["running_" + metric]

    def return_metrics(self, phase):
        outmessage = "{}: loss: {:.4f} - metric: {:.4f} -- img: {:.4f}, ddf: {:.4f}".format(
            phase,
            self.running_dicts["running_loss"],
            self.running_dicts["running_metric"],
            self.running_dicts["running_img_loss"],
            self.running_dicts["running_ddf_loss"]
        )
        if "running_lbl_loss" in self.running_dicts and self.running_dicts["running_lbl_loss"] != 0.0:
            outmessage += ", lbl: {:.4f}".format(self.running_dicts["running_lbl_loss"])
        if "running_sym_img_loss" in self.running_dicts:
            outmessage += " || sym_img: {:.4f} - sym_ddf: {:.4f} - sym_inv: {:.4f}".format(
                self.running_dicts["running_sym_img_loss"], self.running_dicts["running_sym_ddf_loss"], self.running_dicts["running_sym_inv_loss"])
        if "running_noise_img_loss" in self.running_dicts:
            outmessage += " || noise_img: {:.4f} - noise_ddf: {:.4f} - noise_compare: {:.4f}".format(
                self.running_dicts["running_noise_img_loss"], self.running_dicts["running_noise_ddf_loss"], self.running_dicts["running_noise_ddfcompare_loss"])

        if "running_jcb_loss" in self.running_dicts:
            outmessage += " || jac: {:.4f}".format(self.running_dicts["running_jcb_loss"])
        if "running_fold_loss" in self.running_dicts:
            outmessage += " || fold: {:.4f}".format(self.running_dicts["running_fold_loss"])
        outmessage += " || dice: {:.4f}".format(self.running_dicts["running_dice"])
        print(outmessage)

    def write_metrics_to_TB(self, writer, epoch):
        for metric in self.metrics:
            writer.add_scalars('epoch_' + metric, {
                'train': self.train_dicts["train_" + metric],
                'valid': self.valid_dicts["valid_" + metric],
            }, epoch + 1)


def add_noise_to(moving_image, noise_biasfield_transform, noise_smoothing_transform, noise_rescaling_transform):
    noise_moving_image = noise_biasfield_transform(moving_image[0, :, :, :, :])
    noise_moving_image = noise_smoothing_transform(noise_moving_image)
    #noise_moving_image = noise_rescaling_transform(noise_moving_image)
    noise_moving_image = noise_moving_image.unsqueeze(0)
    return noise_moving_image


def getMaskedImage(image, mask, device):
    image = image.to(device)
    mask = mask.to(device)
    mask = AsDiscrete(threshold=0.5)(mask)
    image_masked = MaskIntensity(mask_data=mask)(image)
    return image_masked, mask


def setFreezeParameters(model, freeze):
    if freeze != 0:
        for name, p in model.named_parameters():
            p.requires_grad = False
    if freeze == 1:
        for name, p in model.named_parameters():
            if "encode_convs.2." in name:
                p.requires_grad = True
    if freeze == 2:
        for name, p in model.named_parameters():
            if "encode_convs.1." in name:
                p.requires_grad = True
            if "encode_convs.2." in name:
                p.requires_grad = True
    if freeze == 3:
        for name, p in model.named_parameters():
            if "encode_convs.1." in name:
                p.requires_grad = True
    if freeze == 4:
        for name, p in model.named_parameters():
            if "encode_convs.0." in name:
                p.requires_grad = True
            if "encode_convs.1." in name:
                p.requires_grad = True
            if "encode_convs.2." in name:
                p.requires_grad = True
    if freeze == 5:
        for name, p in model.named_parameters():
            if "encode_convs.1." in name:
                p.requires_grad = True
            if "encode_convs.2." in name:
                p.requires_grad = True
            if "encode_convs.3." in name:
                p.requires_grad = True
    if freeze == 6:
        for name, p in model.named_parameters():
            if "encode_convs.0." in name:
                p.requires_grad = True
            if "encode_convs.1." in name:
                p.requires_grad = True
            if "encode_convs.2." in name:
                p.requires_grad = True
            if "encode_convs.3." in name:
                p.requires_grad = True
    if freeze == 7:
        for name, p in model.named_parameters():
            if "encode_convs.1." in name:
                p.requires_grad = True
            if "encode_convs.2." in name:
                p.requires_grad = True
            if "encode_convs.3." in name:
                p.requires_grad = True
            if "bottom_block." in name:
                p.requires_grad = True


def compute_distance_ddfs(pred_ddf, data, mask, small=False):
    cwd = os.path.abspath("/home/valentini/dev/Mousenet/")
    warp_dir = os.path.join(cwd, "dataset3", "Fakedata", "Warp")
    mri_name = data['moving_image_meta_dict']['filename_or_obj'][0]
    mri_name = mri_name.split('/')[-1].split('_')[0]
    warp_name = os.path.join(warp_dir, mri_name + "_warp.nii.gz")

    truth_ddf = torch.from_numpy(nib.load(warp_name).get_fdata()).to(dtype=torch.float).cuda()
    if small:
        resize = monai.transforms.Resize(spatial_size=(small, small, small))
        truth_ddf = resize(truth_ddf.squeeze()).unsqueeze(0)
    pred_ddf = pred_ddf

    truth_ddf = truth_ddf * mask
    pred_ddf = pred_ddf * mask

    loss = torch.nn.MSELoss()
    distance = loss(truth_ddf, pred_ddf)

    return distance


def compute_landmarks_distance_local(ddf, data, save=None, small=False):
    ddf = ddf.permute(2, 3, 4, 0, 1)

    cwd = os.path.abspath("/home/valentini/dev/Mousenet/")
    affine_csv_dir = os.path.join(cwd, "dataset3", "Feminad", "Landmarks")
    mri_name = data['moving_image_meta_dict']['filename_or_obj'][0]
    mri_name = mri_name.split('/')[-1].split('_')

    moving_image_csv = os.path.join(affine_csv_dir, mri_name[0] + '_' + mri_name[1] + '_landmarks_id_affpos.csv')
    fixed_image_csv = os.path.join(cwd, "dataset3", "Atlas", "P56_Atlas_landmarks_id_scipy.csv")
    pred_image_csv = apply_warp_to_landmarks_df(fixed_image_csv, ddf, small=small)
    distances = compute_csv_distance(moving_image_csv, pred_image_csv, small=small)
    if save is not None:
        pred_image_csv.to_csv(save, index=False)
    return distances


def create_row_csv(x, y, z, t, label, mass, volume, count):
    row = {
        'x': x,
        'y': y,
        'z': z,
        't': t,
        'label': label,
        'mass': mass,
        'volume': volume,
        'count': count,
    }
    return row


def get_csv_positive(csv, out):
    if isinstance(csv, str):
        csv = pd.read_csv(csv, sep=',', header=0)
    df = csv
    df_out = pd.DataFrame()
    for i in range(len(df)):
        row = {
            'x': np.abs(df['x'][i]),
            'y': np.abs(df['y'][i]),
            'z': np.abs(df['z'][i]),
            't': df['t'][i],
            'label': df['label'][i],
            'mass': df['mass'][i],
            'volume': df['volume'][i],
            'count': df['count'][i],
        }
        temp_df = pd.DataFrame(data=row, index=[0])
        df_out = pd.concat([df_out, temp_df], ignore_index=True)
    df_out.to_csv(out, index=False)
    return df_out


def getRegionsAsCSV_antss(mask, out):
    ants_command = "ImageMath 3 " + str(out) + " LabelStats " + str(mask) + " " + str(mask)
    subprocess.call(ants_command.split(" "))
    return pd.read_csv(out, sep=',', header=0)


def getRegionsAsCSV_scipy(mask, out=None):
    if isinstance(mask, str):
        mask = nib.load(mask)
    img = mask.get_fdata().astype(np.uint8)
    regions = regionprops(img)
    df = pd.DataFrame()
    for i, props in enumerate(regions):
        x, y, z = props.centroid
        t = 0
        label = props.label
        mass = props.area
        volume = props.area
        count = len(props.coords)
        row = create_row_csv(x, y, z, t, label, mass, volume, count)
        temp_df = pd.DataFrame(data=row, index=[0])
        df = pd.concat([df, temp_df], ignore_index=True)
    if out != None:
        df.to_csv(out, index=False)
    return df


def compute_csv_distance(target_csv, pred_csv, small=False):
    if isinstance(target_csv, str):
        target_csv = pd.read_csv(target_csv, sep=',', header=0)
    if isinstance(pred_csv, str):
        pred_csv = pd.read_csv(pred_csv, sep=',', header=0)
    TREs = []
    for i in pred_csv['label']:
        if True:
            ind1 = np.where(target_csv['label'] == i)[0]
            ind2 = np.where(pred_csv['label'] == i)[0]
            if len(ind1) == 1 and len(ind2) == 1:
                x1 = target_csv['x'][ind1[0]]
                y1 = target_csv['y'][ind1[0]]
                z1 = target_csv['z'][ind1[0]]

                if small:
                    scale = small / 128
                    x1 = x1 * scale
                    y1 = y1 * scale
                    z1 = z1 * scale

                x2 = pred_csv['x'][ind2[0]]
                y2 = pred_csv['y'][ind2[0]]
                z2 = pred_csv['z'][ind2[0]]

                TRE = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
                if small:
                    TRE = TRE * 1/scale
                TREs.append(TRE)
    return TREs


def apply_warp_to_landmarks_df(df, warp, small=False):
    # DDF in [N,N,N,1,3] -> [3,N,N,N]
    if isinstance(warp, str):
        warp = nib.load(warp)
        warp_data = torch.from_numpy(warp.get_fdata())
    elif isinstance(warp, MetaTensor):
        warp_data = warp
    else:
        warp_data = torch.from_numpy(warp.get_fdata())
    warp_data = warp_data.permute(4, 0, 1, 2, 3).squeeze().cpu().detach().numpy()
    if isinstance(df, str):
        df = pd.read_csv(df)
    new_df = pd.DataFrame()
    for i in range(len(df)):
        x = df['x'][i]
        y = df['y'][i]
        z = df['z'][i]

        if small:
            scale = small / 128 # 128 to 32
            x = x * scale
            y = y * scale
            z = z * scale

        dx, dy, dz = interpolate_warp(warp_data, (x, y, z))

        new_x = x + dx
        new_y = y + dy
        new_z = z + dz

        t = df['t'][i]
        label = df['label'][i]
        mass = df['mass'][i]
        volume = df['volume'][i]
        count = df['count'][i]

        row = create_row_csv(new_x, new_y, new_z, t, label, mass, volume, count)

        temp_df = pd.DataFrame(data=row, index=[0])
        new_df = pd.concat([new_df, temp_df], ignore_index=True)

    return new_df


def interpolate_warp(warp_data, point):
    x_range = np.arange(0, warp_data.shape[1], 1)
    y_range = np.arange(0, warp_data.shape[2], 1)
    z_range = np.arange(0, warp_data.shape[3], 1)

    interp_x = scipy.interpolate.RegularGridInterpolator((x_range, y_range, z_range), warp_data[0, :, :, :],
                                                         fill_value=0)
    dx = interp_x(point)

    interp_y = scipy.interpolate.RegularGridInterpolator((x_range, y_range, z_range), warp_data[1, :, :, :],
                                                         fill_value=0)
    dy = interp_y(point)

    interp_z = scipy.interpolate.RegularGridInterpolator((x_range, y_range, z_range), warp_data[2, :, :, :],
                                                         fill_value=0)
    dz = interp_z(point)

    return dx, dy, dz


def get_warp_from_mat_file(ref, mat):
    warp_outname = "tmp.nii.gz"
    jacobian_command = "antsApplyTransforms -d 3 -n Linear -i " + str(ref) + " -o [" + str(
        warp_outname) + ",1] -r " + str(ref) + " -t " + str(mat)
    subprocess.call(jacobian_command.split(" "))
    return nib.load(warp_outname)


def convert_warp_physical_to_voxel(warp):
    affine = warp.affine
    header = warp.header
    inv_affine = np.linalg.inv(affine)
    inv_affine[:3, 3] = 0
    inv_affine[:3, :3] = inv_affine[:3, :3] * np.diag((-1, -1, 1))

    trans_array = torch.from_numpy(warp.get_fdata()).squeeze()
    warp_voxel = nib.affines.apply_affine(inv_affine, trans_array)
    warp_voxel = torch.from_numpy(warp_voxel).unsqueeze(dim=3).permute(3, 4, 0, 1, 2)
    warp_voxel = nib.Nifti1Image(warp_voxel.numpy(), affine, header)

    return warp_voxel


def convert_warp_voxel_to_physical(warp):
    affine = warp.affine
    header = warp.header
    affine[:3, 3] = 0
    affine[:3, :3] = affine[:3, :3] * np.diag((-1, -1, 1))

    trans_array = torch.from_numpy(warp.get_fdata()).squeeze().permute(1, 2, 3, 0)
    warp_voxel = nib.affines.apply_affine(affine, trans_array)
    warp_voxel = torch.from_numpy(warp_voxel).unsqueeze(dim=3)
    warp_voxel = nib.Nifti1Image(warp_voxel.numpy(), affine, header)

    return warp_voxel


def get_segmentation_splits(data_dir):
    if '/' in data_dir:
        data_dir = data_dir.split('/')[-1]
    if data_dir == 'IRIS':
        return 248, 82, 82
    elif data_dir == 'Painfact':
        return 169, 56, 56
    elif data_dir == 'Feminad':
        return 20, 6, 7
    elif data_dir == 'Femina3':
        return 81, 27, 27
    elif data_dir == 'GIN':
        return 15, 6, 6
    else:
        return 0, 0, 0


def get_file_lists(data_dir, n4=False, resample=False, nii=False, labels=False, affine=False, fold=0):
    dataset = data_dir.split('/')[-1]
    extension = ".nii" if nii else ".nii.gz"

    if affine:
        if "Femina3" in data_dir:            
            mris = sorted(glob(os.path.join(data_dir, 'MRI', '2mois*wt*_M_*affine.nii.gz')))
            masks = sorted(glob(os.path.join(data_dir, 'MaskDilate', '2mois*wt*_M_*affine.nii.gz')))        
        else:
            mris = sorted(glob(os.path.join(data_dir, 'MRI', '*_affine' + extension)))
            masks = sorted(glob(os.path.join(data_dir, 'Mask', '*_affine' + extension)))
        print('=> Using ' + str(dataset) + ' affine registered dataset.')
    elif fold != 0:
        mris = sorted(glob(os.path.join(data_dir, 'MRI', 'MRI_fold*' + extension)))
        masks = sorted(glob(os.path.join(data_dir, 'Mask', 'Mask_fold*' + extension)))
        print('=> Using ' + str(dataset) + ' GIN ' + str(fold) + '-fold dataset.')
    else:
        if "Femina3" in data_dir:
            mris = sorted(glob(os.path.join(data_dir, 'MRI', '*_id' + extension)))
            random.Random(4).shuffle(mris)
            masks = sorted(glob(os.path.join(data_dir, 'MaskDilate', '*_id' + extension)))
            random.Random(4).shuffle(masks)
        else:
            mris = sorted(glob(os.path.join(data_dir, 'MRI', '*_id' + extension)))
            masks = sorted(glob(os.path.join(data_dir, 'Mask', '*_id' + extension)))
            print('=> Using ' + str(dataset) + ' dataset.')

    return mris, masks


def get_file_lists_labels(data_dir, resample=False, affine=False):
    dataset = data_dir.split('/')[-1]
    extension = ".nii.gz"
    #if "GIN" in dataset:
        #if affine:
            #labels = sorted(glob(os.path.join(data_dir, 'Labels', '*_affine' + extension)))
        #else:
            #labels = sorted(glob(os.path.join(data_dir, 'Labels', '*_id' + extension)))
        #return labels
    #else:
        #return None
    if "GIN" in dataset:
        if affine:
            labels = sorted(glob(os.path.join(data_dir, 'LabelsBackFromAtlas', '*_affine' + extension)))
        else:
            labels = sorted(glob(os.path.join(data_dir, 'LabelsBackFromAtlas', '*_id' + extension)))
        return labels
    elif "Femina3" in dataset:        
        if affine:            
            labels = sorted(glob(os.path.join(data_dir, 'Labels2', '*_affine' + extension)))
        else:
            labels = sorted(glob(os.path.join(data_dir, 'Labels2', '*_id' + extension)))
        random.Random(4).shuffle(labels)
        return labels    
    else:
        return None


def getAffineHeader(mri):
    mri = try_load(mri)
    return mri.affine.copy(), mri.header.copy()


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #return torch_directml.device()     


class PairedRegistrationCacheDataset(CacheDataset):
    def __init__(
            self,
            data: Sequence,
            transform: Optional[Union[Sequence[Callable], Callable]] = None,
            cache_rate: float = 1.0,
            num_workers: Optional[int] = 1,
    ) -> None:
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data,
                         transform=transform,
                         cache_rate=cache_rate,
                         num_workers=num_workers,
                         )

    def __getitem__(self, index: int):
        index_moving = index
        index_fixed = torch.randint(self.__len__(), (1,))

        while index_fixed == index_moving:
            index_fixed = torch.randint(self.__len__(), (1,))

        _moving = self._transform(index_moving)
        _fixed = self._transform(index_fixed.item())

        moving_fixed = {
            "moving_image": _fixed['moving_image'],
            "moving_label": _fixed['moving_label'],
            "fixed_image": _moving['moving_image'],
            "fixed_label": _moving['moving_label'],
        }

        return moving_fixed


def getDs(files, transforms, paired):
    if files is not None and transforms is not None:
        if paired:
            return PairedRegistrationCacheDataset(data=files, transform=transforms, cache_rate=1.0, num_workers=4)
        else:
            return CacheDataset(data=files, transform=transforms, cache_rate=1.0, num_workers=4)
    else:
        return None


def getCacheDataset(train_files=None, train_transforms=None,
                    valid_files=None, valid_transforms=None,
                    test_files=None, test_transforms=None,
                    paired=False):
    train_ds = getDs(train_files, train_transforms, paired)
    valid_ds = getDs(valid_files, valid_transforms, paired)
    test_ds = getDs(test_files, test_transforms, paired)

    return train_ds, valid_ds, test_ds


def getDl(dataset, batch_size, shuffle, num_workers=4, pin_memory=True):
    if dataset is not None:
        if pin_memory:
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(),
                              )
        else:
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              )
    else:
        return None


def getDataloader(train_data=None, valid_data=None, test_data=None, batch_size=1, shuffle=True, num_workers=4):
    train_loader = getDl(train_data, batch_size, shuffle, num_workers=num_workers)
    valid_loader = getDl(valid_data, batch_size, shuffle, num_workers=num_workers)
    test_loader = getDl(test_data, batch_size, False, num_workers=1, pin_memory=False)
    return train_loader, valid_loader, test_loader


def loadExistingModel(model, optimizer, ft=None, ct=None, weights=None, registration=False):
    device = getDevice()
    if ft is not None:
        name = ft
    elif ct is not None:
        name = ct
    else:
        name = None
    if name is not None:
        checkpoint = torch.load('./models/' + str(name), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if ct is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('=> LR: ' + str(checkpoint['optimizer_state_dict']['param_groups'][0]['lr']))
            if registration:
                weights = checkpoint['weights']
    return weights


def save_nibabel(img, path, verbose=False):
    nib.save(img, path)
    print_save_file(path, verbose)
    return img


def save_tensor_to_nibabel(img, path, verbose=False):
    if path is not None:
        saver = NiftiSaver()
        saver.save(img, path)
        print_save_file(path, verbose)
    return img


def print_save_file(path, verbose=False):
    if verbose:
        print('=> Saved to ' + str(path))


def getAdamOptimizer(model, lr=0.00001):
    return torch.optim.Adam(model.parameters(), lr)


def getReducePlateauScheduler(optimizer, factor=0.9, patience=10):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      mode='min',
                                                      factor=factor,
                                                      patience=patience,
                                                      verbose=True)


def check_model_name(modelname):
    return modelname if modelname.endswith('.pth') else modelname + '.pth'


def remove_pth_extension(modelname):
    return modelname.split('.pth')[0] if modelname.endswith('.pth') else modelname


def print_model_output(modelname):
    print("=> Saving to " + str(modelname))


def add_weights_to_name(modelname, weights):
    modelname = remove_pth_extension(modelname)
    if len(weights) > 0:
        modelname += '_' + str(weights[0])
        for i, weight in enumerate(weights):
            if i > 0:
                modelname += '-' + str(weights[i])
    modelname += '.pth'
    return modelname


def print_weights(weights):
    print("=> Weights are: " + str(weights))


def get_batch_size(batch=1, training=False):
    return int(batch) if training else 1


def compute_mean_dice(pred_label, truth_label):
    dice_metric = DiceMetric(include_background=False, reduction="mean_channel", get_not_nans=False)
    dice_metric(y_pred=pred_label, y=truth_label)
    metric = dice_metric.aggregate()
    return metric


def make_tmp():
    clear_tmp()
    try:
        os.mkdir('tmp')
    except Exception:
        pass


def clear_tmp():
    try:
        os.rmdir('tmp')
    except Exception:
        pass


def try_load(arg):
    return nib.as_closest_canonical(nib.load(arg)) if isinstance(arg, str) else arg


def getWorst(scores, number=5):
    sorted_by_loss = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_by_metric = sorted(scores, key=lambda x: x[2])
    worst = []
    print('-' * 10)
    print(str(number) + ' worst losses: ')
    for i in range(int(number)):
        print(str(sorted_by_loss[i][0]) + ' - loss : ' + str(sorted_by_loss[i][1]) + ' - metric : ' + str(
            sorted_by_loss[i][2]))
        worst.append(sorted_by_loss[i][0])
    print('-' * 10)
    print(str(number) + ' worst metrics: ')
    for i in range(int(number)):
        print(str(sorted_by_metric[i][0]) + ' - metric : ' + str(sorted_by_metric[i][2]) + ' - loss : ' + str(
            sorted_by_metric[i][1]))
    return worst


def getBest(scores, number=5):
    sorted_by_loss = sorted(scores, key=lambda x: x[1])
    sorted_by_metric = sorted(scores, key=lambda x: x[2], reverse=True)
    best = []
    print('-' * 10)
    print(str(number) + ' best losses: ')
    for i in range(int(number)):
        print(str(sorted_by_loss[i][0]) + ' - loss : ' + str(sorted_by_loss[i][1]) + ' - metric : ' + str(
            sorted_by_loss[i][2]))
        best.append(sorted_by_loss[i][0])
    print('-' * 10)
    print(str(number) + ' best metrics: ')
    for i in range(int(number)):
        print(str(sorted_by_metric[i][0]) + ' - metric : ' + str(sorted_by_metric[i][2]) + ' - loss : ' + str(
            sorted_by_metric[i][1]))
    return best


def getMaskBox(Mask):
    s = Mask.shape
    low_x = 0
    while np.sum(Mask[low_x, :, :]) == 0:
        low_x += 1
    low_y = 0
    while np.sum(Mask[:, low_y, :]) == 0:
        low_y += 1
    low_z = 0
    while np.sum(Mask[:, :, low_z]) == 0:
        low_z += 1
    high_x = s[0] - 1
    while np.sum(Mask[high_x, :, :]) == 0:
        high_x -= 1
    high_y = s[1] - 1
    while np.sum(Mask[:, high_y, :]) == 0:
        high_y -= 1
    high_z = s[2] - 1
    while np.sum(Mask[:, :, high_z]) == 0:
        high_z -= 1

    return low_x, high_x + 1, low_y, high_y + 1, low_z, high_z + 1


def cropMRI(mri, box):
    low_x, high_x, low_y, high_y, low_z, high_z = box
    return mri[low_x:high_x, low_y:high_y, low_z:high_z]


#class NiftiSaver:
#    def __init__(
#            self,
#            output_dir: str = "./",
#            output_postfix: str = "seg",
#            output_ext: str = ".nii.gz",
#            resample: bool = True,
#            mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
#            padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
#            align_corners: bool = False,
#            dtype: Optional[np.dtype] = np.float64,
#    ) -> None:
#        self.output_dir = output_dir
#        self.output_postfix = output_postfix
#        self.output_ext = output_ext
#        self.resample = resample
#        self.mode: GridSampleMode = GridSampleMode(mode)
#        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
#        self.align_corners = align_corners
#        self.dtype = dtype
#        self._data_index = 0
#
#    def save(self, data: Union[torch.Tensor, np.ndarray], filename, meta_data: Optional[Dict] = None) -> None:
#        self._data_index += 1
#        original_affine = meta_data.get("original_affine", None) if meta_data else None
#        affine = meta_data.get("affine", None) if meta_data else None
#        spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None
#
#        if torch.is_tensor(data):
#            data = data.detach().cpu().numpy()
#
#        # change data shape to be (channel, h, w, d)
#        # while len(data.shape) < 4:
#        #    data = np.expand_dims(data, -1)
#        # change data to "channel last" format and write to nifti format file
#        # data = np.moveaxis(data, 0, -1)
#        write_nifti(
#            data,
#            file_name=filename,
#            affine=affine,
#            target_affine=original_affine,
#            resample=self.resample,
#            output_spatial_shape=spatial_shape,
#            mode=self.mode,
#            padding_mode=self.padding_mode,
#            align_corners=self.align_corners,
#            dtype=self.dtype,
#        )
#
#    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
#        for i, data in enumerate(batch_data):  # save a batch of files
#            self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)


def getMasksFromModel():
    prefix = "dataset/IRIS/"
    mris = sorted(glob(os.path.join(prefix, 'N4', "*.nii.gz")))

    for mri in mris:
        number = mri.split('/N4/')[1].split('.nii.gz')[0]
        outname = 'UNetMaskNoDiscrete/Mask' + str(number) + '.nii.gz'
        mri.main('oldclip/N4-augment-split-2.pth', mri, outname)


class QnetLoss:
    def __call__(self, MaskPred, MaskTrue, ScorePred, ScoreTrue):
        mask_loss = monai.losses.DiceLoss(sigmoid=True)(MaskPred, MaskTrue)

        dice = ScoreTrue[:, 0]
        pred_dice = ScorePred[:, 0]
        dice_loss = torch.nn.L1Loss()(pred_dice, dice)

        sens = ScoreTrue[:, 1]
        pred_sens = ScorePred[:, 1]
        sens_loss = torch.nn.L1Loss()(pred_sens, sens)

        spec = ScoreTrue[:, 2]
        pred_spec = ScorePred[:, 2]
        spec_loss = torch.nn.L1Loss()(pred_spec, spec)

        loss = mask_loss * 7 + dice_loss + sens_loss + spec_loss

        return loss, mask_loss, dice_loss, sens_loss, spec_loss


def getConfusionMatrix(pred, truth):
    tp = ((pred + truth) == 2).float().sum()
    tn = ((pred + truth) == 0).float().sum()
    p = truth.sum()
    n = torch.numel(truth) - p
    fn = p - tp
    fp = n - tn
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def compareMasks(pred, truth):
    to_tensor = Compose([EnsureType()])
    pred = to_tensor(pred).squeeze()
    truth = to_tensor(truth).squeeze()
    sensitivity, specificity = getConfusionMatrix(pred, truth)
    dice = compute_mean_dice(pred, truth)
    # hausdorff, _ = hausdorff_metric(pred, truth)
    return dice.item(), sensitivity.item(), specificity.item()


def getThresholdMasksFromRawMRI():
    prefix = "dataset/IRIS/"
    masks = sorted(glob(os.path.join(prefix, 'Mask', "*.nii.gz")))
    mris = sorted(glob(os.path.join(prefix, 'MRI', "*.nii.gz")))
    mri_masks = []

    for i in range(len(mris)):
        print(i, end='\r')
        mri_masks.append(
            {
                'mri': mris[i],
                'mask': masks[i],
            }
        )

    # seuillage du mri pour créer un masque sur un masque plus gros
    seuillage = 10
    seuil = 0.4
    saver = NiftiSaver()
    prefix = 'dataset/MaskDataset6/'
    with open(prefix + 'results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['MaskFilename', 'TruthFilename', 'Dice', 'Sensitivity', 'Specificity'])
        for i in range(len(mri_masks)):
            filename = mri_masks[i]['mri']
            number = filename.split('/MRI/')[1].split('.nii.gz')[0]
            outname = prefix + 'Mask' + str(number)
            for j in range(seuillage - 1):
                print(str(i) + '-' + str(j + 100), end='\r')
                pred = ScaleIntensity()(try_load(mri_masks[i]['mri']).get_fdata())
                truth = try_load(mri_masks[i]['mask']).get_fdata()
                # truth_bigger_shape = (int(truth.shape[0]*1.2), int(truth.shape[1]*1.2), int(truth.shape[2]*1.2))
                # truth_bigger = Resize(truth_bigger_shape, mode="trilinear", align_corners=False)(AddChannel()(truth))
                # truth_bigger = CenterSpatialCrop(truth.shape)(truth_bigger)[0,:,:,:]
                # truth_bigger[truth_bigger >= 0.5] = 1
                # truth_bigger[truth_bigger != 1] = 0
                # dice, _, _ = compareMasks(truth_bigger, truth)
                pred[pred >= seuil / seuillage * (j + 1)] = 1
                pred[pred != 1] = 0
                dice, _, _ = compareMasks(pred, truth)
                # pred[pred + truth_bigger != 2] = 0
                # pred[pred + truth_bigger == 2] = 1
                # dice, _, _ = compareMasks(pred, truth)
                dice, sensitivity, specificity = compareMasks(pred, truth)
                if dice > 0.5:
                    maskoutname = outname + '-' + str(j + 100) + '.nii.gz'
                    # print(maskoutname)
                    pred = pred.squeeze()
                    truthfilename = mri_masks[i]['mask']
                    saver.save(pred, maskoutname)
                    writer.writerow([maskoutname, truthfilename, dice, sensitivity, specificity])


def addUNetMasks():
    # ajout des masks unet
    prefix = "dataset/IRIS/"
    masks = sorted(glob(os.path.join(prefix, 'Mask', "*.nii.gz")))
    mris = sorted(glob(os.path.join(prefix, 'MRI', "*.nii.gz")))
    mri_masks = []

    for i in range(len(mris)):
        print(i, end='\r')
        mri_masks.append(
            {
                'mri': mris[i],
                'mask': masks[i],
            }
        )

    saver = NiftiSaver()
    unet_masks_tri = sorted(glob(os.path.join(prefix, 'UNetMaskTrilinear', "*.nii.gz")))
    prefix = 'dataset/MaskDataset6/'
    with open(prefix + 'results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['MaskFilename', 'TruthFilename', 'Dice', 'Sensitivity', 'Specificity'])
        for i in range(len(mri_masks)):
            print(i, end='\r')
            truth = try_load(mri_masks[i]['mask']).get_fdata().squeeze()
            pred = try_load(unet_masks_tri[i]).get_fdata().squeeze()
            dice, sensitivity, specificity = compareMasks(pred, truth)
            name = unet_masks_tri[i].split('/UNetMaskTrilinear/')[1].split('.nii.gz')[0]
            maskoutname = prefix + str(name) + '.nii.gz'
            truthfilename = mri_masks[i]['mask']
            saver.save(pred, maskoutname)
            writer.writerow([maskoutname, truthfilename, dice, sensitivity, specificity])


def getThresholdUnetMasks():
    # seuillage sur la sortie des unet
    prefix = "dataset/IRIS/"
    masks = sorted(glob(os.path.join(prefix, 'Mask', "*.nii.gz")))
    mris = sorted(glob(os.path.join(prefix, 'UNetMaskNoDiscrete', "*.nii.gz")))
    mri_masks = []

    for i in range(len(mris)):
        print(i, end='\r')
        mri_masks.append(
            {
                'mri': mris[i],
                'mask': masks[i],
            }
        )

    seuillage = 4
    seuil = 1.0
    saver = NiftiSaver()
    prefix = 'dataset/MaskDataset5/'
    with open(prefix + 'results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['MaskFilename', 'TruthFilename', 'Dice', 'Sensitivity', 'Specificity'])
        for i in range(len(mri_masks)):
            filename = mri_masks[i]['mri']
            number = filename.split('/MaskN4Corrected')[1].split('.nii.gz')[0]
            outname = prefix + 'Mask' + str(number)
            for j in range(seuillage - 1):
                print(str(i) + '-' + str(j + 100), end='\r')
                pred = ScaleIntensity()(try_load(mri_masks[i]['mri']).get_fdata())
                truth = try_load(mri_masks[i]['mask']).get_fdata()
                pred[pred >= seuil / seuillage * (j + 1)] = 1
                pred[pred != 1] = 0
                dice, sensitivity, specificity = compareMasks(pred, truth)
                maskoutname = outname + '-' + str(j + 100) + '.nii.gz'
                pred = pred.squeeze()
                truthfilename = mri_masks[i]['mask']
                saver.save(pred, maskoutname)
                writer.writerow([maskoutname, truthfilename, dice, sensitivity, specificity])
