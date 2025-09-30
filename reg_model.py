import torch.nn as nn
import torch
import torch.nn as nn
from monai.networks.blocks import Warp, DVF2DDF

from models import TrilinearGlobalNet, TrilinearLocalNet, TrilinearLocalNetZero
from utils import getDevice


class NullModel(nn.Module):
    def __init__(self):
        super(NullModel, self).__init__()		
        self.dvf = nn.Parameter(torch.zeros((1,3,128,128,128),requires_grad=True))
        self.device = getDevice()
        self.to(self.device)
        self.warp = Warp("bilinear", "border").to(self.device)
        self.warp_nearest = Warp("nearest", "border").to(self.device)
        self.dvf2ddf = DVF2DDF().to(self.device)
        self.use_dvf = True

    def update_phase(self, phase):
        if phase == 'train':
            self.train()
        elif phase == 'valid':
            self.eval()
            
    def forward(self, data):  
        moving_image = data["moving_image"].to(self.device, dtype=torch.float, non_blocking=True)
        fixed_image = data["fixed_image"].to(self.device, dtype=torch.float, non_blocking=True)
        ddf = self.dvf2ddf(self.dvf)
        pred_image = self.warp(moving_image, ddf)
        try:
            moving_label = data["moving_label"].to(self.device, dtype=torch.float, non_blocking=True)
            pred_label = self.warp_nearest(moving_label, ddf)
            del moving_label
        except Exception:
            pred_label = None
        del moving_image, fixed_image
        torch.cuda.empty_cache()
        return ddf, pred_image, pred_label, self.dvf, None, None

class GlobalNet(TrilinearGlobalNet):
    def __init__(self, img_size=128, use_dvf=True):
        super(GlobalNet, self).__init__(
            image_size=[img_size, img_size, img_size],
            spatial_dims=3,
            in_channels=2,
            num_channel_initial=16,
            depth=5,
        )
        self.device = getDevice()
        self.to(self.device)
        self.warp = Warp("bilinear", "border").to(self.device)
        self.warp_nearest = Warp("nearest", "border").to(self.device)
        self.dvf2ddf = DVF2DDF().to(self.device)
        self.use_dvf = use_dvf

    def update_phase(self, phase):
        if phase == 'train':
            self.train()
        elif phase == 'valid':
            self.eval()

    def forward(self, data):
        moving_image = data["moving_image"].to(self.device, dtype=torch.float, non_blocking=True)
        fixed_image = data["fixed_image"].to(self.device, dtype=torch.float, non_blocking=True)
        if self.use_dvf:
            dvf = super().forward(torch.cat((moving_image, fixed_image), dim=1))
            ddf = self.dvf2ddf(dvf)
        else:
            ddf = super().forward(torch.cat((moving_image, fixed_image), dim=1))
        pred_image = self.warp(moving_image, ddf)
        try:
            moving_label = data["moving_label"].to(self.device, dtype=torch.float, non_blocking=True)
            pred_label = self.warp_nearest(moving_label, ddf)
            del moving_label
        except Exception:
            pred_label = None

        del moving_image, fixed_image
        torch.cuda.empty_cache()

        if self.use_dvf:
            return ddf, pred_image, pred_label, dvf
        else:
            return ddf, pred_image, pred_label, None


class LocalNet(TrilinearLocalNet):
    def __init__(self, channels=32, extract=None, use_dvf=True, sym=False):
        if extract is None:
            extract = [0, 1, 2, 3, 4]
        super(LocalNet, self).__init__(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=channels,
            extract_levels=extract,
            out_activation=None,
            out_kernel_initializer="zeros",
            sym=sym
        )
        self.device = getDevice()
        self.to(self.device)
        self.warp = Warp("bilinear", "border").to(self.device)
        self.warp_nearest = Warp("nearest", "border").to(self.device)
        self.dvf2ddf = DVF2DDF().to(self.device)
        self.use_dvf = use_dvf
        self.sym = sym

    def update_phase(self, phase):
        if phase == 'train':
            self.train()
        elif phase == 'valid':
            self.eval()

    def forward(self, data):
        moving_image = data["moving_image"].to(self.device, dtype=torch.float, non_blocking=True)
        fixed_image = data["fixed_image"].to(self.device, dtype=torch.float, non_blocking=True)
        if self.sym:
            if self.use_dvf:
                dvf, dvf2 = super().forward(torch.cat((moving_image, fixed_image), dim=1))
                ddf = self.dvf2ddf(dvf)
                ddf2 = self.dvf2ddf(dvf2)
            else:
                ddf, ddf2 = super().forward(torch.cat((moving_image, fixed_image), dim=1))
        else:
            if self.use_dvf:
                dvf = super().forward(torch.cat((moving_image, fixed_image), dim=1))
                ddf = self.dvf2ddf(dvf)
            else:
                ddf = super().forward(torch.cat((moving_image, fixed_image), dim=1))
        pred_image = self.warp(moving_image, ddf)
        try:
            moving_label = data["moving_label"].to(self.device, dtype=torch.float)
            pred_label = self.warp_nearest(moving_label, ddf)
        except Exception:
            pred_label = None

        if self.sym:
            if self.use_dvf:
                return ddf, pred_image, pred_label, dvf, ddf2, dvf2
            else:
                return ddf, pred_image, pred_label, None, ddf2, None
        else:
            if self.use_dvf:
                return ddf, pred_image, pred_label, dvf, None, None
            else:
                return ddf, pred_image, pred_label, None, None, None
                
                
class LocalNetZero(TrilinearLocalNet):
    def __init__(self, channels=32, extract=None, use_dvf=True, sym=False):
        if extract is None:
            extract = [0, 1, 2, 3, 4]
        super(LocalNetZero, self).__init__(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=channels,
            extract_levels=extract,
            out_activation=None,
            out_kernel_initializer="zeros",
            sym=sym
        )
        self.device = getDevice()
        self.to(self.device)
        self.warp = Warp("bilinear", "border").to(self.device)
        self.warp_nearest = Warp("nearest", "border").to(self.device)
        self.dvf2ddf = DVF2DDF().to(self.device)
        self.use_dvf = use_dvf
        self.sym = sym

    def update_phase(self, phase):
        if phase == 'train':
            self.train()
        elif phase == 'valid':
            self.eval()

    def forward(self, data):
        moving_image = data["moving_image"].to(self.device, dtype=torch.float, non_blocking=True)
        fixed_image = data["fixed_image"].to(self.device, dtype=torch.float, non_blocking=True)
        #rand = torch.rand(size=(1, 1, 128, 128, 128)).to(self.device)
        #rand = torch.normal(0, 1, size=(1, 1, 128, 128, 128)).to(self.device)
        rand = 0.5*torch.ones(size=(1, 1, 128, 128, 128)).to(self.device)     
        #rand = torch.zeros(size=(1, 1, 128, 128, 128)).to(self.device)
        if self.sym:
            if self.use_dvf:
                dvf, dvf2 = super().forward(torch.cat((rand, rand), dim=1))
                ddf = self.dvf2ddf(dvf)
                ddf2 = self.dvf2ddf(dvf2)
            else:
                ddf, ddf2 = super().forward(torch.cat((rand, rand), dim=1))
        else:
            if self.use_dvf:
                dvf = super().forward(torch.cat((rand, rand), dim=1))
                ddf = self.dvf2ddf(dvf)
            else:
                ddf = super().forward(torch.cat((rand, rand), dim=1))
        pred_image = self.warp(moving_image, ddf)
        
        try:
            moving_label = data["moving_label"].to(self.device, dtype=torch.float)
            pred_label = self.warp_nearest(moving_label, ddf)
        except Exception:
            pred_label = None

        if self.sym:
            if self.use_dvf:
                return ddf, pred_image, pred_label, dvf, ddf2, dvf2
            else:
                return ddf, pred_image, pred_label, None, ddf2, None
        else:
            if self.use_dvf:
                return ddf, pred_image, pred_label, dvf, None, None
            else:
                return ddf, pred_image, pred_label, None, None, None
                

class DeformableNet(nn.Module):
    def __init__(self, img_size, pretrain_model=None, channels=16, extract=None, use_dvf=True, sym=False):
        super(DeformableNet, self).__init__()
        if extract is None:
            extract = [0, 1, 2, 3, 4]
        self.globalnet = GlobalNet(img_size, use_dvf=use_dvf)
        self.localnet = LocalNet(channels, extract, use_dvf=use_dvf, sym=sym)
        self.device = getDevice()
        if pretrain_model is not None:
            checkpoint = torch.load('./models/' + str(pretrain_model), map_location=self.device)
            self.globalnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.sym = sym

    def update_phase(self, phase):
        if phase == 'train':
            self.globalnet.train()
            self.localnet.train()
        elif phase == 'valid':
            self.globalnet.eval()
            self.localnet.eval()

    def freeze_affine(self, phase):
        for param in self.globalnet.parameters():
            param.requires_grad = False
        if phase == 'train':
            self.globalnet.eval()
            self.localnet.train()
        elif phase == 'valid':
            self.globalnet.eval()
            self.localnet.eval()

    def forward(self, data):
        affine_ddf, affine_image, affine_label, _ = self.globalnet(data)
        affine = {
            "fixed_image": data["fixed_image"],
            # "fixed_label": data["fixed_label"], # not useful
            "moving_image": affine_image,
            "moving_label": affine_label,
        }
        deformable_ddf, pred_image, pred_label, _ = self.localnet(affine)
        return affine_ddf, deformable_ddf, pred_image, pred_label, affine_image, affine_label


def getRegistrationModel(registration_type, pretrain_model=None, use_ddf=False, sym=False, newmodel=False, img_size=False):
    use_dvf = not use_ddf
    if not img_size:
        img_size = 128
    if newmodel:
        channels = 32
        extract = [0, 1, 2, 3, 4]
    else:
        channels = 16
        extract = [0, 1, 2, 3]
    device = getDevice()
    if registration_type.lower() == 'affine':
        model = GlobalNet(img_size=img_size, use_dvf=use_dvf).to(device)
    elif registration_type.lower() == 'local':
        model = LocalNet(channels=channels, extract=extract, use_dvf=use_dvf, sym=sym).to(device)
    elif registration_type.lower() == 'localzero':
        model = LocalNetZero(channels=channels, extract=extract, use_dvf=use_dvf, sym=sym).to(device)
    elif registration_type.lower() == 'null':
        model = NullModel().to(device)        
    elif registration_type.lower() == 'deformable':
        model = DeformableNet(img_size, pretrain_model=pretrain_model, channels=channels, extract=extract,
                              use_dvf=use_dvf, sym=sym).to(device)
    else:
        model = None
    return model
