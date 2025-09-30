import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import LocalNormalizedCrossCorrelationLoss, DiceLoss

from utils import getDevice


def compute_reg_train_loss(pred_image_masked, pred_mask, fixed_image_masked, fixed_mask, ddf, dvf, weights, reg_type, use_ddf):
    if reg_type == 'affine':
        img_loss, lbl_loss, ddf_loss = get_affine_registration_loss_from_weights(pred_image_masked,
                                                                                 pred_mask,
                                                                                 fixed_image_masked,
                                                                                 fixed_mask,
                                                                                 weights)
    elif reg_type == 'deformable' or reg_type == 'local' or reg_type == 'localzero' or reg_type == 'null':
        if use_ddf:
            img_loss, lbl_loss, ddf_loss = get_deformable_registration_loss_from_weights(pred_image_masked,pred_mask,fixed_image_masked,fixed_mask,ddf,weights)
        else:
            img_loss, lbl_loss, ddf_loss = get_deformable_registration_loss_from_weights(pred_image_masked,pred_mask,fixed_image_masked,fixed_mask,dvf,weights)
    return img_loss, lbl_loss, ddf_loss


def compute_affine_loss(u1, u2, A):
    ## => Affineloss = 1/128³*Sum(x€128³) |Ax+Au'(x) - (x+u(x))|²
    # input = AX+AU2(X) ||| target = X+U1(x)

    device = getDevice()
    A = torch.linalg.inv(A)
    u1 = u1.to(device)
    u2 = u2.to(device)

    a = torch.arange(128, dtype=torch.float64, device=device)
    one = torch.ones([128, 128, 128], dtype=torch.float64, device=device)
    img = torch.stack([torch.meshgrid(a,a,a)[0], torch.meshgrid(a,a,a)[1], torch.meshgrid(a,a,a)[2]])
    img_aff = torch.stack([torch.meshgrid(a,a,a)[0], torch.meshgrid(a,a,a)[1], torch.meshgrid(a,a,a)[2], one])

    A_stack = A.type(torch.float64).repeat(128 * 128 * 128, 1, 1).to(device)

    X = img.view((3, 128 * 128 * 128)).permute(1, 0).unsqueeze(2)
    U1X = torch.as_tensor(u1.as_tensor().view((3, 128 * 128 * 128, 1)).permute(1, 0, 2), dtype=torch.float64,
                          device=device)

    X_aff = img_aff.view((4, 128 * 128 * 128)).permute(1, 0).unsqueeze(2)
    U2X = torch.as_tensor(u2.as_tensor().view((3, 128 * 128 * 128, 1)).permute(1, 0, 2), dtype=torch.float64,
                          device=device)
    one_aff = torch.ones([128 * 128 * 128, 1], dtype=torch.float64, device=device)
    U2X_aff = torch.stack([U2X[:, 0], U2X[:, 1], U2X[:, 2], one_aff], dim=1)
    AXplusAU2X = torch.bmm(A_stack, X_aff + U2X_aff)[:, 0:3]  # .squeeze().permute(1,0).view((4,128,128,128))

    loss = torch.nn.MSELoss()
    affine_loss = loss(AXplusAU2X, X + U1X)

    return affine_loss


def compute_affine_loss_vincent(u1, u2, A):
    ## => Affineloss = 1/128³*Sum(x€128³) |Ax+Au'(x) - (x+u(x))|²

    # input = AX+AU2(X) ||| target = X+U1(x)

    device = getDevice()
    A = torch.linalg.inv(A).type(torch.DoubleTensor).to(device)
    #print(A)
    u1 = u1.squeeze().reshape(3, 128 * 128 * 128).type(torch.DoubleTensor).to(device)
    u2 = u2.squeeze().reshape(3, 128 * 128 * 128).type(torch.DoubleTensor).to(device)

    a = torch.arange(128, dtype=torch.float64, device=device)
    x = torch.stack(torch.meshgrid(a, a, a)).view(3, 128 * 128 * 128).to(device)

    h1 = (x + u1).to(device)

    h2 = x + u2
    h2 = h2.view((3, 128 * 128 * 128)).type(torch.DoubleTensor).to(device)

    h2 = torch.mm(A[:3, :3].to(device), h2) + A[:3, 3].unsqueeze(1).to(device)

    loss = torch.nn.MSELoss()
    affine_loss = loss(h1, h2)
    return affine_loss


# from https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/blob/master/Code/Models.py
def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


# from https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/blob/master/Code/Models.py
def JacobianDet(y_pred):
    device = getDevice()
    imgshape = (128,128,128)
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    grid = np.reshape(grid, (1,3,128,128,128))
    grid = torch.from_numpy(grid).float().to(device)

    J = y_pred + grid
    dy = J[:, :, 1:, :-1, :-1] - J[:, :, :-1, :-1, :-1]
    dx = J[:, :, :-1, 1:, :-1] - J[:, :, :-1, :-1, :-1]
    dz = J[:, :, :-1, :-1, 1:] - J[:, :, :-1, :-1, :-1]

    Jdet0 = dx[:,0,:,:,:] * (dy[:,1,:,:,:] * dz[:,2,:,:,:] - dy[:,2,:,:,:] * dz[:,1,:,:,:])
    Jdet1 = dx[:,1,:,:,:] * (dy[:,0,:,:,:] * dz[:,2,:,:,:] - dy[:,2,:,:,:] * dz[:,0,:,:,:])
    Jdet2 = dx[:,2,:,:,:] * (dy[:,0,:,:,:] * dz[:,1,:,:,:] - dy[:,1,:,:,:] * dz[:,0,:,:,:])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


# from https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/blob/master/Code/Models.py
def jacobian_loss(y_pred):
    neg_Jdet = -1.0 * JacobianDet(y_pred)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


# from https://github.com/zhangjun001/ICNet/blob/master/Code/Models.py
def antifolding_loss(y_pred):
    dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :] - 1
    dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :] - 1
    dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:] - 1

    dy = F.relu(dy) * torch.abs(dy * dy)
    dx = F.relu(dx) * torch.abs(dx * dx)
    dz = F.relu(dz) * torch.abs(dz * dz)
    return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0


# from https://github.com/Soumyadeep-Pal/Diffeomorphic-Image-Registration-Postprocess/blob/main/3D/src/plugin.py
def get_jacobian(flow_init):
    B,C,D,H,W = flow_init.shape
    phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz = Grid2Jac(flow_init,B,C,D,H,W)
    Jac = Jac_reshape_img2mat(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W)
    return Jac


# from https://github.com/Soumyadeep-Pal/Diffeomorphic-Image-Registration-Postprocess/blob/main/3D/src/plugin.py
def Grid2Jac(X, B, C, D, H, W):
    ## Let the grid be phi
    phi_X = X[:, 2, :, :, :]  ## IS x,y,z ok ?
    phi_Y = X[:, 1, :, :, :]
    phi_Z = X[:, 0, :, :, :]
    device = getDevice()
    phi_X_dx = torch.zeros(phi_X.shape).to(device)
    phi_X_dx[:, :, :, 0:W - 1] = phi_X[:, :, :, 1:W] - phi_X[:, :, :, 0:W - 1]
    phi_X_dy = torch.zeros(phi_X.shape).to(device)
    phi_X_dy[:, :, 0:H - 1, :] = phi_X[:, :, 1:H, :] - phi_X[:, :, 0:H - 1, :]
    phi_X_dz = torch.zeros(phi_X.shape).to(device)
    phi_X_dz[:, 0:D - 1, :, :] = phi_X[:, 1:D, :, :] - phi_X[:, 0:D - 1, :, :]

    phi_Y_dx = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dx[:, :, :, 0:W - 1] = phi_Y[:, :, :, 1:W] - phi_Y[:, :, :, 0:W - 1]
    phi_Y_dy = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dy[:, :, 0:H - 1, :] = phi_Y[:, :, 1:H, :] - phi_Y[:, :, 0:H - 1, :]
    phi_Y_dz = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dz[:, 0:D - 1, :, :] = phi_Y[:, 1:D, :, :] - phi_Y[:, 0:D - 1, :, :]

    phi_Z_dx = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dx[:, :, :, 0:W - 1] = phi_Z[:, :, :, 1:W] - phi_Z[:, :, :, 0:W - 1]
    phi_Z_dy = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dy[:, :, 0:H - 1, :] = phi_Z[:, :, 1:H, :] - phi_Z[:, :, 0:H - 1, :]
    phi_Z_dz = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dz[:, 0:D - 1, :, :] = phi_Z[:, 1:D, :, :] - phi_Z[:, 0:D - 1, :, :]

    return phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz


# from https://github.com/Soumyadeep-Pal/Diffeomorphic-Image-Registration-Postprocess/blob/main/3D/src/plugin.py
def Jac_reshape_img2mat(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B, C,
                        D, H, W):
    num_pixels = D * H * W
    device = getDevice()

    Jac = torch.zeros(B, num_pixels, C, C).to(device)

    Jac[:, :, 0, 0] = phi_X_dx.reshape(B, num_pixels)
    Jac[:, :, 0, 1] = phi_X_dy.reshape(B, num_pixels)
    Jac[:, :, 0, 2] = phi_X_dz.reshape(B, num_pixels)

    Jac[:, :, 1, 0] = phi_Y_dx.reshape(B, num_pixels)
    Jac[:, :, 1, 1] = phi_Y_dy.reshape(B, num_pixels)
    Jac[:, :, 1, 2] = phi_Y_dz.reshape(B, num_pixels)

    Jac[:, :, 2, 0] = phi_Z_dx.reshape(B, num_pixels)
    Jac[:, :, 2, 1] = phi_Z_dy.reshape(B, num_pixels)
    Jac[:, :, 2, 2] = phi_Z_dz.reshape(B, num_pixels)

    return Jac


def get_affine_registration_loss_from_weights(affine_image, affine_label,
                                              fixed_image, fixed_label,
                                              weights):
    image_loss = MultiScaleNCC(win=7, scale=3)
    #image_loss = NCC()
    #image_loss = LocalNormalizedCrossCorrelationLoss()
    #image_loss = torch.nn.MSELoss()
    #image_loss = torch.nn.L1Loss()
    img_loss = weights[0] * (1 + image_loss(affine_image, fixed_image))  # LNCCLoss is -LNCC so 1 + Loss = 1-LNCC

    label_loss = DiceLoss()
    #label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8, 16])
    lbl_loss = weights[1] * label_loss(affine_label, fixed_label)

    ddf_loss = torch.tensor(0)

    return img_loss, lbl_loss, ddf_loss


def get_deformable_registration_loss_from_weights(pred_image, pred_label,
                                                  fixed_image, fixed_label,
                                                  ddf, weights):
    #image_loss = MultiScaleNCC(win=7, scale=3)
    image_loss = LocalNormalizedCrossCorrelationLoss()
    deformable_loss = 1 + image_loss(pred_image, fixed_image)  # LNCCLoss is -LNCC so 1 + Loss = 1-LNCC
    img_loss = weights[0] * deformable_loss

    lbl_loss = torch.tensor(0)

    #regularization = BendingEnergyLoss()
    #ddf_loss = weights[2] * regularization(ddf)
    ddf_loss = weights[2] * smoothloss(ddf)

    return img_loss, lbl_loss, ddf_loss


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=7, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class MultiScaleNCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=None, eps=1e-5, scale=3, kernel=3):
        super(MultiScaleNCC, self).__init__()
        self.num_scale = scale
        self.kernel = kernel
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = F.avg_pool3d(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = F.avg_pool3d(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)