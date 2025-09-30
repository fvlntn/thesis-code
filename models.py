# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,
    RegistrationResidualConvBlock,
    get_conv_block,
    get_deconv_block,
    get_conv_layer,
)
from monai.networks.utils import meshgrid_ij
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers import Conv, Norm, Pool, same_padding


class SYMNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SYMNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)

        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        f_yx = self.dc10(d3)

        return f_xy, f_yx

class RegistrationExtractionBlock(nn.Module):
    """
    The Extraction Block used in RegUNet.
    Extracts feature from each ``extract_levels`` and takes the average.
    """

    def __init__(
        self,
        spatial_dims: int,
        extract_levels: Tuple[int],
        num_channels: Union[Tuple[int], List[int]],
        out_channels: int,
        kernel_initializer: Optional[str] = "kaiming_uniform",
        activation: Optional[str] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions
            extract_levels: spatial levels to extract feature from, 0 refers to the input scale
            num_channels: number of channels at each scale level,
                List or Tuple of length equals to `depth` of the RegNet
            out_channels: number of output channels
            kernel_initializer: kernel initializer
            activation: kernel activation function
        """
        super().__init__()
        self.extract_levels = extract_levels
        self.max_level = max(extract_levels)
        self.layers = nn.ModuleList(
            [
                get_conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[d],
                    out_channels=out_channels,
                    norm=None,
                    act=activation,
                    initializer=kernel_initializer,
                )
                for d in extract_levels
            ]
        )

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        """
        Args:
            x: Decoded feature at different spatial levels, sorted from deep to shallow
            image_size: output image size
        Returns:
            Tensor of shape (batch, `out_channels`, size1, size2, size3), where (size1, size2, size3) = ``image_size``
        """
        feature_list = [
            F.interpolate(layer(x[self.max_level - level]), size=image_size, mode="trilinear")
            for layer, level in zip(self.layers, self.extract_levels)
        ]
        out: torch.Tensor = torch.mean(torch.stack(feature_list, dim=0), dim=0)
        return out


class RegUNet(nn.Module):
    """
    Class that implements an adapted UNet. This class also serve as the parent class of LocalNet and GlobalNet
    Reference:
        O. Ronneberger, P. Fischer, and T. Brox,
        “U-net: Convolutional networks for biomedical image segmentation,”,
        Lecture Notes in Computer Science, 2015, vol. 9351, pp. 234–241.
        https://arxiv.org/abs/1505.04597
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        extract_levels: Optional[Tuple[int]] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
        sym = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            depth: input is at level 0, bottom is at level depth.
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
            encode_kernel_sizes: kernel size for down-sampling
        """
        super().__init__()
        if not extract_levels:
            extract_levels = (depth,)
        if max(extract_levels) != depth:
            raise AssertionError

        # save parameters
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channel_initial = num_channel_initial
        self.depth = depth
        self.out_kernel_initializer = out_kernel_initializer
        self.out_activation = out_activation
        self.out_channels = out_channels
        self.extract_levels = extract_levels
        self.pooling = pooling
        self.concat_skip = concat_skip
        self.sym = sym

        if isinstance(encode_kernel_sizes, int):
            encode_kernel_sizes = [encode_kernel_sizes] * (self.depth + 1)
        if len(encode_kernel_sizes) != self.depth + 1:
            raise AssertionError
        self.encode_kernel_sizes: List[int] = encode_kernel_sizes

        self.num_channels = [self.num_channel_initial * (2**d) for d in range(self.depth + 1)]
        self.min_extract_level = min(self.extract_levels)

        # init layers
        # all lists start with d = 0
        self.encode_convs: nn.ModuleList
        self.encode_pools: nn.ModuleList
        self.bottom_block: nn.Sequential
        self.decode_deconvs: nn.ModuleList
        self.decode_convs: nn.ModuleList
        self.output_block: nn.Module
        #print(self.sym)
        if self.sym:
            self.output_sym_block: nn.Module

        # build layers
        self.build_layers()

    def build_layers(self):
        self.build_encode_layers()
        self.build_decode_layers()

    def build_encode_layers(self):
        # encoding / down-sampling
        self.encode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=self.in_channels if d == 0 else self.num_channels[d - 1],
                    out_channels=self.num_channels[d],
                    kernel_size=self.encode_kernel_sizes[d],
                )
                for d in range(self.depth)
            ]
        )
        self.encode_pools = nn.ModuleList(
            [self.build_down_sampling_block(channels=self.num_channels[d]) for d in range(self.depth)]
        )
        self.bottom_block = self.build_bottom_block(
            in_channels=self.num_channels[-2], out_channels=self.num_channels[-1]
        )

    def build_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                
            ),
            RegistrationResidualConvBlock(
                spatial_dims=self.spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def build_down_sampling_block(self, channels: int):
        return RegistrationDownSampleBlock(spatial_dims=self.spatial_dims, channels=channels, pooling=self.pooling)

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            RegistrationResidualConvBlock(
                spatial_dims=self.spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def build_decode_layers(self):
        self.decode_deconvs = nn.ModuleList(
            [
                self.build_up_sampling_block(in_channels=self.num_channels[d + 1], out_channels=self.num_channels[d])
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )
        self.decode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=(2 * self.num_channels[d] if self.concat_skip else self.num_channels[d]),
                    out_channels=self.num_channels[d],
                    kernel_size=3,
                )
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )
        # extraction
        self.output_block = self.build_output_block()
        if self.sym:
            self.output_sym_block = self.build_output_block()

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def build_output_block(self) -> nn.Module:
        return RegistrationExtractionBlock(
            spatial_dims=self.spatial_dims,
            extract_levels=self.extract_levels,
            num_channels=self.num_channels,
            out_channels=self.out_channels,
            kernel_initializer=self.out_kernel_initializer,
            activation=self.out_activation,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])
        Returns:
            Tensor in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3]), with the same spatial size as ``x``
        """
        image_size = x.shape[2:]
        skips = []  # [0, ..., depth - 1]
        encoded = x
        for encode_conv, encode_pool in zip(self.encode_convs, self.encode_pools):
            skip = encode_conv(encoded)
            encoded = encode_pool(skip)
            skips.append(skip)
        decoded = self.bottom_block(encoded)

        outs = [decoded]

        for i, (decode_deconv, decode_conv) in enumerate(zip(self.decode_deconvs, self.decode_convs)):
            decoded = decode_deconv(decoded)
            if self.concat_skip:
                decoded = torch.cat([decoded, skips[-i - 1]], dim=1)
            else:
                decoded = decoded + skips[-i - 1]
            decoded = decode_conv(decoded)
            outs.append(decoded)

        out = self.output_block(outs, image_size=image_size)
        if self.sym:
            out2 = self.output_sym_block(outs, image_size=image_size)
            return out, out2
        else:        
            return out


class AffineHead(nn.Module):
    def __init__(self, spatial_dims: int, image_size: List[int], decode_size: List[int], in_channels: int):
        super().__init__()
        self.spatial_dims = spatial_dims
        if spatial_dims == 2:
            in_features = in_channels * decode_size[0] * decode_size[1]
            out_features = 6
            out_init = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        elif spatial_dims == 3:
            in_features = in_channels * decode_size[0] * decode_size[1] * decode_size[2]
            out_features = 12
            out_init = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        else:
            raise ValueError(f"only support 2D/3D operation, got spatial_dims={spatial_dims}")

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.grid = self.get_reference_grid(image_size)  # (spatial_dims, ...)

        # init weight/bias
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(out_init)

    @staticmethod
    def get_reference_grid(image_size: Union[Tuple[int], List[int]]) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in image_size]
        grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
        return grid.to(dtype=torch.float)

    def affine_transform(self, theta: torch.Tensor):
        # (spatial_dims, ...) -> (spatial_dims + 1, ...)
        grid_padded = torch.cat([self.grid, torch.ones_like(self.grid[:1])])

        # grid_warped[b,p,...] = sum_over_q(grid_padded[q,...] * theta[b,p,q]
        if self.spatial_dims == 2:
            grid_warped = torch.einsum("qij,bpq->bpij", grid_padded, theta.reshape(-1, 2, 3))
        elif self.spatial_dims == 3:
            grid_warped = torch.einsum("qijk,bpq->bpijk", grid_padded, theta.reshape(-1, 3, 4))
        else:
            raise ValueError(f"do not support spatial_dims={self.spatial_dims}")
        return grid_warped

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        f = x[0]
        self.grid = self.grid.to(device=f.device)
        theta = self.fc(f.reshape(f.shape[0], -1))
        out: torch.Tensor = self.affine_transform(theta) - self.grid
        return out


class TrilinearGlobalNet(RegUNet):
    """
    Build GlobalNet for image registration.
    Reference:
        Hu, Yipeng, et al.
        "Label-driven weakly-supervised learning
        for multimodal deformable image registration,"
        https://arxiv.org/abs/1711.01666
    """

    def __init__(
        self,
        image_size: List[int],
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
    ):
        for size in image_size:
            if size % (2**depth) != 0:
                raise ValueError(
                    f"given depth {depth}, "
                    f"all input spatial dimension must be divisible by {2 ** depth}, "
                    f"got input of size {image_size}"
                )
        self.image_size = image_size
        self.decode_size = [size // (2**depth) for size in image_size]
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=spatial_dims,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=encode_kernel_sizes,
        )

    def build_output_block(self):
        return AffineHead(
            spatial_dims=self.spatial_dims,
            image_size=self.image_size,
            decode_size=self.decode_size,
            in_channels=self.num_channels[-1],
        )
        
    
class AdditiveUpSampleBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = get_deconv_block(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = [size * 2 for size in x.shape[2:]]
        deconved = self.deconv(x)
        resized = F.interpolate(x, size=output_size, mode="trilinear")
        resized = torch.sum(torch.stack(resized.split(split_size=resized.shape[1] // 2, dim=1), dim=-1), dim=-1)
        out: torch.Tensor = deconved + resized
        return out


class TrilinearLocalNet(RegUNet):
    """
    Reimplementation of LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        pooling: bool = True,
        use_addictive_sampling: bool = True,
        concat_skip: bool = False,
        sym: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            use_addictive_sampling: whether use additive up-sampling layer for decoding.
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
        """
        self.use_additive_upsampling = use_addictive_sampling
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=extract_levels,
            depth=max(extract_levels),
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=[7] + [3] * max(extract_levels),
            sym=sym,
        )
        

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels
            )

        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)
               

class RegistrationResidualConvBlockZero(nn.Module):
    """
    A block with skip links and layer - norm - activation.
    Only changes the number of channels, the spatial size is kept same.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, num_layers: int = 2, kernel_size: int = 3
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions
            in_channels: number of input channels
            out_channels: number of output channels
            num_layers: number of layers inside the block
            kernel_size: kernel_size
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([Norm[Norm.BATCH, spatial_dims](out_channels) for _ in range(num_layers)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])

        Returns:
            Tensor in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3]),
            with the same spatial size as ``x``
        """
        skip = x
        for i, (conv, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            x = conv(x)
            x = norm(x)
            if i == self.num_layers - 1:
                # last block
                x = x # + skip | remove skip for zero
            x = act(x)
        return x
        
class RegUNetZero(nn.Module):
    """
    Class that implements an adapted UNet. This class also serve as the parent class of LocalNet and GlobalNet
    Reference:
        O. Ronneberger, P. Fischer, and T. Brox,
        “U-net: Convolutional networks for biomedical image segmentation,”,
        Lecture Notes in Computer Science, 2015, vol. 9351, pp. 234–241.
        https://arxiv.org/abs/1505.04597
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        extract_levels: Optional[Tuple[int]] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
        sym = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            depth: input is at level 0, bottom is at level depth.
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
            encode_kernel_sizes: kernel size for down-sampling
        """
        super().__init__()
        if not extract_levels:
            extract_levels = (depth,)
        if max(extract_levels) != depth:
            raise AssertionError

        # save parameters
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channel_initial = num_channel_initial
        self.depth = depth
        self.out_kernel_initializer = out_kernel_initializer
        self.out_activation = out_activation
        self.out_channels = out_channels
        self.extract_levels = extract_levels
        self.pooling = pooling
        self.concat_skip = concat_skip
        self.sym = sym

        if isinstance(encode_kernel_sizes, int):
            encode_kernel_sizes = [encode_kernel_sizes] * (self.depth + 1)
        if len(encode_kernel_sizes) != self.depth + 1:
            raise AssertionError
        self.encode_kernel_sizes: List[int] = encode_kernel_sizes

        self.num_channels = [self.num_channel_initial * (2**d) for d in range(self.depth + 1)]
        self.min_extract_level = min(self.extract_levels)

        # init layers
        # all lists start with d = 0
        self.encode_convs: nn.ModuleList
        self.encode_pools: nn.ModuleList
        self.bottom_block: nn.Sequential
        self.decode_deconvs: nn.ModuleList
        self.decode_convs: nn.ModuleList
        self.output_block: nn.Module
        #print(self.sym)
        if self.sym:
            self.output_sym_block: nn.Module

        # build layers
        self.build_layers()

    def build_layers(self):
        self.build_encode_layers()
        self.build_decode_layers()

    def build_encode_layers(self):
        # encoding / down-sampling
        self.encode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=self.in_channels if d == 0 else self.num_channels[d - 1],
                    out_channels=self.num_channels[d],
                    kernel_size=self.encode_kernel_sizes[d],
                )
                for d in range(self.depth)
            ]
        )
        self.encode_pools = nn.ModuleList(
            [self.build_down_sampling_block(channels=self.num_channels[d]) for d in range(self.depth)]
        )
        self.bottom_block = self.build_bottom_block(
            in_channels=self.num_channels[-2], out_channels=self.num_channels[-1]
        )

    def build_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                norm=None,
            ),
            #RegistrationResidualConvBlockZero(
            #    spatial_dims=self.spatial_dims,
            #    in_channels=out_channels,
            #    out_channels=out_channels,
            #    kernel_size=kernel_size,
            #),
        )

    def build_down_sampling_block(self, channels: int):
        return RegistrationDownSampleBlock(spatial_dims=self.spatial_dims, channels=channels, pooling=self.pooling)

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,                
                norm=None,
            ),
        )

    def build_decode_layers(self):
        self.decode_deconvs = nn.ModuleList(
            [
                self.build_up_sampling_block(in_channels=self.num_channels[d + 1], out_channels=self.num_channels[d])
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )
        self.decode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=(2 * self.num_channels[d] if self.concat_skip else self.num_channels[d]),
                    out_channels=self.num_channels[d],
                    kernel_size=3,
                )
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )
        # extraction
        self.output_block = self.build_output_block()
        if self.sym:
            self.output_sym_block = self.build_output_block()

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def build_output_block(self) -> nn.Module:
        return RegistrationExtractionBlock(
            spatial_dims=self.spatial_dims,
            extract_levels=self.extract_levels,
            num_channels=self.num_channels,
            out_channels=self.out_channels,
            kernel_initializer=self.out_kernel_initializer,
            activation=self.out_activation,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])
        Returns:
            Tensor in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3]), with the same spatial size as ``x``
        """
        image_size = x.shape[2:]
        skips = []  # [0, ..., depth - 1]
        encoded = x
        for encode_conv, encode_pool in zip(self.encode_convs, self.encode_pools):
            encoded = encode_pool(encode_conv(encoded))
            
        decoded = self.bottom_block(encoded)

        outs = [decoded]

        for i, (decode_deconv, decode_conv) in enumerate(zip(self.decode_deconvs, self.decode_convs)):
            decoded = decode_conv(decode_deconv(decoded))
            outs.append(decoded)

        out = self.output_block(outs, image_size=image_size)
        if self.sym:
            out2 = self.output_sym_block(outs, image_size=image_size)
            return out, out2
        else:        
            return out    
            
            
class TrilinearLocalNetZero(RegUNetZero):
    """
    Reimplementation of LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        pooling: bool = True,
        use_addictive_sampling: bool = True,
        concat_skip: bool = False,
        sym: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            use_addictive_sampling: whether use additive up-sampling layer for decoding.
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
        """
        self.use_additive_upsampling = use_addictive_sampling
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=extract_levels,
            depth=max(extract_levels),
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=[7] + [3] * max(extract_levels),
            sym=sym,
        )
        

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm=None,
        )

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels
            )

        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels) 