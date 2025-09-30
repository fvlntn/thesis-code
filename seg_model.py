import monai

from utils import getDevice


def getUNetForExtraction():
    device = getDevice()
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def getUNetForSegmentation():
    device = getDevice()
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def getUNETRForSegmentation(img_size=128):
    device = getDevice()
    model = monai.networks.nets.UNETR(
        in_channels=1,
        out_channels=4,
        img_size=(img_size, img_size, img_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    return model
