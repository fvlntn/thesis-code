#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import sys

import monai
import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
from monai.utils import set_determinism
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from seg_data import getSegmentationDataset
from transforms_dict import getSegmentationPostProcessingForLabel, getSegmentationPostProcessingForLabelOutput
from utils import getReducePlateauScheduler, loadExistingModel
from utils import print_model_output, check_model_name, getDevice

# In[2]:


#Parameters
modelname = "test_seg_labels_unetr.pth"
dataset = "painfactlabels"
ft = None
ct = None
batchsize = 1
num_epochs = 200
factor = 0.9
patience = 10
augment = True
N4 = False


# In[3]:


#Modelname and device
torch.multiprocessing.set_sharing_strategy('file_system')
modelname = check_model_name(modelname)
print_model_output(modelname)
set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
device = getDevice()


# In[4]:


#postprocessing
outputs_processing = getSegmentationPostProcessingForLabelOutput()
labels_processing = getSegmentationPostProcessingForLabel()

#activations
softmax = Activations(other=nn.Softmax(dim=1))


# In[5]:


#dataloaders
dataloaders, size = getSegmentationDataset(dataset=dataset, batch=batchsize, augment=augment, training=True, n4=N4)


# In[6]:


def getUNETRForSegmentation():        
    device = getDevice()              
    model = monai.networks.nets.UNETR(
        in_channels=1,
        out_channels=4,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,           
    ).to(device)                      
    return model                      


# In[7]:


#Metric MUNet
def metric_munet(preds, labels):  
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()    
    labels[np.where(labels == np.amax(labels, axis=0))] = 1
    labels[labels != 1] = 0
    dice=2*np.sum(labels*preds,(1,2,3))/(np.sum((labels+preds),(1,2,3))+1)    
    return dice


# In[8]:


#Loss
def loss_Dice(preds, labels):
    dice = 1-torch.div(
        torch.sum(torch.mul(torch.mul(labels,preds),2)),
        torch.sum(torch.mul(preds,preds)) + torch.sum(torch.mul(labels,labels))
        )    
    return dice

def loss_CE(input, target):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()        
        device = getDevice()
        weight = torch.FloatTensor([1.0, 5.0, 5.0, 20.0]).to(device, non_blocking=True)
        return nn.CrossEntropyLoss(reduction="mean", weight=weight)(input, target)

loss_GDice = monai.losses.GeneralizedDiceLoss(other_act=nn.Softmax(dim=1))
weight = torch.FloatTensor([1.0, 5.0, 5.0, 20.0]).to(device, non_blocking=True)
loss_DiceCE = monai.losses.DiceCELoss(other_act=nn.Softmax(dim=1), ce_weight=weight)

def loss_GDiceCE(input, target, lambda_gdice=1.0, lambda_ce=1.0):    
    GDice = loss_GDice(input, target)
    CE = loss_CE(input, target)    
    GDiceCELoss = lambda_gdice*GDice + lambda_ce*CE
    return GDiceCELoss


# In[9]:


#Train loop
best_loss = np.inf
writer = SummaryWriter()

torch.backends.cudnn.benchmark = True

#Model optimizer and scheduler
model = getUNETRForSegmentation()
lr = 5e-4#/np.sqrt(6)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = getReducePlateauScheduler(optimizer, patience=patience, factor=factor)
loadExistingModel(model, optimizer, ft, ct)

train_dices = []
valid_dices = []

for epoch in range(num_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")

    train_loss = 0
    valid_loss = 0

    for phase in ['train', 'valid']:
        model.train()
        
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0        
        metrics = []
        for i in range(4):
            metrics.append([])

        for i, data in enumerate(dataloaders[phase]):
            print("{}/{}".format(
                i, len(dataloaders[phase])), end='\r'
            )
            optimizer.zero_grad()      
            
            with torch.set_grad_enabled(phase == 'train'):
                inputs, labels = data["img"].to(device, non_blocking=True), data["seg"].to(device, non_blocking=True)
                labels = labels.squeeze(2) 
                
                if phase == 'train':        
                    outputs = model(inputs)
                else:
                    outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)                
                loss = loss_GDiceCE(outputs, labels)
                #loss = loss_DiceCE(outputs, labels)
                preds = [outputs_processing(pred) for pred in decollate_batch(outputs)]
                labels = [labels_processing(label) for label in decollate_batch(labels)]  
                for j in range(len(preds)):
                    metric = metric_munet(preds[j], labels[j]) 
                    
                    for k in range(4):
                        metrics[k].append(metric[k])                

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()

        running_loss /= size[phase]
        metrics_mean = [np.mean(x) for x in metrics]
        running_metric = np.mean(metrics_mean)      

        print(
            "{}: loss: {:.4f}, dice: {:.4f}".format(
                phase, running_loss, running_metric
            )
        )
        print(
            "dices: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                metrics_mean[0], metrics_mean[1], metrics_mean[2], metrics_mean[3]
            )
        )
        
        if phase == 'train':
            train_loss = running_loss               
            train_dices.append(running_metric)
        elif phase == 'valid':
            valid_loss = running_loss 
            valid_dices.append(running_metric)
            scheduler.step(running_loss)
            if running_loss < best_loss:
                best_loss = running_loss
                best_epoch = epoch + 1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                    './models/' + str(modelname))

                print(
                    "best loss {:.4f} at epoch {}".format(
                        best_loss, best_epoch
                    )
                )            
    writer.add_scalars('epoch_loss', {
        'train': train_loss,
        'valid': valid_loss,
    }, epoch + 1)
    

print(f"train completed")
writer.close()


# In[ ]:


for pred in preds:
    print(pred.shape)


# In[ ]:


x = outputs[0,:,64,64,64]
print(x)
x = preds[0][:,64,64,64]
print(x)
x = labels[0][:,64,64,64]
print(x)
print(outputs.shape)
print(preds.shape)
print(labels.shape)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Mean Dice")
x = [(i + 1) for i in range(len(train_dices))]
y = train_dices
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [(i + 1) for i in range(len(valid_dices))]
y = valid_dices
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()


# In[ ]:


import nibabel as nib
from glob import glob
import os
labels = sorted(glob(os.path.join('dataset', 'Painfact-Segmentation', 'Label', "*.nii.gz")))
weights_a = []
weights_b = []
weights_c = []
weights_d = []
for i in range(len(labels)):
    print(i, end='\r')
    label = nib.load(labels[i]).get_fdata()
    a = np.where(label == 0)[0].shape[0]
    b = np.where(label == 1)[0].shape[0]
    c = np.where(label == 2)[0].shape[0]
    d = np.where(label == 3)[0].shape[0]
    e = 305*216*227
    weight_a = (1/a) * e/4
    weight_b = (1/b) * e/4
    weight_c = (1/c) * e/4
    weight_d = (1/d) * e/4    
    weights_a.append(weight_a)
    weights_b.append(weight_b)
    weights_c.append(weight_c)
    weights_d.append(weight_d)
    


# In[ ]:


print(np.mean(weights_a)/np.mean(weights_a))
print(np.mean(weights_b)/np.mean(weights_a))
print(np.mean(weights_c)/np.mean(weights_a))
print(np.mean(weights_d)/np.mean(weights_a))


# In[ ]:





# In[ ]:


from glob import glob
import os
mris = sorted(glob(os.path.join("dataset", "Painfact-Segmentation", 'MRI', "*.nii.gz")))
masks = sorted(glob(os.path.join("dataset", "Painfact-Segmentation", 'Label', "*.nii.gz")))
mri = mris[0]
mask = masks[0]


# In[ ]:


from monai.transforms import Compose, RandRotate90d, CropForegroundd, RandCropByPosNegLabeld
from monai.transforms import LoadImaged, AddChanneld, ScaleIntensityd, EnsureTyped, RandShiftIntensityd
from monai.transforms import RandRotated

from transforms import GetLabelsAsOneHotd, Shaped

# In[ ]:


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"],
            minv=0.0, 
            maxv=1.0,
        ),        
        CropForegroundd(
            keys=["image", "label"], 
            source_key="label",
            margin=5,
        ),    
        Shaped(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),           
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
            spatial_axes=(1, 2),
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
            spatial_axes=(2, 0),
        ),
        RandRotated(
            keys=["image", "label"],
            range_x=np.pi/4,
            range_y=np.pi/4,
            range_z=np.pi/4,
            prob=0.25,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),   
        GetLabelsAsOneHotd(
                keys=["label"],
                get=True,
                skip=False,
            ),
        EnsureTyped(
                keys=["image", "label"],
        ),
    ]
)


# In[ ]:


for i in range(len(mris)):
    print("{}/{}".format(
        i, len(mris)), end='\r'
    )
    mdr = {'image': mris[i], 'label': masks[i]}
    mdr2 = train_transforms(mdr)
    print('-'*10)

