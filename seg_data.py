import os
import sys

import transforms_dict
from utils import get_batch_size, get_file_lists, getDataloader, getCacheDataset, get_segmentation_splits, get_file_lists_labels


def getSegmentationTransforms(augment, eval_augment, crop=False, labels=False):
    if augment or eval_augment:
        print('=> Using augmented transforms for train set')
        train_transforms = transforms_dict.getSegmentationTrainingTransforms(crop, labels)
    else:
        train_transforms = transforms_dict.getSegmentationValidationTransforms(crop, labels)
    if eval_augment:
        print("=> Using augmented transforms for valid/test set")
        valid_transforms = transforms_dict.getSegmentationTrainingTransforms(crop, labels)
    else:
        valid_transforms = transforms_dict.getSegmentationValidationTransforms(crop, labels)
    return train_transforms, valid_transforms


def getSegmentationNEATDataset(batch, training, augment, eval_augment, exvivo=True):
    if exvivo:
        data_dir = os.path.join(os.getcwd(), 'dataset3', 'NeAt-Ex')
    else:
        data_dir = os.path.join(os.getcwd(), 'dataset3', 'NeAt-In')
    train_len, val_len, test_len = get_segmentation_splits("NeAt")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len)



def getSegmentationCERMEPDataset(batch, training, augment, eval_augment):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'CERMEP')
    train_len, val_len, test_len = get_segmentation_splits("CERMEP")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  nii=True)


def getSegmentationIRISDataset(batch, training, augment, eval_augment, n4=False, resample=False, crop=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'IRIS')
    train_len, val_len, test_len = get_segmentation_splits("IRIS")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  n4=n4, resample=resample)


def getSegmentationPainfactDataset(batch, training, augment, eval_augment, n4=False, resample=False, labels=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Painfact')
    train_len, val_len, test_len = get_segmentation_splits("Painfact")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  n4=n4, resample=resample, labels=labels)


def getSegmentationFeminadDataset(batch, training, augment, eval_augment, n4=False, resample=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Feminad')
    train_len, val_len, test_len = get_segmentation_splits("Feminad")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  n4=n4, resample=resample)
                                  
def getSegmentationFemina3Dataset(batch, training, augment, eval_augment, n4=False, resample=False, labels=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Femina3')
    train_len, val_len, test_len = get_segmentation_splits("Femina3")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  n4=n4, resample=resample, labels=labels)                               

def getSegmentationGINDataset(batch, training, augment, eval_augment, n4=False, resample=False, labels=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'GIN')
    train_len, val_len, test_len = get_segmentation_splits("GIN")
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  n4=n4, resample=resample, labels=labels)


def getSegmentationMultipleDataset(batch, training, augment, eval_augment, dataset, resample=False):
    data_dir = []
    if "iris" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'IRIS'))
    if "painfact" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Painfact'))
    if "feminad" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Feminad'))
    if "femina3" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Femina3'))    
    if "gin" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'GIN'))
    return handleSegmentationData(data_dir, batch, training, augment, eval_augment, 0, 0, 0,
                                  resample=resample)


def handleSegmentationData(data_dir, batch, training, augment, eval_augment, train, val, test,
                           nii=False, n4=False, crop=False, resample=False, labels=False
                           ):
    if type(data_dir) == str:
        if not labels:
            mris, masks = get_file_lists(data_dir, nii=nii, n4=n4, resample=resample, labels=labels)
            data_dicts = [
                {
                    "img": mris[idx],
                    "seg": masks[idx],
                }
                for idx in range(len(mris))
            ]
            train_files = data_dicts[:train]
            valid_files = data_dicts[train:train+val]
            test_files = data_dicts[train+val:train+val+test]
        else:
            mris, masks = get_file_lists(data_dir, nii=nii, n4=n4, resample=resample, labels=labels)
            labels = get_file_lists_labels(data_dir)
            data_dicts = [
                {
                    "img": mris[idx],
                    "seg": labels[idx],
                }
                for idx in range(len(mris))
            ]
            train_files = data_dicts[:train]
            valid_files = data_dicts[train:train + val]
            test_files = data_dicts[train + val:train + val + test]
    else:
        train_files, valid_files, test_files = [], [], []
        for d_dir in data_dir:
            mris, masks, = get_file_lists(d_dir, nii=nii, n4=n4, resample=resample, labels=labels)
            train, val, test = get_segmentation_splits(d_dir)
            data_dicts = [
                {
                    "img": mris[idx],
                    "seg": masks[idx],
                }
                for idx in range(len(mris))
            ]
            train_files += data_dicts[:train]
            valid_files += data_dicts[train:train+val]
            test_files += data_dicts[train+val:train+val+test]

    train_transforms, valid_transforms = getSegmentationTransforms(augment, eval_augment, crop=crop, labels=labels)
    batch_size = get_batch_size(batch, training)

    train_ds, valid_ds, test_ds = getCacheDataset(train_files=train_files, train_transforms=train_transforms,
                                                  valid_files=valid_files, valid_transforms=valid_transforms,
                                                  test_files=test_files, test_transforms=valid_transforms
                                                  )

    train_loader, valid_loader, test_loader = getDataloader(train_data=train_ds, valid_data=valid_ds, test_data=test_ds,
                                                            batch_size=batch_size, shuffle=training)

    dataloader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    size = {'train': len(train_ds), 'valid': len(valid_ds), 'test': len(test_ds)}

    return dataloader, size


def getSegmentationDataset(dataset, batch, training, augment, eval_augment, n4=False, resample=False, labels=False):
    lowd = str(dataset).lower()
    # TODO : Handle Crop again, if needed
    multiple = ["painfact" in lowd, "iris" in lowd, "feminad" in lowd, "gin" in lowd, "femina3" in lowd]
    if lowd == 'gin':
        return getSegmentationGINDataset(batch, training, augment, eval_augment, n4=n4, resample=resample, labels=labels)
    elif lowd == 'iris':
        return getSegmentationIRISDataset(batch, training, augment, eval_augment, n4=n4, resample=resample)
    elif lowd == 'painfact':
        return getSegmentationPainfactDataset(batch, training, augment, eval_augment, n4=n4, resample=resample, labels=labels)
    elif lowd == 'feminad':
        return getSegmentationFeminadDataset(batch, training, augment, eval_augment, n4=n4, resample=resample)
    elif lowd == 'femina3':
        return getSegmentationFemina3Dataset(batch, training, augment, eval_augment, n4=n4, resample=resample, labels=labels)    
    elif sum(multiple) >= 2:
        return getSegmentationMultipleDataset(batch, training, augment, eval_augment, lowd, resample=resample)
    else:
        print("Couldn't find dataset: " + str(dataset))
        sys.exit(1)
