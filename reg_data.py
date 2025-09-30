import os
import sys
from glob import glob

from transforms_dict import getRegistrationTrainingTransforms, getRegistrationValidationTransforms
from utils import get_batch_size, get_file_lists, getDataloader, getCacheDataset, get_file_lists_labels


def getRegistrationTransforms(augment, eval_augment, mask=False, noise=False, img_size=False):
    if augment or eval_augment:
        train_transforms = getRegistrationTrainingTransforms(mask, noise, img_size=img_size)
        print('=> Using augmented transforms for train set')
    else:
        train_transforms = getRegistrationValidationTransforms(mask, img_size=img_size)
    if eval_augment:
        valid_transforms = getRegistrationTrainingTransforms(mask, noise, img_size=img_size)
        print("=> Using augmented transforms for valid/test set")
    else:
        valid_transforms = getRegistrationValidationTransforms(mask, img_size=img_size)
    return train_transforms, valid_transforms


def getRegistrationIRISDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'IRIS')
    train_len, val_len, test_len = 2000, 0, 2000  # all dataset as training
    # train_len, val_len, test_len = 248, 82, 82
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)


def getRegistrationPainfactDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Painfact')
    train_len, val_len, test_len = 2000, 0, 2000  # all dataset as training
    # train_len, val_len, test_len = 169, 56, 56
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)


def getRegistrationFeminadDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False, ith=0):
    # data_dir = os.path.join(os.getcwd(), 'dataset3', 'Feminad')
    # FIX: Workaround for reg_tuning without hard coding folder
    cwd = os.path.abspath("/home/valentini/dev/Mousenet/")
    data_dir = os.path.join(cwd, 'dataset3', 'Feminad')

    train_len, val_len, test_len = 33, 0, 33
    # train_len, val_len, test_len = 33, 0, 0
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size, ith=ith)
                                  
                                  
def getRegistrationFemina3Dataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Femina3')
    train_len, val_len, test_len = 2000, 0, 2000  # all dataset as training
    print(data_dir)
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)
                                  

def getRegistrationGINDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False, ith=0, meantemplate=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'GIN')
    train_len, val_len, test_len = 2000, 0, 2000  # all dataset as training
    # train_len, val_len, test_len = 169, 56, 56
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size, ith=ith, meantemplate=meantemplate)


def getRegistrationNeatexDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Neatex')
    train_len, val_len, test_len = 2000, 0, 2000
    # train_len, val_len, test_len = 33, 0, 0
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)


def getRegistrationNeatinDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Neatin')
    train_len, val_len, test_len = 2000, 0, 2000
    # train_len, val_len, test_len = 33, 0, 0
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)


def getRegistrationFakeDataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=True, noise=False, img_size=False):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'Fakedata')
    train_len, val_len, test_len = 2000, 0, 2000
    # train_len, val_len, test_len = 33, 0, 0
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)

def getRegistrationGIN3Dataset(batch, training, augment, eval_augment, atlas=True, mask=False, affine=False, validfeminad=False, noise=False, img_size=False, fold=0):
    data_dir = os.path.join(os.getcwd(), 'dataset3', 'GI3N')
    train_len, val_len, test_len = 18, 9, 0  # 3-fold for 27 mris
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                      atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise,
                                      img_size=img_size, fold=fold)

def getRegistrationMultipleDataset(batch, training, augment, eval_augment, dataset, atlas=True, mask=False,
                                   affine=False, noise=False, img_size=False, ith=0):
    data_dir = []
    if "iris" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'IRIS'))
    if "painfact" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Painfact'))
    if "feminad" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Feminad'))
    if "femina3" in dataset:
        data_dir.append(os.zpath.join(os.getcwd(), 'dataset3', 'Femina3'))
    if "neatex" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Neatex'))
    if "neatin" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Neatin'))
    if "fake" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'Fakedata'))
    if "gin" in dataset:
        data_dir.append(os.path.join(os.getcwd(), 'dataset3', 'GIN'))
    train_len, val_len, test_len = 2000, 0, 2000  # all dataset as training
    return handleRegistrationData(data_dir, batch, training, augment, eval_augment, train_len, val_len, test_len,
                                  atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size, ith=ith)


def handleRegistrationData(data_dir, batch, training, augment, eval_augment, train, val, test,
                           atlas=True, mask=False, nii=False, affine=False, validfeminad=False, noise=False, img_size=False, ith=0, fold=0, meantemplate=False):
    labels = []
    if type(data_dir) == str:
        mris, masks = get_file_lists(data_dir, affine=affine, nii=nii, fold=fold)
        labels = get_file_lists_labels(data_dir, affine=affine)
        ##TODO : IMPLEMENT LABELS
    else:
        mris, masks = [], []
        for d_dir in data_dir:
            mris_tmp, masks_tmp = get_file_lists(d_dir, affine=affine, nii=nii, fold=fold)
            mris += mris_tmp
            masks += masks_tmp

    if len(mris) != len(masks):
        print(len(mris))
        print(len(masks))
        return Exception("MRIs != Masks")

    cwd = os.getcwd()
    if meantemplate:
        print("=> Using meantemplate instead of Allen")
        data_dicts = [
            {
                "fixed_image": os.path.join(cwd, "dataset3", "Atlas", "mean_allreg.nii.gz"),
                "fixed_label": os.path.join(cwd, "dataset3", "Atlas", "Gin_Mask_deformablemean.nii.gz"),
                "moving_image": mris[idx],
                "moving_label": masks[idx],
            }
            for idx in range(len(mris))
        ]
    else:
        data_dicts = [
            {
                "fixed_image": os.path.join(cwd, "dataset3", "Atlas", "P56_Atlas_128_norm_id.nii.gz"),
                "fixed_label": os.path.join(cwd, "dataset3", "Atlas", "P56_Annotation_128_norm_id_mask_dilated.nii.gz"),
                "moving_image": mris[idx],
                "moving_label": masks[idx],
            }
            for idx in range(len(mris))
        ]
        print(os.path.join(cwd, "dataset3", "Atlas", "P56_Atlas_128_norm_id.nii.gz"))

    if noise:
        for idx in range(len(mris)):
            data_dicts[idx]["original_image"] = mris[idx]

    if validfeminad:
        feminad_mris = sorted(glob(os.path.join(cwd, "dataset3", "Feminad", "MRI", "*_affine.nii.gz")))
        feminad_masks = sorted(glob(os.path.join(cwd, "dataset3", "Feminad", "Mask", "*_affine.nii.gz")))
        validfeminad_dicts = [
            {
                "fixed_image": os.path.join(cwd, "dataset3", "Atlas", "P56_Atlas_128_norm_id.nii.gz"),
                "fixed_label": os.path.join(cwd, "dataset3", "Atlas", "P56_Annotation_128_norm_id_mask_dilated.nii.gz"),
                "moving_image": feminad_mris[idx],
                "moving_label": feminad_masks[idx],
            }
            for idx in range(len(feminad_mris))
        ]
        train_files = data_dicts[:train]
        valid_files = validfeminad_dicts
        # test_files = data_dicts[train+val:train+val+test]
    else:
        if ith != 0:
            train_files = data_dicts[ith-1:ith]
            valid_files = data_dicts[ith-1:ith]
            #train_files = [data_dicts[ith-1] for i in range(len(data_dicts))]
            #valid_files = [data_dicts[ith-1] for i in range(len(data_dicts))]
        elif fold != 0:
            if fold == 1:
                train_files = data_dicts[9:27]
                valid_files = data_dicts[0:9]
            if fold == 2:
                train_files = data_dicts[0:9] + data_dicts[18:27]
                valid_files = data_dicts[9:18]
            if fold == 3:
                train_files = data_dicts[0:18]
                valid_files = data_dicts[18:27]
        else:
            train_files = data_dicts[:train]
            valid_files = data_dicts[:train]
        # valid_files = data_dicts[train:train+val]
        # test_files = data_dicts[train+val:train+val+test]

    train_transforms, valid_transforms = getRegistrationTransforms(augment, eval_augment, mask=mask, noise=noise, img_size=img_size)
    batch_size = get_batch_size(batch, training)

    if ith == 0:    
        train_ds, valid_ds, _ = getCacheDataset(train_files=train_files, train_transforms=train_transforms,
                                                  valid_files=valid_files, valid_transforms=valid_transforms,
                                                  test_files=valid_files, test_transforms=valid_transforms,
                                                  paired=not atlas)
        train_loader, valid_loader, _ = getDataloader(train_data=train_ds, valid_data=valid_ds, test_data=valid_ds,
                                                            batch_size=batch_size, shuffle=training)
    else:
        train_ds, valid_ds, test_ds = getCacheDataset(train_files=train_files, train_transforms=train_transforms,
                                                  valid_files=train_files, valid_transforms=train_transforms,
                                                  test_files=train_files, test_transforms=train_transforms,
                                                  paired=not atlas)
        train_loader, valid_loader, test_loader = getDataloader(train_data=train_ds, valid_data=valid_ds, test_data=test_ds,
                                                            batch_size=batch_size, shuffle=False)
    
    dataloader = {'train': train_loader, 'valid': valid_loader, 'test': valid_loader}
    size = {'train': len(train_ds), 'valid': len(valid_ds), 'test': len(valid_ds)}

    return dataloader, size


def getRegistrationDataset(dataset, batch, training, augment, eval_augment, atlas=True, mask=False, validfeminad=False, noise=False, img_size=False, ith=0, fold=0, meantemplate=False):
    lowd = str(dataset).lower()
    multiple = [i in lowd for i in ["painfact", "iris", "feminad", "neatex", "neatin", "fake", "gin", "femina3"]]
    affine = "affine" in lowd
    if affine:
        lowd = lowd.replace("affine", "")
    if lowd == 'iris':
        return getRegistrationIRISDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)
    elif lowd == 'painfact':
        return getRegistrationPainfactDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)
    elif lowd == 'feminad':
        return getRegistrationFeminadDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size, ith=ith)
    elif lowd == 'femina3':
        return getRegistrationFemina3Dataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, validfeminad=validfeminad, noise=noise, img_size=img_size)
    elif lowd == 'neatex':
        return getRegistrationNeatexDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)
    elif lowd == 'neatin':
        return getRegistrationNeatinDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)
    elif lowd == 'fake':
        return getRegistrationFakeDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size)
    elif lowd == 'gin':
        return getRegistrationGINDataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size, ith=ith, meantemplate=meantemplate)
    elif lowd == 'gi3n':
        return getRegistrationGIN3Dataset(batch, training, augment, eval_augment, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size, fold=fold)
    elif sum(multiple) >= 2:
        return getRegistrationMultipleDataset(batch, training, augment, eval_augment, lowd, atlas=atlas, mask=mask, affine=affine, noise=noise, img_size=img_size, ith=ith)
    else:
        print("Couldn't find dataset: " + str(dataset))
        sys.exit(1)
