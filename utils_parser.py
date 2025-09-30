import sys


def handleFinetuningParser(finetuning, continuetraining):
    if finetuning is not None and continuetraining is None:
        print("=> Loading " + str(finetuning) + " for finetuning")

    if finetuning is None and continuetraining is not None:
        print("=> Loading " + str(continuetraining) + " to continue training")


def handleRegistrationTypeParser(registration_type, training=True):
    if registration_type.lower() != 'affine' and registration_type.lower() != 'deformable' and registration_type.lower() != "local" and registration_type.lower() != 'localzero' and registration_type.lower() != 'null' :
        print("Should specify model type as affine, deformable or local")
        sys.exit(1)
    if training:
        print("=> Training " + registration_type.lower() + " registration model")


def handleAffinePretrainingParser(pretraining, registration_type):
    if pretraining is not None and registration_type.lower() == 'deformable':
        print("=> Loading " + str(pretraining) + " for affine registration")
    elif pretraining is not None and registration_type.lower() != 'deformable':
        print("=> Can only use pretrained affine for affine+deformable model")
        sys.exit(1)


def handleDatasetParser(dataset):
    if dataset is None:
        print("Should specify dataset. See -h for instructions.")
        sys.exit(1)


def handleModelOutputParser(model):
    if model is None:
        print("Should specify model name. See -h for instructions.")
        sys.exit(1)


def handleLossesWeightsParser(weights, registration_type):
    if len(weights) != 3:
        print("Weights are in the wrong format. -w 10 1 1 => 10 / 1 / 1 weights.")
        sys.exit(1)
    if registration_type.lower() == 'affine':
        new_weights = [weights[0], weights[1], 0]
    else:
        new_weights = weights
    return new_weights


def handleAtlasRegistrationParser(atlas):
    if atlas:
        print('=> Performing to-atlas registration.')
    else:
        print('=> Performing paired registration.')


def handleSkullstripRegistrationParser(skullstrip):
    if skullstrip:
        print('=> Applying skullstripping preprocessing')
