import argparse
import logging
import sys

import numpy as np
import torch
from monai.networks.blocks import Warp
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

import utils_parser
from loss import antifolding_loss
from loss import compute_reg_train_loss
from loss import jacobian_loss
from reg_data import getRegistrationDataset
from reg_model import getRegistrationModel
from utils import compute_mean_dice, getAdamOptimizer, getReducePlateauScheduler, loadExistingModel, getDevice, \
    getMaskedImage, get_noise_transforms, compute_model_input, get_symcompare_loss, compute_distance_ddfs
from utils import print_model_output, print_weights, add_weights_to_name, compute_landmarks_distance_local, \
    setFreezeParameters, TBLogger, add_noise_to, get_noise_loss, get_sym_loss, weight_histograms

from models import TrilinearLocalNet
import nibabel as nib
from torchinfo import summary


def train(modelname, dataset, ft, ct, batchsize, max_epochs, lr, patience, weights, registration_type, atlas, mask, pt,
          newmodel, validfeminad, freeze, cycle_consistent_training, use_jacobian_loss, affine_consistent_training,
          use_antifolding_loss, use_ddf, inverse_consistent_training, ddf_consistent_training, sym, small, ith, fold, noaug, meantemplate):

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    #modelname = add_weights_to_name(modelname, weights)
    modelname = modelname + '.pth'
    print_model_output(modelname)
    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = getDevice()

    sym_warp = Warp("bilinear", "zeros").to(device)
    sym_warp_nearest = Warp("nearest", "zeros").to(device)
    sym_warp_reflection = Warp("bilinear", "reflection").to(device)

    if small:
        img_size = small
    else:
        img_size = 128

    model = getRegistrationModel(registration_type, pretrain_model=pt, use_ddf=use_ddf, sym=sym, newmodel=newmodel, img_size=small)

    # channels = 16
    # extract = [0, 1, 2, 3]
    # model = TrilinearLocalNet(spatial_dims=3,in_channels=2,out_channels=3,num_channel_initial=channels,extract_levels=extract,out_activation=None,out_kernel_initializer="zeros",sym=sym)
    # summary(model, input_size=(1,2,128,128,128))

    optimizer = getAdamOptimizer(model, lr)
    scheduler = getReducePlateauScheduler(optimizer, factor=0.1, patience=patience)
    if ft or ct:
        weights = loadExistingModel(model, optimizer, ft, ct, weights=weights, registration=True)
    print_weights(weights)

    if noaug:
        use_augment = False
    else:
        use_augment = True
    dataloaders, size = getRegistrationDataset(dataset=dataset,batch=batchsize,training=True,augment=use_augment,
                                               eval_augment=False,atlas=atlas,mask=mask,validfeminad=validfeminad,
                                               noise=ddf_consistent_training,img_size=small,ith=ith,fold=fold,
                                               meantemplate=meantemplate)

    writer = SummaryWriter(comment='_' + modelname)
    # sizelol = 33

    noise_smoothing_transform, noise_biasfield_transform, noise_rescaling_transform = get_noise_transforms()

    metric_logger = TBLogger(sym, ddf_consistent_training, use_jacobian_loss, use_antifolding_loss)
    
    phases = ['train']#, 'valid']
    
    for epoch in range(-1, max_epochs):
        setFreezeParameters(model, freeze)
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        metric_logger.reset_train()
        metric_logger.reset_valid()
        sym = 0.001
        if epoch > 40:
            sym = 0.01
        if epoch > 80:
            sym = 0.1
        if epoch > 120:
            sym = 1.0
        if epoch > 160:
            sym = 10
        if epoch > 200:
            sym = 100
        print(str(epoch) + ' - ' + str(sym))
        for phase in phases:
            sizelol = len(dataloaders[phase])
            if epoch == -1 and phase == 'train':
                continue
            if pt is None:
                model.update_phase(phase)
            else:
                model.freeze_affine(phase)

            metric_logger.reset_running()

            for i, data in enumerate(dataloaders[phase]):
                #if i >= sizelol:
                #    break
                #if epoch == 0 and i == 0:
                #    writer.add_graph(model.__class__.__bases__[0], input_to_model=data, verbose=False)
                #if phase == 'valid':
                #    weight_histograms(writer, epoch, model)

                print(i, end='\r')
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    moving_image = data['moving_image'].to(device)
                    moving_label = data['moving_label'].to(device)
                    moving_image_masked, moving_mask = getMaskedImage(moving_image, moving_label, device)

                    fixed_image = data['fixed_image'].to(device)
                    fixed_label = data['fixed_label'].to(device)
                    fixed_image_masked, fixed_mask = getMaskedImage(fixed_image, fixed_label, device)
                    
                    print(fixed_image.shape)
                    print(fixed_label.shape)
                    print(moving_image.shape)
                    print(moving_label.shape)

                    if registration_type.lower() == 'affine' or registration_type.lower() == 'local' or registration_type.lower() == 'localzero' or registration_type.lower() == 'null':
                        ddf, pred_image, pred_label, dvf, ddf2, dvf2 = model(data)
                    # elif registration_type.lower() == 'deformable':
                    #     affine_ddf, ddf, pred_image, pred_label, affine_image, affine_label = model(data)
                    #     affine_image_masked = getMaskedImage(affine_image, affine_label, device)
                    pred_image_masked, _ = getMaskedImage(pred_image, fixed_label, device)
                    _, pred_mask = getMaskedImage(pred_image, pred_label, device)

                    if sym:
                        predsym_image = sym_warp(fixed_image, ddf2).to(device, non_blocking=True)
                        predsym_label = sym_warp_nearest(fixed_label, ddf2).to(device, non_blocking=True)
                        predsym_image_masked, _ = getMaskedImage(predsym_image, moving_label, device)
                        # _, predsym_mask = getMaskedImage(predsym_image, predsym_label, device)

                    # if "neatin" in dataset:
                    #     fixed_regions = data['fixed_regions'].to(device, non_blocking=True)
                    #     fixed_regions_np = fixed_regions.cpu().detach().numpy().squeeze()
                    #     moving_regions = data['moving_regions'].to(device, dtype=torch.float, non_blocking=True)
                    #     pred_regions = model.warp_nearest(moving_regions, ddf)
                    #     pred_regions_np = pred_regions.cpu().detach().numpy().squeeze()

                    img_loss, lbl_loss, ddf_loss = compute_reg_train_loss(pred_image_masked, None, fixed_image_masked, None, ddf, dvf, weights, registration_type.lower(), use_ddf)
                    #img_loss, lbl_loss, ddf_loss = compute_reg_train_loss(pred_image, None, fixed_image, None, ddf, dvf, weights, registration_type.lower(), use_ddf)
                    loss = img_loss + lbl_loss + ddf_loss
                    metric_logger.updates(["img_loss", "lbl_loss", "ddf_loss"], [img_loss.item(), lbl_loss.item(), ddf_loss.item()])
                    if sym:
                        sym_img_loss, _, sym_ddf_loss = compute_reg_train_loss(predsym_image_masked, None, moving_image_masked, None, ddf2, dvf2, weights, registration_type.lower(), use_ddf)
                        sym_inv_loss = get_sym_loss(ddf, ddf2, sym, sym_warp_reflection, img_size)
                        loss += sym_img_loss + sym_ddf_loss + sym_inv_loss
                        metric_logger.updates(["sym_img_loss", "sym_ddf_loss", "sym_inv_loss"], [sym_img_loss.item(), sym_ddf_loss.item(), sym_inv_loss.item()/sym])
                    if use_jacobian_loss:
                        jcb_loss = jacobian_loss(ddf) / (img_size*img_size*img_size)
                        loss += jcb_loss
                        metric_logger.updates(["jcb_loss"], [jcb_loss.item()])
                    if use_antifolding_loss:
                        fold_loss = antifolding_loss(ddf)
                        loss += fold_loss
                        metric_logger.updates(["fold_loss"], [fold_loss.item()])

                    dice_metric = compute_mean_dice(pred_mask, fixed_mask)
                    metric_logger.updates(["dice"], [dice_metric.item()])
                    if registration_type.lower() == 'local' and phase == 'valid' and (validfeminad or dataset == 'feminad'):
                        metric = np.mean(compute_landmarks_distance_local(ddf, data, small=small)[1:6])
                    # elif registration_type.lower() == 'local' and "neatin" in dataset:
                    #     metric = evaluate(fixed_regions_np, pred_regions_np, metric="DSC", multi_class=True, n_classes=41)
                    #     metric = np.mean(metric)
                    elif "fake" in dataset:
                        metric = compute_distance_ddfs(ddf, data, fixed_mask, small=small)
                    else:
                        metric = torch.zeros(1)
                    metric_logger.updates(["metric"], [metric.item()])
                    torch.cuda.empty_cache()

                    if not ddf_consistent_training:
                        if phase == 'train':
                            loss.backward()
                        metric_logger.update("loss", loss.item())
                    else:
                        if registration_type.lower() == 'affine' or registration_type.lower() == 'local':

                            noise_moving_image = add_noise_to(data['original_image'], noise_biasfield_transform, noise_smoothing_transform, noise_rescaling_transform)
                            noise_moving_image = noise_moving_image.to(device)

                            noise_ddf, noise_pred_image, noise_pred_label, noise_dvf, _, _ = model(compute_model_input(fixed_image, fixed_label, noise_moving_image, moving_label))

                            noise_pred_image_masked, _ = getMaskedImage(noise_pred_image, fixed_label, device)
                            # _, noise_pred_mask = getMaskedImage(noise_pred_image, noise_pred_label, device)

                            noise_img_loss, _, noise_ddf_loss = compute_reg_train_loss(noise_pred_image_masked, None, fixed_image_masked, None, noise_ddf, noise_dvf, weights, registration_type.lower(), use_ddf)
                            #loss += noise_img_loss + noise_ddf_loss
                            metric_logger.updates(["noise_img_loss", "noise_ddf_loss"],[noise_img_loss.item(), noise_ddf_loss.item()])

                            noise_ddfcompare_loss = get_noise_loss(noise_ddf, ddf, ddf_consistent_training)
                            loss += noise_ddfcompare_loss
                            metric_logger.updates(["noise_ddfcompare_loss"], [noise_ddfcompare_loss.item()])

                            if phase == 'train':
                                loss.backward()
                            metric_logger.update("loss", loss.item())
                            torch.cuda.empty_cache()

                    # if inverse_consistent_training:
                    #     if registration_type.lower() == 'affine' or registration_type.lower() == 'local':
                    #         inv_ddf, inv_pred_image, inv_pred_label, inv_dvf = model(
                    #             {"fixed_image": data["moving_image"], "fixed_label": data["moving_label"],
                    #              "moving_image": data["fixed_image"], "moving_label": data["fixed_label"]})
                    #
                    #         inv_pred_image = inv_pred_image.to(device, non_blocking=True)
                    #         inv_pred_label = inv_pred_label.to(device, non_blocking=True)
                    #         inv_pred_mask = AsDiscrete(threshold=0.5)(inv_pred_label)
                    #         inv_pred_image_masked = MaskIntensity(mask_data=inv_pred_mask)(inv_pred_image)
                    #
                    #         inv_fixed_image = data['moving_image'].to(device, non_blocking=True)
                    #         inv_fixed_label = data['moving_label'].to(device, non_blocking=True)
                    #         inv_fixed_mask = AsDiscrete(threshold=0.5)(inv_fixed_label)
                    #         inv_fixed_image_masked = MaskIntensity(mask_data=inv_fixed_mask)(inv_fixed_image)
                    #
                    #         if use_ddf:
                    #             inv_img_loss, inv_lbl_loss, inv_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 inv_pred_image_masked,
                    #                 inv_pred_mask,
                    #                 inv_fixed_image_masked,
                    #                 inv_fixed_mask,
                    #                 inv_ddf,
                    #                 weights)
                    #         else:
                    #             inv_img_loss, inv_lbl_loss, inv_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 inv_pred_image_masked,
                    #                 inv_pred_mask,
                    #                 inv_fixed_image_masked,
                    #                 inv_fixed_mask,
                    #                 inv_dvf,
                    #                 weights)
                    #
                    #         inv_loss = inv_img_loss + inv_lbl_loss + inv_ddf_loss
                    #         if phase == 'train':
                    #             inv_loss.backward()
                    #         loss += inv_loss
                    #
                    #         del inv_ddf, inv_pred_image, inv_pred_label, inv_dvf, inv_fixed_image, inv_fixed_label
                    #         torch.cuda.empty_cache()
                    #
                    # if cycle_consistent_training:
                    #     print("NOT WORKING ANYMORE!!!")
                    #     ########## NOT WORKING ANYMORE DUE TO PREDIMAGE CHANGE FROM INV CONSISTENT TRAINIG FOR MEMORY
                    #     cycle_data = {
                    #         "fixed_image": data["moving_image"],
                    #         "fixed_label": data["moving_label"],
                    #         "moving_image": pred_image,
                    #         "moving_label": pred_label,
                    #     }
                    #     if registration_type.lower() == 'local':
                    #         cycle_ddf, cycle_pred_image, cycle_pred_label, cycle_dvf = model(cycle_data)
                    #         cycle_pred_image = cycle_pred_image.to(device, non_blocking=True)
                    #         cycle_pred_label = cycle_pred_label.to(device, non_blocking=True)
                    #         cycle_pred_mask = AsDiscrete(threshold=0.5)(cycle_pred_label)
                    #         cycle_pred_image_masked = MaskIntensity(mask_data=cycle_pred_mask)(cycle_pred_image)
                    #
                    #         cycle_fixed_image = data["moving_image"].to(device, non_blocking=True)
                    #         cycle_fixed_label = data["moving_label"].to(device, non_blocking=True)
                    #         cycle_fixed_mask = AsDiscrete(threshold=0.5)(cycle_fixed_label)
                    #         cycle_fixed_image_masked = MaskIntensity(mask_data=cycle_fixed_mask)(cycle_fixed_image)
                    #
                    #         if use_ddf:
                    #             cycle_img_loss, cycle_lbl_loss, cycle_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 cycle_pred_image_masked,
                    #                 cycle_pred_mask,
                    #                 cycle_fixed_image_masked,
                    #                 cycle_fixed_mask,
                    #                 cycle_ddf,
                    #                 weights)
                    #         else:
                    #             cycle_img_loss, cycle_lbl_loss, cycle_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 cycle_pred_image_masked,
                    #                 cycle_pred_mask,
                    #                 cycle_fixed_image_masked,
                    #                 cycle_fixed_mask,
                    #                 cycle_dvf,
                    #                 weights)
                    #         cycle_loss = cycle_img_loss + cycle_lbl_loss + cycle_ddf_loss
                    #         if use_jacobian_loss:
                    #             cycle_jcb_loss = jacobian_loss(cycle_ddf) / (128 * 128 * 128)
                    #             cycle_loss = cycle_loss + cycle_jcb_loss
                    #         if use_antifolding_loss:
                    #             cycle_fold_loss = antifolding_loss(cycle_ddf)
                    #             cycle_loss = cycle_loss + cycle_fold_loss
                    #         if phase == 'train':
                    #             cycle_loss.backward()
                    #         loss += cycle_loss
                    #
                    # if affine_consistent_training:
                    #     print("NOT WORKING ANYMORE!!!")
                    #     ########## NOT WORKING ANYMORE DUE TO PREDIMAGE CHANGE FROM INV CONSISTENT TRAINIG FOR MEMORY
                    #     randaffine_transform = RandAffine(
                    #         mode='bilinear',
                    #         prob=1.0,
                    #         rotate_range=(np.pi / 90, np.pi / 90, np.pi / 90),
                    #         scale_range=(0.05, 0.05, 0.05),
                    #         translate_range=(2, 2, 2),
                    #     )
                    #     randaffine_smoothing_transform = RandGaussianSmooth(
                    #         prob=1.0,
                    #         sigma_x=(1, 2),
                    #         sigma_y=(1, 2),
                    #         sigma_z=(1, 2),
                    #     )
                    #
                    #     randaffine_biasfield_transform = RandBiasField(
                    #         prob=1.0,
                    #         coeff_range=(0.7, 1.0),
                    #     )
                    #
                    #     randaffine_moving_image = randaffine_transform(data["moving_image"][0, :, :, :, :]).unsqueeze(0)
                    #
                    #     randaffine_matrix = randaffine_transform.rand_affine_grid.get_transformation_matrix()
                    #     randaffine_transform_nearest = Affine(
                    #         mode='nearest',
                    #         affine=randaffine_matrix
                    #     )
                    #     randaffine_moving_label, _ = randaffine_transform_nearest(data["moving_label"][0, :, :, :, :])
                    #     randaffine_moving_label = randaffine_moving_label.unsqueeze(0)
                    #     randaffine_data = {
                    #         "fixed_image": fixed_image,
                    #         "fixed_label": fixed_label,
                    #         "moving_image": randaffine_moving_image,
                    #         "moving_label": randaffine_moving_label,
                    #     }
                    #
                    #     if registration_type.lower() == 'local':
                    #         randaffine_ddf, randaffine_pred_image, randaffine_pred_label, randaffine_dvf = model(
                    #             randaffine_data)
                    #         randaffine_pred_image = randaffine_pred_image.to(device, non_blocking=True)
                    #         randaffine_pred_label = randaffine_pred_label.to(device, non_blocking=True)
                    #         randaffine_pred_mask = AsDiscrete(threshold=0.5)(randaffine_pred_label)
                    #         randaffine_pred_image_masked = MaskIntensity(mask_data=randaffine_pred_mask)(
                    #             randaffine_pred_image)
                    #
                    #         if use_ddf:
                    #             affine_img_loss, affine_lbl_loss, affine_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 randaffine_pred_image_masked,
                    #                 randaffine_pred_mask,
                    #                 fixed_image_masked,
                    #                 fixed_mask,
                    #                 randaffine_ddf,
                    #                 weights)
                    #         else:
                    #             affine_img_loss, affine_lbl_loss, affine_ddf_loss = get_deformable_registration_loss_from_weights(
                    #                 randaffine_pred_image_masked,
                    #                 randaffine_pred_mask,
                    #                 fixed_image_masked,
                    #                 fixed_mask,
                    #                 randaffine_dvf,
                    #                 weights)
                    #
                    #         loss2 = affine_img_loss + affine_lbl_loss + affine_ddf_loss
                    #
                    #         affine_loss = compute_affine_loss_vincent(ddf, randaffine_ddf, randaffine_matrix)
                    #         affine_loss_2 = compute_affine_loss(ddf, randaffine_ddf, randaffine_matrix)
                    #         loss = loss + loss2 + affine_consistent_training * affine_loss.type(torch.cuda.DoubleTensor)

                    if phase == 'train':
                        optimizer.step()
                        torch.cuda.empty_cache()

                metric_logger.one_pass_complete(data["fixed_image"].size(0))

            metric_logger.half_epoch_complete(sizelol, phase)

            if phase == 'train': #(phase == 'valid' and not validfeminad) or (phase == 'train' and validfeminad):
                scheduler.step(metric_logger.return_running("running_loss"))
                metric_logger.check_new_best_loss(epoch, model, optimizer, weights, lr, modelname)
                torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'weights': weights,
            'epoch': epoch,
            'lr': lr,
               },
                './models/' + modelname.replace('.pth','_last.pth')
               )

        metric_logger.write_metrics_to_TB(writer, epoch)

    metric_logger.print_train_complete(writer)


def parseArguments():
    parser = argparse.ArgumentParser(description="Train 3D mouse brain registration model.")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="Specify learning rate.")
    parser.add_argument("-p", "--patience", type=int, default=500, help="Specify patience for LR Plateau.")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Max epochs for training")
    parser.add_argument("-o", "--output", help="Model name for save")
    parser.add_argument("-d", "--dataset", default='IRIS', help="Dataset name for training.")
    parser.add_argument("-ft", "--finetuning", help="Load existing model for finetuning.")
    parser.add_argument("-ct", "--continuetraining", help="Load existing model to continue training.")
    parser.add_argument("-pt", "--pretraining", help="Load existing model for affine registration.")
    parser.add_argument("-t", "--type", type=str, help="Specify affine/deformable/local registration")
    parser.add_argument("-a", "--atlas", action='store_true',
                        help="Perform to-atlas registration instead of paired registration")
    parser.add_argument("-m", "--mask", action='store_true', help="Skullstrip dataset if available")
    parser.add_argument("-w", "--weights", nargs='+', type=float, default=[1.0, 0, 2.0],
                        help="Loss weights for 1) ImageLoss 2) LabelLoss 3) DDF. Default : [1,1,1]")
    parser.add_argument("-newmodel", "--newmodel", action='store_true',
                        help="True: Depth 5 Channels 32; False: Depth 4 Channels 16")
    parser.add_argument(
        "-validfeminad", "--validfeminad", action='store_true',
        help="True: Validate on Feminad with Landmarks")
    parser.add_argument("-freeze", "--freeze", type=int, default=0, help="Freeze Xth layer")

    parser.add_argument("-cycleconsistenttraining", "--cycleconsistenttraining", action='store_true',
                        help="UseCycleConsistentTraining")
    parser.add_argument("-inverseconsistenttraining", "--inverseconsistenttraining", action='store_true',
                        help="UseInverseConsistentTraining")
    parser.add_argument("-affineconsistenttraining", "--affineconsistenttraining", type=float, default=0.0,
                        help="UseAffineConsistentTraining")
    parser.add_argument("-ddfconsistenttraining", "--ddfconsistenttraining", type=float, default=0.0,
                        help="UseDDFConsistentTraining")

    parser.add_argument("-jacobianloss", "--jacobianloss", action='store_true', help="UseJacobianDetLoss")
    parser.add_argument("-antifoldingloss", "--antifoldingloss", action='store_true', help="UseAntiFoldingLoss")
    parser.add_argument("-ddf", "--ddf", action='store_true', help="Use DDF instead of DVF2DDF")
    parser.add_argument("-i", "--inverse", action='store_true', help="Use inverse in loss")
    parser.add_argument("-sym", "--sym", type=float, default=0.0, help="Use symmetric model")
    parser.add_argument("-s", "--small", type=int, default=128, help="Use small dataset for faster testing")
    parser.add_argument("-ith", "--ith", type=int, default=0, help="Use i-th image for 1-image training")
    parser.add_argument("-kfold", "--kfold", type=int, default=0, help="Kth-Fold for validating on GIN dataset")
    parser.add_argument("-noaug", "--noaug", action='store_true', help="Dont use augment for training")
    parser.add_argument("-meantemplate", "--meantemplate", action='store_true', help="Use mean template instead of allen")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    utils_parser.handleModelOutputParser(args.output)
    utils_parser.handleDatasetParser(args.dataset)
    utils_parser.handleFinetuningParser(args.finetuning, args.continuetraining)

    utils_parser.handleRegistrationTypeParser(args.type)
    args.weights = utils_parser.handleLossesWeightsParser(args.weights, args.type)
    utils_parser.handleAtlasRegistrationParser(args.atlas)
    utils_parser.handleSkullstripRegistrationParser(args.mask)

    utils_parser.handleAffinePretrainingParser(args.pretraining, args.type)

    if args.small != 128:
        print("=> Using small images for faster testing")

    if args.newmodel:
        print("=> Using new model (d5channels32)")
    else:
        print("=> Using old model (d4channels16)")

    if args.validfeminad:
        print("=> Validating on Feminad")

    if args.jacobianloss:
        print("=> Use jacobian loss")

    if args.cycleconsistenttraining:
        print("=> Use cycle consistent training")

    if args.ddfconsistenttraining:
        print("=> Use ddf consistent training")

    if args.affineconsistenttraining:
        print("=> Use affine consistent training")

    if args.inverseconsistenttraining:
        print("=> Use inverse consistent training")

    if args.ddf:
        print("=> Use DDF")
    else:
        print("=> Use DVF2DDF")

    if args.kfold:
        if args.dataset.lower() != 'gi3n':
            sys.exit(1)
        else:
            print("=> Using kfold on GIN")
            print("=> Validating on fold " + str(args.kfold))

    if args.sym:
        print("=> Using symmetric models")

    if args.ith != 0:
        print("=> Using i-th image for 1-image training")

    train(args.output, args.dataset, args.finetuning, args.continuetraining, args.batchsize, args.epochs,
          args.learningrate, args.patience, args.weights, args.type, args.atlas, args.mask, args.pretraining,
          args.newmodel,
          args.validfeminad, args.freeze, args.cycleconsistenttraining, args.jacobianloss,
          args.affineconsistenttraining, args.antifoldingloss,
          args.ddf, args.inverseconsistenttraining, args.ddfconsistenttraining, args.sym,
          args.small, args.ith, args.kfold, args.noaug, args.meantemplate)
