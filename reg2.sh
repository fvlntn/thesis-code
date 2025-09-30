####################################################################################################################################
# First results (1 0 2)

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario_feminad	
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 500 -d painfactaffine -o smoothloss_local_scenario_painfact
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 200 -d feminadaffine -o smoothloss_local_scenario_painfact_finetune_freeze -ft scenario/smoothloss_local_scenario_painfact_1.0-0.0-2.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 200 -d feminadaffine -o smoothloss_local_scenario_painfact_finetune_nofreeze -ft scenario/smoothloss_local_scenario_painfact_1.0-0.0-2.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 500 -d painfactaffine -o smoothloss_local_scenario_painfact_newmodel


####################################################################################################################################
# Second results (1 0 8) with new augmentation + new model

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario1_feminad_newmodel -newmodel

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d painfactaffine -o smoothloss_local_scenario2_painfact_newmodel -newmodel -validfeminad
#cp models/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth models/scenario-smoothloss/

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 2 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze2 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 1 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze1 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 3 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze3 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 4 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze4 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 5 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze5 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 6 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze6 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 7 -o smoothloss_local_scenario3_painfact_finetune_newmodel_freeze7 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel


#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario1_feminad

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d painfactaffine -o smoothloss_local_scenario2_painfact -validfeminad

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 2 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze2 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 1 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze1 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 3 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze3 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 4 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze4 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 5 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze5 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 6 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze6 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -freeze 7 -o smoothloss_local_scenario3_painfact_finetune_oldmodel_freeze6 -ft scenario-fix-108/smoothloss_local_scenario2_painfact_1.0-0.0-8.0.pth

####################################################################################################################################
# Third results with JacDet loss / AntiFolding Loss / CycleConsistent training

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_newmodel_jac -jacobianloss -newmodel
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_newmodel_fold -antifoldingloss -newmodel

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_cycle -cycleconsistenttraining
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_jac -jacobianloss
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_fold -antifoldingloss
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_base
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_cycle_jac -cycleconsistenttraining -jacobianloss
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 500 -d feminadaffine -o smoothloss_local_scenario4_feminad_oldmodel_cycle_fold -cycleconsistenttraining -antifoldingloss

#########
# Fourth
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 1000 -d feminadaffine -o local_overfit_feminad_old_all -affineconsistenttraining -antifoldingloss -jacobianloss
#
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 1000 -d painfactaffine -o local_deeplearning_painfact_old -validfeminad
#cp models/local_deeplearning_painfact_old_1.0-0.0-8.0.pth models/paper/
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 400 -d feminadaffine -freeze 2 -o local_finetune_feminad_old_freeze2 -ft paper/local_deeplearning_painfact_old_1.0-0.0-8.0.pth -affineconsistenttraining


######################
# Paper Affine 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 1000 -d feminadaffine -o new_local_overfit_feminad_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 200 -d feminadaffine -o new_local_overfit_feminad_old_aff -affineconsistenttraining

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 1000 -d feminadaffine -o new_local_overfit_feminad_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 1000 -d feminadaffine -o new_local_overfit_feminad_old_aff -affineconsistenttraining

######################
# Test New Dataset
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 2 -d feminadaffine -o testnewdataset_1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 2 -d feminadginaffine -o testnewdataset_2 -affineconsistenttraining
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 2 -d feminadgin -o testnewdataset_3 -affineconsistenttraining
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 2 -d feminadginirispainfactaffine -o testnewdataset_4

################
# Test Scenarios with new Dataset

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 300 -d painfactatffine -o scenario2_old  -validfeminad
#cp models/scenario2_old_1.0-0.0-0.1.pth models/test-new/
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 2 -o scenario3_old -ft test-new/scenario2_old_1.0-0.0-0.1.pth

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d painfactaffine -o scenario2_old  -validfeminad
#cp models/scenario2_old_1.0-0.0-8.0.pth models/test-new/
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 2 -o scenario3_old -ft test-new/scenario2_old_1.0-0.0-8.0.pth

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d painfactaffine -o scenario2_old  -validfeminad
#cp models/scenario2_old_1.0-0.0-1.0.pth models/test-new/
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 200 -d feminadaffine -freeze 2 -o scenario3_old -ft test-new/scenario2_old_1.0-0.0-1.0.pth

#############
# Test PaperAffine with new dataset


#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old_aff -affineconsistenttraining 0.01 

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_old_aff -affineconsistenttraining 0.01

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_old_noreg_aff -affineconsistenttraining 0.01  -ft test-new/scenario2_old_1.0-0.0-8.0.pth 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_old_noreg_aff -affineconsistenttraining 0.1 -ft test-new/scenario2_old_1.0-0.0-8.0.pth 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_old_noreg_aff -affineconsistenttraining 1 -ft test-new/scenario2_old_1.0-0.0-8.0.pth 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_old_noreg_aff -affineconsistenttraining 10 -ft test-new/scenario2_old_1.0-0.0-8.0.pth 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_old_noreg_aff -affineconsistenttraining 100 -ft test-new/scenario2_old_1.0-0.0-8.0.pth 

###############
# Test Inverse Consistent Training

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_inv -inverseconsistenttraining

###############

# Test Symetric Model Training

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1 -s 64
# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_sym -sym 1.0 -s 64
# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_sym -s 64
# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 100 -d feminadaffine -o scenario1_sym -sym 1.0 -ddfconsistenttraining 0.1 -s 64

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 4 -lr 0.0001 -t local -e 50 -d feminadaffine -o scenario1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 50 -d feminadaffine -o scenario1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 50 -d feminadaffine -o scenario1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 50 -d feminadaffine -o scenario1_sym -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 50 -d ginaffine -sym -o gin_sym_ft2 -ft scenario1_sym_1.0-0.0-8.0.pth


# Test Inverse Consistent Training + Noise Consistent Training

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 50 -d feminadaffine -o scenario1_sym_noise -sym 0.1 -ddfconsistenttraining 0.1

# Fakedata

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake1 -s 32 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake2 -s 32 -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake3 -s 32 -sym 0.1 -ddfconsistenttraining 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake4 -s 32 -sym 0.1 -ddfconsistenttraining 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake5 -s 32 -ddfconsistenttraining 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake6 -s 32 -sym 0.1 -ddfconsistenttraining 0.01 
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake7 -s 32 -sym 0.1 -ddfconsistenttraining 1.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake8 -s 32 -sym 1.0 -ddfconsistenttraining 1.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake2 -s 32 -sym 1.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake9 -s 32 -ddfconsistenttraining 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake91 -s 32 -ddfconsistenttraining 1.0

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 200 -d fakeaffine -o fake3 -s 32 -sym 0.1 -ddfconsistenttraining 0.1

# Feminad Real Data

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -sym 0.1 -o gin-baseline+sym-0.1+ddf-8.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.001 -lr 0.0001 -t local -e 300 -d ginaffine -sym 0.1 -o gin-baseline+sym-0.1+noise-8.0 -ddfconsistenttraining 0.1

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -o gin-baseline+ddf-8.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 300 -d ginaffine -o gin-baseline+noise-0.1 -ddfconsistenttraining 0.1

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 300 -d ginaffine -o gin-baseline
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0 -lr 0.0001 -t local -e 300 -d ginaffine -sym 0.1 -o gin-baseline+sym-0.1

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -sym 0.1 -o gin-baseline+ddf-8.0+sym-0.1+noise-0.1 -ddfconsistenttraining 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -o gin-baseline+ddf-8.0+noise-0.1 -ddfconsistenttraining 0.1



################
# Test Scenarios with GIN + SYM

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d painfactaffine -o scenario2_ddf+sym -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d painfactaffine -o scenario2_ddf

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -o scenario1_ddf+sym -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -o scenario1_ddf

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -freeze 2 -o scenario3_real_ddf+sym -ft scenario2_ddf+sym_1.0-0.0-8.0.pth -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -freeze 2 -o scenario3_ddf -ft scenario2_ddf_1.0-0.0-8.0.pth


################
# Test Scenarios with FEMINAD

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_feminad_ddf
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -freeze 2 -o scenario3_feminad_ddf -ft scenario2_ddf_1.0-0.0-8.0.pth

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -o scenario1_feminad_ddf+sym -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -freeze 2 -o scenario3_feminad_ddf+sym -ft scenario2_ddf+sym_1.0-0.0-8.0.pth -sym 0.1


## Replace PAINFACT with IRIS

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d irisaffine -o scenario2iris_ddf+sym -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d irisaffine -o scenario2iris_ddf

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -freeze 2 -o scenario3iris_ddf+sym -ft scenario2iris_ddf+sym_1.0-0.0-8.0.pth -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d ginaffine -freeze 2 -o scenario3iris_ddf -ft scenario2iris_ddf_1.0-0.0-8.0.pth

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -freeze 2 -o scenario3iris_feminad_ddf+sym -ft scenario2iris_ddf+sym_1.0-0.0-8.0.pth -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.0001 -t local -e 300 -d feminadaffine -freeze 2 -o scenario3iris_feminad_ddf -ft scenario2iris_ddf_1.0-0.0-8.0.pth



#####################

# Check best lambda1 on GIN: result is 1.0

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.001 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.01 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 10 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 100 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1000 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda1_gin

# Check best lambda2 on GIN: result is 0.01

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym0.001_gin1 -sym 0.001
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym0.01_gin2 -sym 0.01
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym0.1_gin3 -sym 0.1
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym1_gin4 -sym 1.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym10_gin5 -sym 10.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym100_gin6 -sym 100.0
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym1000_gin7 -sym 1000.0

# Test best sym on GIN without aug 

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginaffine -o lambda2_sym0.01_gin_noaug -sym 0.01

# Test best sym-model on GIN+Feminad

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1.0 -lr 0.0001 -t local -e 300 -d ginfeminadaffine -o sym_ginfeminad -sym 0.01

# Test best sym-model on GIN+IRIS

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d ginirisaffine -o sym_giniris -sym 0.01

# Test model without architecture

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t localzero -e 300 -d ginaffine -o test_without_architecture -sym 0.01

# Restart DL-Painfact & DL-Iris with lambda1 = 1 and lambda2 = 0.01 (was lambda1=8 and lambda2=0.1)

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d irisaffine -o dl_iris+sym -sym 0.01
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d painfactaffine -o dl_pain+sym -sym 0.01

# Kfold GIN DL

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d gi3n -o dl_gin1 -kfold 1 -sym 0.001
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d gi3n -o dl_gin2 -kfold 2
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d gi3n -o dl_gin3 -kfold 3

# Pair-null test

# CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.01 -t null -e 300 -d ginaffine -o null_gin

##############################################

# Useless for now
# Check best lambda1 on Feminad: result is 8.0 or 1.0 below

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.001 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.01 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.1 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 10 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 100 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1000 -lr 0.0001 -t local -e 300 -d feminadaffine -o lambda1_feminad

# Test best sym-model on Feminad: 

#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8.0 -lr 0.0001 -t local -e 300 -d feminadaffine -o sym_feminad -sym 0.01
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1.0 -lr 0.0001 -t local -e 300 -d feminadaffine -o sym_feminad -sym 0.01

# Test best sym-model on Feminad3

CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d femina3affine -o feminad3_Mwt_sym100 -sym 100

