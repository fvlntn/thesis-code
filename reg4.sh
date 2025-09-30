CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 500 -d neatinaffine -o neatin_scenario1_maskedloss -newmodel
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 500 -d neatinaffine -o neatin_scenario1_continue -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 200 -d neatinaffine -freeze 2 -o neatin_scenario3 -ft scenario-smoothloss/smoothloss_local_scenario2_painfact_newmodel_1.0-0.0-8.0.pth -newmodel
