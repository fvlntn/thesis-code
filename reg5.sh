CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.0001 -t local -e 300 -d fakeaffine -newmodel -o fake_scenario1_masked
##################
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.0001 -t local -e 300 -d fakeaffine -newmodel -o fake_scenario1_masked

