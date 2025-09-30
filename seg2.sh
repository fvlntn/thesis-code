#python seg_brain_train.py -d IrisFeminad -o brain_irisfeminad.pth 
#python seg_brain_train.py -d IrisFeminad -o brain_resample_irisfeminad.pth -r
#python seg_brain_train.py -d PainfactIrisFeminad -o brain_resample_painfactirisfeminad.pth -r

CUDA_VISIBLE_DEVICES=0 python3 seg_brain_train.py -d irisfemina3 -o brain_new_irisfemina3.pth
CUDA_VISIBLE_DEVICES=0 python3 seg_brain_train.py -d femina3 -o brain_new_femina3.pth
#CUDA_VISIBLE_DEVICES=0 python3 seg_brain_train.py -d iris -o brain_new_iris.pth
