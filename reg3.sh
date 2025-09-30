#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 2000 -d feminad -o template_affine_feminad_newloss						
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 214 -d painfactfeminad -o template_affine_painfactfeminad_newloss				
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 149 -d irisfeminad -o template_affine_irisfeminad_newloss					
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 92 -d painfactirisfeminad -o template_affine_painfactirisfeminad_newloss	 		
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 240 -d painfact -o template_affine_painfact_newloss	
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 1 0 -lr 0.0001 -t affine -e 161 -d iris -o template_affine_iris_newloss						

# MRIs / Epochs
# 33 => 2000
# 309 => 214
# 445 => 149
# 721 => 92
# 276 => 240
# 412 => 161