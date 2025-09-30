CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 16 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule		
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 8 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule	
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 4 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule	
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 2 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule	
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 1 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule	
CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 1 0 0.5 -lr 0.001 -t local -e 250 -d feminadaffine -o template_local_feminad_schedule				
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 3 1 2 -lr 0.00001 -t local -e 250 -d painfactfeminadaffine -o template_localnet_trilinear_painfactfeminad 			
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 2 0 1 -lr 0.00001 -t local -e 92  -d painfactirisfeminad -o template_localnet_trilinear_painfactirisfeminad 		
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 3 1 20 -lr 0.00001 -t local -e 2500 -d painfactaffine -o template_localnet_trilinear_painfact 					
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 3 1 2 -lr 0.00001 -t local -e 161 -d iris -o template_localnet_trilinear_iris 						
#CUDA_VISIBLE_DEVICES=0 python3 reg_train.py -a -w 3 1 2 -lr 0.00001 -t local -e 149 -d irisfeminad -o template_localnet_trilinear_irisfeminad 					

# MRIs / Epochs
# 33 => 2000
# 309 => 214
# 445 => 149
# 721 => 92
# 276 => 240
# 412 => 161
