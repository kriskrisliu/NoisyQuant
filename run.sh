GPU=$1
MODEL=$2
calib_num=$3
IN_ROOT=$4

# for example:
# GPU=0
# MODEL=vit_base_patch16_224
# calib_num=256
# IN_ROOT="/DATASETS/imagenet"

mkdir -p LOG

# easyQuant method
CUDA_VISIBLE_DEVICES=$GPU python validate.py ${IN_ROOT}/val/ --model $MODEL --pretrained --quant --with_noisy_quant --calib_root ${IN_ROOT}/train --calib_num $calib_num --percentile > LOG/log-${MODEL}-calib${calib_num}-percentile.log

# easyQuant + mean bias
CUDA_VISIBLE_DEVICES=$GPU python validate.py ${IN_ROOT}/val/ --model $MODEL --pretrained --quant --with_noisy_quant --calib_root ${IN_ROOT}/train --calib_num $calib_num --percentile --search_mean > LOG/log-${MODEL}-calib${calib_num}-percentile-mean.log

# easyQuant + noisy bias
CUDA_VISIBLE_DEVICES=$GPU python validate.py ${IN_ROOT}/val/ --model $MODEL --pretrained --quant --with_noisy_quant --calib_root ${IN_ROOT}/train --calib_num $calib_num --percentile --search_noisy > LOG/log-${MODEL}-calib${calib_num}-percentile-noisy.log

# easyQuant + shifted noisy bias
CUDA_VISIBLE_DEVICES=$GPU python validate.py ${IN_ROOT}/val/ --model $MODEL --pretrained --quant --with_noisy_quant --calib_root ${IN_ROOT}/train --calib_num $calib_num --percentile --search_mean --search_noisy > LOG/log-${MODEL}-calib${calib_num}-percentile-mean-noisy.log

# show final results
echo "LOG/log-${MODEL}-calib${calib_num}-percentile.log"
cat LOG/log-${MODEL}-calib${calib_num}-percentile.log | grep \"top1\":
echo "LOG/log-${MODEL}-calib${calib_num}-percentile-mean.log"
cat LOG/log-${MODEL}-calib${calib_num}-percentile-mean.log | grep \"top1\":
echo "LOG/log-${MODEL}-calib${calib_num}-percentile-noisy.log"
cat LOG/log-${MODEL}-calib${calib_num}-percentile-noisy.log | grep \"top1\":
echo "LOG/log-${MODEL}-calib${calib_num}-percentile-mean-noisy.log"
cat LOG/log-${MODEL}-calib${calib_num}-percentile-mean-noisy.log | grep \"top1\":