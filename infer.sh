export BASICSR_JIT=True
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --model_path /home/lihaifeng/BFR/MDGAN/experiments/DCTP/models/net_g_10000.pth \
  --input /home/lihaifeng/MDGAN0/deg_all_lr \
  --output /home/lihaifeng/HSI_SOTA/a_codes/results 