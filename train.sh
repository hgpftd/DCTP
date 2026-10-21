export BASICSR_JIT=True
export PYTHONPATH=$PYTHONPATH:/home/XX/DCTP

CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 --use_env basicsr_dctp/train.py -opt options/DCTP.yml --auto_resume --launcher pytorch
