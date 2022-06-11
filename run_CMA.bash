# # TRAINING (Single GPU)
# flag="--exp_name cont-cwp-cma-ori
#       --run-type train
#       --exp-config run_CMA.yaml
#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]
#       IL.batch_size 16
#       IL.lr 1e-4
#       IL.epochs 50
#       IL.schedule_ratio 0.75
#       "
# python run.py $flag


# # TRAINING (Single node multiple GPUs)
# flag="--exp_name cont-cwp-cma-ori
#       --run-type train
#       --exp-config run_CMA.yaml
#       GPU_NUMBERS 2
#       SIMULATOR_GPU_IDS [0,1]
#       TORCH_GPU_IDS [0,1]
#       IL.batch_size 16
#       IL.lr 1e-4
#       IL.epochs 50
#       IL.schedule_ratio 0.75
#       "
# python -m torch.distributed.launch --nproc_per_node=2 run.py $flag


# # EVALUATION
flag="--exp_name cont-cwp-cma-ori
      --run-type eval
      --exp-config run_CMA.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      EVAL_CKPT_PATH_DIR logs/checkpoints/cont-cwp-cma-ori/cma_ckpt_best.pth
      "
python run.py $flag


# # INFERENCE
# flag="--exp_name cont-cwp-cma-ori
#       --run-type inference
#       --exp-config run_CMA.yaml
#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]
#       EVAL.SAVE_RESULTS False
#       INFERENCE.PREDICTIONS_FILE test
#       INFERENCE.SPLIT test
#       INFERENCE.CKPT_PATH logs/checkpoints/cont-cwp-cma-ori/cma_ckpt_best.pth
#       "
# python run.py $flag
