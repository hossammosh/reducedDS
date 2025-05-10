# #!/bin/sh
# #SBATCH --job-name=train
# #SBATCH --account=g.hlwn032  # Your account name
# #SBATCH --partition=gpu  # Specify GPU partition
# #SBATCH --ntasks=1  # Number of tasks (usually 1)
# #SBATCH --cpus-per-task=4  # Number of CPU cores per task
# #SBATCH --output=output.log  # Output log file
# #SBATCH --error=errors.log  # Error log file
# #SBATCH --nodes=2  # Number of nodes
# #SBATCH --gres=gpu:1  # Number of GPUs per node
# #SBATCH --time=240:00:00  # Expected running time
#
#
#
# # Activate your Conda environment
#
# # Ensure CUDA is using the correct GPU
# #export CUDA_VISIBLE_DEVICES=0,1
#
# # Check GPU status before training
# echo "Checking GPU status before training"
# nvidia-smi
#
# # Run your Python training script
# echo "Starting training"
# --nproc_per_node=1 --nnodes=2
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2  lib/train/run_training.py --script seqtrack --config seqtrack_b256 --save_dir .
# ##python -m torch.distributed.launch --nproc_per_node 2 lib/train/run_training.py --script seqtrack --config seqtrack_b256 --save_dir .
#
# # Check GPU status after training
# echo "Checking GPU status after training"
#
#
