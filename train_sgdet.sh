#!/bin/zsh

#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p p100
#SBATCH --job-name=sgdet_train
#SBATCH --output=slurm_out/train_sgdet_vctree_%A-%a.out

source activate sgebm
python -m torch.distributed.launch --master_port 10021 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /scratch/hdd001/home/suhail/sgebm/glove MODEL.PRETRAINED_DETECTOR_CKPT ~/ScenGraphEBM/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ~/ScenGraphEBM/checkpoints/vctree-sgdet-exmp
