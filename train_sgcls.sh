CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 10035 --nproc_per_node=4 \
    tools/relation_train_net.py --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
    MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree \
    SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 4 \
    DTYPE float16 SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
    SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR /home/ubuntu/ssd/scenegraph/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/ubuntu/ssd/scenegraph/checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR /home/ubuntu/ssd/scenegraph/checkpoints/sgcls-joint-casual-tde-vctree-baseline \
    MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 1024 \
    SOLVER.PRE_VAL False  MODEL.BASE_ONLY True 

