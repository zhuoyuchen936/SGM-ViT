#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd /home/pdongaa/workspace/SGM-ViT
source .venv/bin/activate
python3 scripts/train_fusion_net.py   --stage both   --arch mask_residual_lite   --out-root artifacts/fusion_phase1_loss_fix   --sceneflow-epochs 20   --kitti-epochs 10   2>&1 | tee artifacts/fusion_phase1_loss_fix_train.log
echo 'TRAINING DONE'
sleep 3600
