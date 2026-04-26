#!/bin/bash
# Phase 8 Pareto: sequentially train 6 variants, profile, eval. No user input required.
set -e
cd /home/pdongaa/workspace/SGM-ViT
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

OUT_ROOT=artifacts/fusion_phase8_pareto
PROFILE_DIR=results/phase8_pareto/profile
EVAL_DIR=results/phase8_pareto/eval
mkdir -p "$PROFILE_DIR" "$EVAL_DIR"

IN_CH=${PHASE8_IN_CHANNELS:-7}  # override to 8 if B5 gate fails

# Variants: (backbone, head_ch)
VARIANTS=(
  "b0 24"
  "b0 48"
  "b1 24"
  "b1 48"
  "b2 24"
  "b2 48"
)

for v in "${VARIANTS[@]}"; do
  read -r BB HC <<< "$v"
  NAME="${BB}_h${HC}"
  OUT_DIR="$OUT_ROOT/$NAME"
  echo "========================================"
  echo "[phase8] $(date +%T)  $NAME  (in_ch=$IN_CH)"
  echo "========================================"

  # Skip if final eval already exists.
  if [ -f "$EVAL_DIR/$NAME.json" ]; then
    echo "[phase8] $NAME already evaluated; skipping."
    continue
  fi

  # Default batch + crop. B2 needs smaller batch for memory.
  BATCH_SF=8
  BATCH_MIX=4
  if [ "$BB" == "b2" ]; then
    BATCH_SF=6
    BATCH_MIX=3
  fi

  # Stage 1: SF pretrain
  if [ ! -f "$OUT_DIR/sf_pretrain/best.pt" ]; then
    echo "[phase8] $NAME SF pretrain ..."
    python scripts/train_effvit.py \
      --stage sf_pretrain --cache-root artifacts/fusion_cache_v3 \
      --dataset sceneflow --train-split train --val-split val \
      --epochs 20 --lr 3e-4 --batch-size $BATCH_SF \
      --variant $BB --head-ch $HC --in-channels $IN_CH \
      --hole-aug-prob 0.5 --hole-aug-max-frac 0.3 \
      --out-dir "$OUT_DIR/sf_pretrain" 2>&1 | tee "/tmp/phase8_${NAME}_sf.log"
  fi

  # Stage 2: mixed finetune
  if [ ! -f "$OUT_DIR/mixed_finetune/best.pt" ]; then
    echo "[phase8] $NAME mixed finetune ..."
    python scripts/train_effvit.py \
      --stage mixed_finetune --cache-root artifacts/fusion_cache_v3 \
      --mixed-datasets kitti,sceneflow --mixed-weights 1.0,1.0 \
      --mixed-val-dataset kitti --val-split val \
      --epochs 15 --lr 5e-5 --batch-size $BATCH_MIX \
      --variant $BB --head-ch $HC --in-channels $IN_CH \
      --hole-aug-prob 0.3 --hole-aug-max-frac 0.3 \
      --init-ckpt "$OUT_DIR/sf_pretrain/best.pt" \
      --out-dir "$OUT_DIR/mixed_finetune" 2>&1 | tee "/tmp/phase8_${NAME}_mix.log"
  fi

  # Profile (params + GFLOPs)
  python scripts/profile_effvit.py \
    --ckpt "$OUT_DIR/mixed_finetune/best.pt" \
    --input-hw 384 768 \
    --out-json "$PROFILE_DIR/$NAME.json"

  # Eval on 4 datasets
  python scripts/eval_effvit.py \
    --ckpt "$OUT_DIR/mixed_finetune/best.pt" \
    --cache-root artifacts/fusion_cache_v3 \
    --out-dir "results/phase8_pareto/eval_$NAME" 2>&1 | tee "/tmp/phase8_${NAME}_eval.log"
  # Normalize eval summary location.
  cp "results/phase8_pareto/eval_$NAME/summary.json" "$EVAL_DIR/$NAME.json"

  echo "[phase8] $(date +%T)  $NAME done."
done

# Aggregate + plot
echo "[phase8] $(date +%T) aggregating Pareto ..."
python scripts/pareto_analyze.py \
  --profile-dir "$PROFILE_DIR" \
  --eval-dir "$EVAL_DIR" \
  --out-dir "results/phase8_pareto"

echo "[phase8] ALL DONE $(date +%T)"
echo DONE > /tmp/phase8_pareto.done
