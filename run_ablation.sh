#!/usr/bin/env bash
# 消融实验队列（串行执行，每个配置只跑 Fold 1，约 60-70 分钟/个）
#   E3_ste     : CGL 直通梯度修复（前向与 E1 完全一致，仅补回梯度）—— 检验 CGL 若可反传是否有增益
#   E2_cgl_w3  : CGL 权重减半 —— 三点对照（0 / 3 / 6）
#   E4_iter1   : 单轮迭代（L_max=1）—— 迭代式 ECG 机制消融
set -u
cd "$(dirname "$0")/_repo" || exit 1
PY=../.venv/Scripts/python.exe

COMMON="--dataset_type EventStoryLine --cache_path ./data/database/exist \
--model_name ./model/bert-base-uncased --no_of_classes 2 --t_lr 2e-5 --lr 1e-4 \
--w 0.6 --threshold 2 --epoch 30 --batch_size 1 --beta_intra 0.3 --beta_inter 0.7 \
--min_iteration 2 --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0 --grad_ckpt"

run () {
  name="$1"; shift
  echo "===== START ${name} $(date '+%F %T') ====="
  $PY -u main.py $COMMON --method "$name" --fold_start 1 --fold_end 1 "$@"
  echo "===== END ${name} exit=$? $(date '+%F %T') ====="
}

run E3_ste_termgrad --cgl_w 6.0 --cgl_mode ste --max_iteration 9
run E2_cgl_w3       --cgl_w 3.0 --cgl_mode hard --max_iteration 9
run E4_iter1        --cgl_w 6.0 --cgl_mode hard --max_iteration 1

echo "ALL DONE $(date '+%F %T')"
