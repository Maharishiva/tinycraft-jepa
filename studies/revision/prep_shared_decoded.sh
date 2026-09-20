#!/bin/bash
# Shared-CNN campaign preparation on one GPU: gray-world full-label head refits,
# codec parity for the six remaining world/family pairs, and timing smokes.
set -u
cd /home/rldoeswork/personal/tinycraft-jepa-iclr-20260919
GPU=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.90 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
PY=/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python
OUT=runs/revision/shared_decoded
STOCH=/home/rldoeswork/personal/stoch
GRAY=/dev/shm/redcube
step() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { step "FAILED: $*"; echo FAILED > $OUT/prep.STATE; exit 1; }
echo RUNNING > $OUT/prep.STATE
mkdir -p $OUT/validation $OUT/smoke
for w in gray_frame_still gray_frame_wind gray_episode_wind; do
  if [ ! -f $OUT/heads/$w/model_probe_fullheads.npz ]; then
    step "refit heads $w"
    $PY -m studies.revision.refit_full_label_heads --model $GRAY/$w/tile_factored/model_probe.npz --train $GRAY/$w/train.npz --out $OUT/heads/$w/model_probe_fullheads.npz --world $w > $OUT/head_$w.log 2>&1 || fail "heads $w"
    tail -1 $OUT/head_$w.log | cut -c1-300
  fi
  step "codec jepa $w"
  $PY -m studies.revision.check_shared_decoded --family jepa --model $GRAY/$w/tile_factored/model_probe.npz --heads $OUT/heads/$w/model_probe_fullheads.npz --packet $OUT/packets/$w.npz --config $w --out $OUT/validation/${w}_jepa_codec.json > $OUT/validation/${w}_jepa_codec.log 2>&1 || fail "codec jepa $w"
  tail -1 $OUT/validation/${w}_jepa_codec.log | cut -c1-300
  step "codec flow $w"
  $PY -m studies.revision.check_shared_decoded --family flow --model $STOCH/$w/flow/model.npz --packet $OUT/packets/$w.npz --config $w --out $OUT/validation/${w}_flow_codec.json > $OUT/validation/${w}_flow_codec.log 2>&1 || fail "codec flow $w"
  tail -1 $OUT/validation/${w}_flow_codec.log | cut -c1-300
done
smoke() {
  local name=$1 w=$2 fam=$3 src=$4; shift 4
  step "smoke $name"
  $PY -u -m studies.revision.shared_decoded_ppo --config $w --family $fam --train-source $src "$@" --eval-packet $OUT/packets/$w.npz --out $OUT/smoke/$name --policy-seed 0 --updates 2 --smoke > $OUT/smoke_$name.log 2>&1 || fail "smoke $name"
  grep '^\[final\]' $OUT/smoke_$name.log | cut -c1-420
}
smoke gate_jepa_dream gate jepa dream --model $STOCH/gate/tile_factored_s1/model_probe.npz --heads $OUT/heads/gate/model_probe_fullheads.npz --data $STOCH/gate/train.npz
smoke gate_flow_dream gate flow dream --model $STOCH/gate/flow/model.npz --data $STOCH/gate/train.npz
smoke gate_flow_real gate flow real --model $STOCH/gate/flow/model.npz --data $STOCH/gate/train.npz
smoke gray_frame_still_flow_dream gray_frame_still flow dream --model $STOCH/gray_frame_still/flow/model.npz --data $GRAY/gray_frame_still/train.npz
step "PREP DONE"
echo DONE > $OUT/prep.STATE
