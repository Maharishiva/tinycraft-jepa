#!/bin/bash
# Start the second campaign worker on GPU0 only once the unrelated tinywmcraft
# confirmation run has fully exited and no compute app holds that device.
set -u
cd /home/rldoeswork/personal/tinycraft-jepa-iclr-20260919
PY=/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python
OUT=runs/revision/shared_decoded
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
DEADLINE=$(date -u -d "2026-09-20T07:30:00Z" +%s)
while true; do
  foreign=$(pgrep -fc "confirm.sh|tinywm-pr2v2-confirm" || true)
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID" || true)
  if [ "$foreign" = "0" ] && [ "$busy" = "0" ]; then
    echo "[$(date -u +%FT%TZ)] GPU0 free; starting worker"
    exec $PY -u studies/revision/queue_shared_decoded.py --gpu 0 --admission-file studies/revision/shared_decoded_admission.json --python $PY
  fi
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    echo "[$(date -u +%FT%TZ)] deadline reached while GPU0 stayed busy (foreign=$foreign busy=$busy); not starting"
    exit 0
  fi
  sleep 60
done
