#!/bin/bash
#SBATCH --job-name=eaagn
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-16:00:00
#SBATCH --array=1-6
#SBATCH --output=logs/%x_%A_%a.out
#
# Striding array worker (scheduler-native; no login-node daemons). Each task runs
# joblist lines i with (i-1) % NTASKS == TASK_ID-1. Lines are idempotent
# (train.py skips when summary.json exists; stages resume from checkpoints).
set -uo pipefail
RUN_ROOT="${RUN_ROOT:-$RUN_ROOT}"
BASE="${BASE:-$RUN_ROOT/code}"
JOBLIST="${JOBLIST:?set JOBLIST}"
cd "$BASE"
source "${VENV:-$RUN_ROOT/venv}/bin/activate"
export PYTHONUTF8=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[worker] host=$(hostname) task=$TASK_ID/$NTASKS joblist=$JOBLIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
i=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    if [ $(( i % NTASKS )) -eq $(( TASK_ID - 1 )) ]; then
        echo "[worker $TASK_ID] line $((i+1)): $line"
        eval "$line" || echo "[worker $TASK_ID] FAILED line $((i+1)): $line"
    fi
    i=$((i+1))
done < "$JOBLIST"
echo "[worker $TASK_ID] done"
