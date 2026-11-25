#!/bin/bash
#SBATCH --job-name=eaagn_cpu
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU latency benchmarks: 4 dedicated cores reserved, 1 compute thread used (no core sharing).
set -uo pipefail
RUN_ROOT="${RUN_ROOT:-$RUN_ROOT}"
BASE="${BASE:-$RUN_ROOT/code_v2}"
JOBLIST="${JOBLIST:?set JOBLIST}"
cd "$BASE"
source "${VENV:-$RUN_ROOT/venv}/bin/activate"
export PYTHONUTF8=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"; NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[cpu] host=$(hostname) $(lscpu | grep 'Model name')"
i=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    if [ $(( i % NTASKS )) -eq $(( TASK_ID - 1 )) ]; then
        echo "[cpu $TASK_ID] $line"; eval "$line" || echo "[cpu $TASK_ID] FAILED: $line"
    fi
    i=$((i+1))
done < "$JOBLIST"
