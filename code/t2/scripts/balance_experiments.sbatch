#!/bin/bash

#SBATCH -J spmv_bal
#SBATCH -o spmv_bal_%j.out
#SBATCH -e spmv_bal_%j.err
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=04:00:00

set -uo pipefail

PROCS=(1 2 4 8 16 32 48 64)
PROCS_KNEE=(16 24 32 40 44 48 56 64)
ITERS=5

SYNTH_N=400000
AVG_DEG=20
CONTRAST=8
CLUSTER_FRAC=0.25
N_SPIKES=4
SPIKE_CONTRAST=10000
REPS=4000
REPS_COST=12000
REPS_SPIKE=15000
REPS_COST_SPIKE=40000

EPOCH_MS=50
EPOCH_MS_COST=100
COST_KEEP=0.95

RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
CSV="$RESULTS_DIR/balancing_results.csv"

cd "$SLURM_SUBMIT_DIR"
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

export OMPI_MCA_coll_han_priority=0
export OMPI_MCA_coll_adapt_priority=0

make clean
make BENCH_CSV=1 -j

mkdir -p "$RESULTS_DIR"

export MAL_AFFINITY=0
export MAL_LOG_LEVEL=ERROR
export MAL_LOG_ALL_RANKS=0
export MAL_TIMING=0
export MAL_EPOCH_CHANGE_MODE=1

echo "config,pattern,exec,version,mode,nproc,nproc_active,n,time,errors,iter" > "$CSV"

run_case() {
	local config="$1" pattern="$2" np="$3" bin="$4" reps="$5"; shift 5
	local kv iter out line
	for kv in "$@"; do export "$kv"; done
	local errdir="$RESULTS_DIR/fail_logs"; mkdir -p "$errdir"
	for iter in $(seq 1 "$ITERS"); do
		local errf="$errdir/${config}_${pattern}_np${np}_it${iter}.err"
		out=$(mpirun -n "$np" --bind-to none ./build/"$bin" --synthetic="$SYNTH_N" --avg_deg="$AVG_DEG" --pattern="$pattern" --contrast="$CONTRAST" --cluster_frac="$CLUSTER_FRAC" --n_spikes="$N_SPIKES" --reps="$reps" 2>"$errf")
		line=$(printf '%s\n' "$out" | awk -F, '/^CSV,/{l=$0} END{if(l!="")print substr(l,5)}')
		if [[ -n "$line" ]]; then
			echo "${config},${pattern},${line},${iter}" >> "$CSV"
			echo "[OK] ${config} ${pattern} np=${np} iter=${iter}: ${line}"
			rm -f "$errf"
		else
			echo "${config},${pattern},${bin},NA,NA,${np},NA,NA,NA,NA,${iter}" >> "$CSV"
			echo "[FAIL] ${config} ${pattern} np=${np} iter=${iter} — stderr saved: $errf"
			echo "----- last stderr lines -----"; tail -n 15 "$errf" 2>/dev/null; echo "-----------------------------"
		fi
	done
	for kv in "$@"; do unset "${kv%%=*}"; done
}

echo "=============================================================="
echo " E1: malleable+LB vs static, on BALANCED and IMBALANCED matrices"
echo "=============================================================="
for pattern in uniform clustered; do
	for np in "${PROCS[@]}"; do
		run_case "normal" "$pattern" "$np" spmv_imbal_normal "$REPS"
		run_case "malleable_lb" "$pattern" "$np" spmv_imbal "$REPS" MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=1 MAL_INITIAL_SIZE="$np" MAL_EPOCH_INTERVAL_MS="$EPOCH_MS"
		run_case "malleable_nolb" "$pattern" "$np" spmv_imbal "$REPS" MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=0 MAL_INITIAL_SIZE="$np" MAL_EPOCH_INTERVAL_MS="$EPOCH_MS"
	done
done

echo "=============================================================="
echo " DONE. Results: $CSV"
echo " Plot: time vs nproc, one line per config, faceted by pattern."
echo " uniform -> malleable_lb == normal (no penalty)."
echo " clustered -> malleable_lb < normal (load balancing wins)."
echo "=============================================================="

