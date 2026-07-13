#!/bin/bash

#SBATCH -J heat_knee
#SBATCH -o heat_knee_%j.out
#SBATCH -e heat_knee_%j.err
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=04:00:00
#SBATCH --exclusive

set -uo pipefail

PROCS_KNEE=(8 16 24 32 40 48 56 64)
ITERS=5

CELLS=2000000
STEPS=25000
RESID=4

RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
CSV="$RESULTS_DIR/knee_results.csv"

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
export MAL_STENCIL_RESID_REDUCES=$RESID

echo "config,exec,version,mode,nproc,nproc_active,n,time,errors,iter" > "$CSV"

run_case() {
	local config="$1" np="$2" bin="$3"; shift 3
	local kv iter out line
	for kv in "$@"; do export "$kv"; done
	local errdir="$RESULTS_DIR/fail_logs"; mkdir -p "$errdir"
	for iter in $(seq 1 "$ITERS"); do
		local errf="$errdir/${config}_np${np}_it${iter}.err"
		out=$(mpirun -n "$np" --bind-to none ./build/"$bin" --cells="$CELLS" --steps="$STEPS" 2>"$errf")
		line=$(printf '%s\n' "$out" | awk -F, '/^CSV,/{l=$0} END{if(l!="")print substr(l,5)}')
		if [[ -n "$line" ]]; then
			echo "${config},${line},${iter}" >> "$CSV"
			echo "[OK] ${config} np=${np} iter=${iter}: ${line}"
			rm -f "$errf"
		else
			echo "${config},${bin},NA,NA,${np},NA,NA,NA,NA,${iter}" >> "$CSV"
			echo "[FAIL] ${config} np=${np} iter=${iter} — stderr saved: $errf"
			tail -n 12 "$errf" 2>/dev/null
		fi
	done
	for kv in "$@"; do unset "${kv%%=*}"; done
}

echo "=============================================================="
echo " E2b: static knee sweep (heat_normal) — time(N) and C(N)=N*time; knee N* < 64"
echo "=============================================================="
for np in "${PROCS_KNEE[@]}"; do
	run_case "knee_static" "$np" heat_normal
done

echo "=============================================================="
echo " E2a: COST policy (heat) launched at 64 — sheds idle ranks to the knee on its own"
echo "=============================================================="
run_case "cost_shrink" 64 heat \
	MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 \
	MAL_RESIZE_POLICY=cost MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=64 \
	MAL_COST_KEEP_FRACTION=0.97 MAL_EPOCH_INTERVAL_MS=300 \
	MAL_COST_SAMPLE_STEP=16 \
	MAL_COST_SAMPLE_MEAS=1

echo "=============================================================="
echo " E2a control: malleable resize OFF (heat_nolb) at the full width — low-overhead reproduction"
echo "=============================================================="
run_case "heat_nolb" 64 heat \
	MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=0 \
	MAL_INITIAL_SIZE=64 MAL_EPOCH_INTERVAL_MS=100

echo "=============================================================="
echo " DONE. Results: $CSV"
echo " E2a: cost_shrink row -> nproc_active is the converged optimum (< 64), at ~knee wall time."
echo " E2b: plot time vs nproc (knee_static): falls, bottoms at N*, then RISES -> optimum < maximum;"
echo " C(N)=nproc*time has its minimum at N*. The COST nproc_active should land near N*."
echo "=============================================================="

