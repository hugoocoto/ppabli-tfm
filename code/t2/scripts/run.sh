#!/bin/bash

set -euo pipefail

EXECS=()
EXEC_ARGS=()

for arg in "$@"; do

	if [[ "$arg" == "--"* ]]; then

		EXEC_ARGS+=("$arg")

	else

		EXECS+=("$arg")

	fi

done

if [[ ${#EXECS[@]} -eq 0 ]]; then

	echo "Usage: $0 <exec1> [exec2] [--arg=val ...]" >&2
	exit 1

fi

if [[ ${#EXECS[@]} -gt 2 ]]; then

	echo "[ERROR] At most 2 executables supported" >&2
	exit 1

fi

PROCS=(1 2 4 8 16 32 64)
ITERS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${SCRIPT_DIR}/out"

CSVS=()

for EXEC in "${EXECS[@]}"; do

	CSV="${SCRIPT_DIR}/${EXEC}_data.csv"
	CSVS+=("$CSV")

	if [[ -f "$CSV" ]]; then

		rm -f "$CSV"
		rm -f "${CSV}.lock"

	fi

	if [[ ! -f "$CSV" ]]; then

		echo "exec,version,mode,nproc,nproc_active,n,time,errors,iter,exit_code" > "$CSV"

	fi

done

module load cesga/2025 gcc/14.3.0 openmpi/5.0.9
#make clean
#make BENCH_CSV=1

for EXEC in "${EXECS[@]}"; do

	chmod +x "${SCRIPT_DIR}/build/${EXEC}"

done

JOB_LABEL=$(IFS='+'; echo "${EXECS[*]}")

for nproc in "${PROCS[@]}"; do

	SBATCH_ARGS=(
		-n "$nproc"
		-J "TFM.${JOB_LABEL}.${nproc}"
		-o "${SCRIPT_DIR}/out/${JOB_LABEL}.${nproc}.o"
		-e "${SCRIPT_DIR}/out/${JOB_LABEL}.${nproc}.e"
	)

	JOB_ARGS=("$nproc" "$ITERS" "${#EXECS[@]}")

	for idx in "${!EXECS[@]}"; do

		JOB_ARGS+=("${SCRIPT_DIR}/build/${EXECS[$idx]}" "${CSVS[$idx]}")

	done

	if [[ ${#EXEC_ARGS[@]} -gt 0 ]]; then

		JOB_ARGS+=("--" "${EXEC_ARGS[@]}")

	fi

	JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/job.sh" "${JOB_ARGS[@]}")
	echo "[SUBMIT] job_id=${JOB_ID} execs=${JOB_LABEL} nproc=${nproc} iters=${ITERS}"

done
