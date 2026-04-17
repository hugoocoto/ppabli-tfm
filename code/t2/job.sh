#!/bin/bash

#SBATCH --mem 50G
#SBATCH -c 5
#SBATCH -t 01:00:00

set -euo pipefail

NPROC=$1
ITERS=$2
N_EXECS=$3
shift 3

EXEC_PATHS=()
CSV_LIST=()

for ((e = 0; e < N_EXECS; e++)); do

	EXEC_PATHS+=("$1")
	CSV_LIST+=("$2")
	shift 2

done

if [[ "${1:-}" == "--" ]]; then

	shift

fi

EXEC_ARGS=("$@")

ITER_TIMEOUT_SEC=${MAL_ITER_TIMEOUT_SEC:-0}
CONTINUE_ON_ERROR=${MAL_CONTINUE_ON_ERROR:-0}

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

export MAL_RESIZE_ENABLED=1
export MAL_MALLEABILITY_ENABLED=1
export MAL_INITIAL_SIZE=1
export MAL_AFFINITY=0
export MAL_EPOCH_INTERVAL_MS=500
export MAL_LOG_LEVEL=DEBUG

run_single() {

	local exec_path="$1"
	local csv="$2"
	local iter="$3"
	local lock="${csv}.lock"
	local exec_name
	exec_name=$(basename "$exec_path")

	local out_file
	out_file=$(mktemp "${TMPDIR:-/tmp}/malleable_job.XXXXXX")

	local exit_code=0
	local start_ns end_ns elapsed

	start_ns=$(date +%s%N)

	if [[ "$ITER_TIMEOUT_SEC" -gt 0 ]]; then

		if timeout --foreground "${ITER_TIMEOUT_SEC}s" mpirun "$exec_path" "${EXEC_ARGS[@]}" 2>&1 | tee "$out_file"; then

			exit_code=0

		else

			exit_code=$?

		fi

	else

		if mpirun "$exec_path" "${EXEC_ARGS[@]}" 2>&1 | tee "$out_file"; then

			exit_code=0

		else

			exit_code=$?

		fi

	fi

	end_ns=$(date +%s%N)
	elapsed=$(echo "scale=6; ($end_ns - $start_ns) / 1000000000" | bc)

	local bench_line result_line
	bench_line=$(awk '/^CSV,/{line=$0} END{if (line != "") print line}' "$out_file")

	if [[ -n "$bench_line" ]]; then

		result_line="${bench_line#CSV,},${iter},${exit_code}"

	else

		result_line="${exec_name},${NPROC},${iter},${elapsed},${exit_code}"

	fi

	(
		flock 200
		printf '%s\n' "$result_line" >> "$csv"
	) 200>"$lock"

	rm -f "$out_file"

	if [[ "$exit_code" -ne 0 ]]; then

		echo "[JOB] mpirun failed exec=${exec_name} nproc=${NPROC} iter=${iter} exit_code=${exit_code}" >&2
		return "$exit_code"

	fi

	return 0

}

for i in $(seq 1 "$ITERS"); do

	overall_exit=0

	for idx in "${!EXEC_PATHS[@]}"; do

		if ! run_single "${EXEC_PATHS[$idx]}" "${CSV_LIST[$idx]}" "$i"; then

			overall_exit=$?

			if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then

				exit "$overall_exit"

			fi

		fi

	done

done
