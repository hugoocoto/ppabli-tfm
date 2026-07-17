#!/bin/bash

set -u

parse_cpuset() {

	local raw="$1" part a b c
	local -a out=()

	IFS=',' read -ra fields <<< "$raw"

	for part in "${fields[@]}"; do

		if [[ "$part" == *-* ]]; then

			a="${part%-*}"
			b="${part#*-}"

			for ((c = a; c <= b; c++)); do

				out+=("$c")

			done

		else

			out+=("$part")

		fi

	done

	printf '%s\n' "${out[@]}" | sort -n

}

RAW="$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status 2>/dev/null)"

if [[ -z "$RAW" ]]; then

	exec "$@"

fi

CORES=()

while IFS= read -r line; do

	CORES+=("$line")

done < <(parse_cpuset "$RAW")

LO="${CORES[0]}"
HI="${CORES[${#CORES[@]} - 1]}"
R="${OMPI_COMM_WORLD_RANK:-0}"

if [[ "${MAL_REPORT:-0}" == "1" ]]; then

	SIB="$(cat "/sys/devices/system/cpu/cpu$LO/topology/thread_siblings_list" 2>/dev/null || echo NA)"
	echo "RANKMAP $R $LO $HI $SIB"
	exit 0

fi

if [[ "${MAL_PIN_MAIN_ONLY:-0}" == "1" ]]; then

	exec taskset -c "$LO" "$@"

fi

export MAL_MAIN_CORE="$LO"

if [[ "${PLACEMENT:-separated}" == "colocated" ]]; then

	export MAL_WORKER_CORE="$LO"

elif [[ "${PLACEMENT:-separated}" == "smt" ]]; then

	SIB_RAW="$(cat "/sys/devices/system/cpu/cpu$LO/topology/thread_siblings_list" 2>/dev/null || echo "$LO")"
	SIB="$LO"

	while IFS= read -r c; do

		if [[ "$c" != "$LO" ]]; then

			SIB="$c"
			break

		fi

	done < <(parse_cpuset "$SIB_RAW")

	export MAL_WORKER_CORE="$SIB"

else

	export MAL_WORKER_CORE="$HI"

fi

exec "$@"
