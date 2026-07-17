#!/bin/bash

#SBATCH -J suite_exp
#SBATCH -o suite_exp_%j.out
#SBATCH -e suite_exp_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=07:00:00

set -uo pipefail

NP32=32
NP64=64
MAP32="--map-by ppr:32:node:pe=2 --bind-to core"
MAP32S="--map-by ppr:32:node:pe=1 --bind-to core"
MAP64="--map-by ppr:64:node:pe=1 --bind-to core"

CONT_N=500000
CONT_WORK=40000
CONT_BUF_MB=32
CONT_ITERS=6
CONT_EPOCH=40
CONT_DOSE=(160 640)
NOISY_RANKS=(1 2 3 4 16 17 18 19)

CAP_N=500000
CAP_WORK=40000
CAP_ITERS=8
CAP_EPOCH=40

OVH_PROCS=(8 16 32 64)
OVH_ITERS=5
OVH_SYNTH_N=400000
OVH_AVG_DEG=20
OVH_REPS=10000
OVH_EPOCH=50

NOI_PROCS=(8 16 32 64)
NOI_US=(0 100 300 600)
NOI_FRAC=4
NOI_ITERS=4
NOI_N_PER_RANK=25000
NOI_WORK=300
NOI_EPOCH=40

NM_ITERS=4
NM_N=500000
NM_WORK=40000
NM_EPOCH=40
NM_MODES=(none cpu cache)

MAX_SECONDS=900
NODE_SCRATCH="${LOCAL_SCRATCH:-${TMPDIR:-/tmp}}"
RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
RUN="$RESULTS_DIR/suite_run_${SLURM_JOB_ID}"
ALLCSV="$RESULTS_DIR/suite_results.csv"
ALLTM="$RESULTS_DIR/suite_threads.csv"
ALLTIMING="$RESULTS_DIR/suite_timing.csv"
ALLENERGY="$RESULTS_DIR/suite_energy.csv"
LEGACY="$RESULTS_DIR/contention_data.csv"
STOP="$RUN/stop"

cd "$SLURM_SUBMIT_DIR"
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

if [[ -n "$NODE0" ]]; then

	HOST1="--host ${NODE0}:64"

else

	HOST1=""

fi

MAP32="$HOST1 $MAP32"
MAP32S="$HOST1 $MAP32S"
MAP64="$HOST1 $MAP64"
MAP_2N="--map-by ppr:16:node:pe=2 --bind-to core"

echo "[nodes] allocated=$NUM_NODES single-node sections pinned to ${NODE0:-<local>}; multi-node section uses all nodes"

for tool in taskset numactl lscpu; do

	if ! command -v "$tool" >/dev/null 2>&1; then

		echo "[FATAL] required tool '$tool' not found"
		exit 1

	fi

done

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll_han_priority=0
export OMPI_MCA_coll_adapt_priority=0

export MAL_START_AT_UNIVERSE=1
export MAL_RESIZE_ENABLED=0
export MAL_LOG_LEVEL=ERROR
export MAL_LOG_ALL_RANKS=0
export MAL_TIMING=0
export MAL_EPOCH_CHANGE_MODE=1

PAPI_PFX="${PAPI_PREFIX:-}"

if [[ -z "$PAPI_PFX" ]]; then

	for cand in "$HOME/opt/papi" "$HOME/papi/src/install" "${PAPI_HOME:-}" "${EBROOTPAPI:-}"; do

		if [[ -n "$cand" && -f "$cand/include/papi.h" ]]; then

			PAPI_PFX="$cand"
			break

		fi

	done

fi

PAPI_ON=0

if [[ -n "$PAPI_PFX" && -f "$PAPI_PFX/include/papi.h" ]]; then

	PAPI_ON=1

fi

PFM_PFX="${PFM_PREFIX:-}"

if [[ "$PAPI_ON" == "1" && -z "$PFM_PFX" ]]; then

	if ! nm "$PAPI_PFX/lib/libpapi.a" 2>/dev/null | grep -q " T pfm_initialize"; then

		for cand in "$HOME/papi/src/libpfm4" "$HOME/opt/papi" "${LIBPFM_HOME:-}"; do

			if [[ -n "$cand" && ( -f "$cand/lib/libpfm.a" || -f "$cand/lib/libpfm.so" ) ]]; then

				PFM_PFX="$cand"
				break

			fi

		done

	fi

fi

make clean
make BENCH_CSV=1 USE_PAPI="$PAPI_ON" PAPI_PREFIX="$PAPI_PFX" PFM_PREFIX="$PFM_PFX" -j

if [[ "$PAPI_ON" == "1" && ! -x ./build/noise_bench ]]; then

	echo "[WARN] PAPI build failed, retrying without it"
	PAPI_ON=0
	make clean
	make BENCH_CSV=1 -j

fi

for bin in noise_bench noise_bench_normal noise_daemon spmv_imbal spmv_imbal_normal; do

	if [[ ! -x "./build/$bin" ]]; then

		echo "[FATAL] build did not produce ./build/$bin"
		exit 1

	fi

done

mkdir -p "$RESULTS_DIR" "$RUN"
chmod +x scripts/rank_pin.sh

git rev-parse HEAD > "$RUN/commit.txt" 2>/dev/null || true
lscpu > "$RUN/lscpu.txt" 2>/dev/null || true
env | grep -E '^(MAL_|OMPI_MCA_)' > "$RUN/env.txt" || true

if [[ ! -f "$ALLCSV" ]]; then

	echo "experiment,kernel,sharing,placement,noise_mode,noise_target,noise_us,n_noisy,yield,epoch_ms,config,np,iter,exec,version,mode,nproc,nproc_active,n,time,errors" > "$ALLCSV"

fi

if [[ ! -f "$ALLTM" ]]; then

	echo "experiment,kernel,sharing,placement,noise_mode,noise_target,noise_us,n_noisy,yield,epoch_ms,config,np,iter,rank,noisy,compute_core,worker_core,n_tids,schedstat_ok,iters_done,wall_loop_s,wall_total_s,compute_cpu_s,proc_cpu_s,other_cpu_s,t_kernel_s,t_check_s,compute_runq_s,worker_cpu_s,worker_runq_s,nvcsw,nivcsw,cycles,insns,llc_ref,llc_miss,ref_cycles,papi_ok,papi_cyc,papi_ins,papi_l3tcm,papi_refcyc,papi_ipc,papi_membound,errors" > "$ALLTM"

fi

if [[ ! -f "$ALLTIMING" ]]; then

	echo "experiment,kernel,sharing,placement,noise_mode,noise_target,noise_us,n_noisy,yield,epoch_ms,config,np,iter,rank,size,name,value" > "$ALLTIMING"

fi

if [[ ! -f "$ALLENERGY" ]]; then

	echo "experiment,kernel,sharing,placement,noise_mode,noise_target,noise_us,n_noisy,yield,epoch_ms,config,np,iter,joules" > "$ALLENERGY"

fi

read_energy_uj() {

	local total=0 f v ok=0

	for f in /sys/class/powercap/intel-rapl:*/energy_uj; do

		[[ -r "$f" ]] || continue
		v=$(cat "$f" 2>/dev/null) || continue
		total=$((total + v))
		ok=1

	done

	if [[ "$ok" == "1" ]]; then

		echo "$total"

	fi

}

ENERGY_ON=0

if [[ -n "$(read_energy_uj)" ]]; then

	ENERGY_ON=1

fi

echo "[energy] RAPL powercap readable: $ENERGY_ON (0 => suite_energy.csv stays empty; usual on clusters with restricted perf/powercap)"

if [[ ! -f "$LEGACY" ]]; then

	echo "config,kernel,placement,noise,yield,epoch_ms,iter,rank,noisy,compute_core,worker_core,n_tids,schedstat_ok,iters_done,wall_loop_s,wall_total_s,compute_cpu_s,proc_cpu_s,other_cpu_s,t_kernel_s,t_check_s,compute_runq_s,worker_cpu_s,worker_runq_s,nvcsw,nivcsw,cycles,insns,llc_ref,llc_miss,ref_cycles,papi_ok,papi_cyc,papi_ins,papi_l3tcm,papi_refcyc,papi_ipc,papi_membound,errors" > "$LEGACY"

fi

echo "[setup] papi=$PAPI_ON schedstat=$(cat /proc/sys/kernel/sched_schedstats 2>/dev/null) paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"

declare -A LO32_OF HI32_OF LO32S_OF

MAL_REPORT=1 mpirun -n "$NP32" $MAP32 ./scripts/rank_pin.sh /bin/true > "$RUN/bind32.txt" 2>/dev/null

SMT_ON=0

while read -r tag r lo hi sib; do

	if [[ "$tag" == "RANKMAP" ]]; then

		LO32_OF[$r]="$lo"
		HI32_OF[$r]="$hi"

		if [[ "$sib" == *","* || "$sib" == *"-"* ]]; then

			SMT_ON=1

		fi

	fi

done < "$RUN/bind32.txt"

echo "[smt] Thread siblings on compute cores: SMT_ON=$SMT_ON (0 => SMT arm skipped, documented as platform limitation)"

MAL_REPORT=1 mpirun -n "$NP32" $MAP32S ./scripts/rank_pin.sh /bin/true > "$RUN/bind32s.txt" 2>/dev/null

while read -r tag r lo hi sib; do

	if [[ "$tag" == "RANKMAP" ]]; then

		LO32S_OF[$r]="$lo"

	fi

done < "$RUN/bind32s.txt"

if [[ ${#LO32_OF[@]} -lt $NP32 || ${#LO32S_OF[@]} -lt $NP32 ]]; then

	echo "[FATAL] preflight mapped ${#LO32_OF[@]}/${#LO32S_OF[@]} of $NP32 ranks"
	exit 1

fi

EXP=NA; KERN=NA; SHARING=NA; PLACE=NA; NMODE=none; NTGT=NA; NUS=0; NN=0; YLD=0; EPOCH=40; CFG=NA; NPV=32; ITV=1

run_one() {

	local map="$1" bin="$2" args="$3" env_extra="$4"
	local out line errf meta

	meta="${EXP},${KERN},${SHARING},${PLACE},${NMODE},${NTGT},${NUS},${NN},${YLD},${EPOCH},${CFG},${NPV},${ITV}"
	errf="$RUN/e_${EXP}_${KERN}_${SHARING}_${PLACE}_${NMODE}_${NTGT}_${NUS}_${YLD}_${EPOCH}_${CFG}_np${NPV}_it${ITV}.err"

	local e0="" e1=""

	if [[ "$ENERGY_ON" == "1" ]]; then

		e0=$(read_energy_uj)

	fi

	out=$(env $env_extra PLACEMENT="$PLACE" OMPI_MCA_mpi_yield_when_idle="$YLD" OMPI_MCA_opal_progress_yield_when_idle="$YLD" mpirun -n "$NPV" $map ./scripts/rank_pin.sh ./build/"$bin" $args 2>"$errf")

	if [[ "$ENERGY_ON" == "1" ]]; then

		e1=$(read_energy_uj)

		if [[ -n "$e0" && -n "$e1" && "$e1" -gt "$e0" ]]; then

			echo "${meta},$(awk -v a="$e0" -v b="$e1" 'BEGIN{printf "%.3f", (b - a) / 1000000}')" >> "$ALLENERGY"

		fi

	fi
	line=$(printf '%s\n' "$out" | awk -F, '/^CSV,/{l=$0} END{if(l!="")print substr(l,5)}')

	if [[ -n "$line" ]]; then

		echo "${meta},${line}" >> "$ALLCSV"
		echo "[OK] $meta : $line"
		rm -f "$errf"

	else

		echo "${meta},${bin},NA,NA,${NPV},NA,NA,NA,NA" >> "$ALLCSV"
		echo "[FAIL] $meta — stderr: $errf"
		tail -n 8 "$errf" 2>/dev/null

	fi

	printf '%s\n' "$out" | awk -F, -v m="$meta" '/^TM,/{ $1=""; sub(/^,/,""); print m","$0 }' OFS=, >> "$ALLTM"
	printf '%s\n' "$out" | awk -F, -v m="$meta" '/^TIMING,/{ $1=""; sub(/^,/,""); print m","$0 }' OFS=, >> "$ALLTIMING"

	if [[ "$EXP" == "contention" ]]; then

		printf '%s\n' "$out" | sed -n "s/^TM,/${CFG},${KERN},${PLACE},${NMODE},${YLD},${EPOCH},${ITV},/p" >> "$LEGACY"

	fi

}

DAEMON_PIDS=()
DAEMONS_OK=1

cleanup() {

	touch "$STOP" 2>/dev/null
	for pid in "${DAEMON_PIDS[@]:-}"; do

		kill -TERM "$pid" 2>/dev/null

	done
	wait 2>/dev/null

}

trap cleanup EXIT

socket_of() {

	cat "/sys/devices/system/cpu/cpu$1/topology/physical_package_id" 2>/dev/null || echo 0

}

start_daemons() {

	local mode="$1"; shift
	local cores=("$@") core sock rf
	DAEMON_PIDS=()
	rm -f "$RUN"/rdy.* "$STOP"

	for core in "${cores[@]}"; do

		sock="$(socket_of "$core")"
		rf="$RUN/rdy.$core"

		if [[ "$mode" == "cpu" ]]; then

			taskset -c "$core" ./build/noise_daemon --mode=cpu --ready-file="$rf" --stop-file="$STOP" --max-seconds="$MAX_SECONDS" >/dev/null 2>>"$RUN/daemon.err" &

		else

			numactl -C "$core" -m "$sock" ./build/noise_daemon --mode="$mode" --scratch="$NODE_SCRATCH/noise_io_${SLURM_JOB_ID}.$core" --ready-file="$rf" --stop-file="$STOP" --max-seconds="$MAX_SECONDS" >/dev/null 2>>"$RUN/daemon.err" &

		fi

		DAEMON_PIDS+=("$!")

	done

	local waited=0 ready

	while true; do

		ready=1

		for core in "${cores[@]}"; do

			if [[ ! -f "$RUN/rdy.$core" ]]; then

				ready=0

			fi

		done

		if [[ "$ready" == "1" ]]; then

			break

		fi

		sleep 0.2
		waited=$((waited + 1))

		if [[ $waited -gt 250 ]]; then

			break

		fi

	done

	DAEMONS_OK=1

	for core in "${cores[@]}"; do

		if [[ ! -f "$RUN/rdy.$core" ]]; then

			DAEMONS_OK=0

		fi

	done

	for pid in "${DAEMON_PIDS[@]:-}"; do

		if ! kill -0 "$pid" 2>/dev/null; then

			DAEMONS_OK=0

		fi

	done

	if [[ "$DAEMONS_OK" != "1" ]]; then

		echo "[WARN] stressor degraded; see $RUN/daemon.err"

	fi

}

stop_daemons() {

	touch "$STOP"
	for pid in "${DAEMON_PIDS[@]:-}"; do

		kill -TERM "$pid" 2>/dev/null

	done
	wait "${DAEMON_PIDS[@]:-}" 2>/dev/null
	DAEMON_PIDS=()
	rm -f "$RUN"/rdy.* "$STOP" "$NODE_SCRATCH"/noise_io_"${SLURM_JOB_ID}".* 2>/dev/null

}

noisy_cores() {

	local which="$1" r out=()

	for r in "${NOISY_RANKS[@]}"; do

		case "$which" in
			lo32) out+=("${LO32_OF[$r]}") ;;
			hi32) out+=("${HI32_OF[$r]}") ;;
			lo32s) out+=("${LO32S_OF[$r]}") ;;
		esac

	done

	echo "${out[@]}"

}

kern_args() {

	case "$1" in
		mem) echo "--kernel=mem --buf_mb=$CONT_BUF_MB" ;;
		imbal) echo "--kernel=imbal" ;;
		*) echo "--kernel=fp" ;;
	esac

}

cfg_env() {

	local config="$1" np="$2" epoch="$3"

	case "$config" in
		normal) echo "MAL_PIN_MAIN_ONLY=1" ;;
		malleable_nolb) echo "MAL_MALLEABILITY_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=0 MAL_INITIAL_SIZE=$np MAL_EPOCH_INTERVAL_MS=$epoch" ;;
		*) echo "MAL_MALLEABILITY_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 MAL_INITIAL_SIZE=$np MAL_EPOCH_INTERVAL_MS=$epoch" ;;
	esac

}

cfg_bin() {

	if [[ "$1" == "normal" ]]; then

		echo noise_bench_normal

	else

		echo noise_bench

	fi

}

echo "================ SECTION A: contention (same-core vs separate-core) ================"

export MAL_AFFINITY=1
EXP=contention; SHARING=dedicated; NTGT=compute; NUS=0; NN=${#NOISY_RANKS[@]}; NPV=$NP32

for iter in $(seq 1 "$CONT_ITERS"); do

	ITV=$iter

	for noise in none cpu; do

		NMODE=$noise

		if [[ "$noise" != "none" ]]; then

			start_daemons cpu $(noisy_cores lo32)

		fi

		for kern in fp mem imbal; do

			KERN=$kern

			for placement in separated colocated; do

				PLACE=$placement

				for config in normal malleable_nolb malleable_lb; do

					CFG=$config; YLD=0; EPOCH=$CONT_EPOCH
					run_one "$MAP32" "$(cfg_bin "$config")" "--n=$CONT_N --work=$CONT_WORK $(kern_args "$kern") --noise_us=0" "$(cfg_env "$config" "$NPV" "$EPOCH")"

				done

			done

		done

		if [[ "$noise" != "none" ]]; then

			stop_daemons

		fi

	done

	NMODE=none

	for kern in fp mem imbal; do

		KERN=$kern

		for placement in separated colocated; do

			PLACE=$placement; CFG=malleable_lb; YLD=1; EPOCH=$CONT_EPOCH
			run_one "$MAP32" noise_bench "--n=$CONT_N --work=$CONT_WORK $(kern_args "$kern") --noise_us=0" "$(cfg_env malleable_lb "$NPV" "$EPOCH")"

		done

	done

	KERN=fp; YLD=0

	for epoch in "${CONT_DOSE[@]}"; do

		EPOCH=$epoch

		for placement in separated colocated; do

			PLACE=$placement; CFG=malleable_lb
			run_one "$MAP32" noise_bench "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_lb "$NPV" "$epoch")"

		done

	done

done

echo "================ SECTION B: capacity equivalence (worker's extra core at fixed node size) ================"

export MAL_AFFINITY=1
EXP=capacity; KERN=fp; SHARING=NA; NMODE=none; NTGT=NA; NUS=0; NN=0; YLD=0; EPOCH=$CAP_EPOCH

for iter in $(seq 1 "$CAP_ITERS"); do

	ITV=$iter

	NPV=$NP64; PLACE=separated; CFG=normal
	run_one "$MAP64" noise_bench_normal "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "MAL_PIN_MAIN_ONLY=1"

	NPV=$NP64; PLACE=colocated; CFG=malleable_nolb
	run_one "$MAP64" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_nolb "$NP64" "$CAP_EPOCH")"

	NPV=$NP64; PLACE=colocated; CFG=malleable_lb
	run_one "$MAP64" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_lb "$NP64" "$CAP_EPOCH")"

	NPV=$NP32; PLACE=separated; CFG=normal
	run_one "$MAP32" noise_bench_normal "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "MAL_PIN_MAIN_ONLY=1"

	NPV=$NP32; PLACE=separated; CFG=malleable_nolb
	run_one "$MAP32" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_nolb "$NP32" "$CAP_EPOCH")"

	NPV=$NP32; PLACE=separated; CFG=malleable_lb
	run_one "$MAP32" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_lb "$NP32" "$CAP_EPOCH")"

done

echo "================ SECTION C: worker overhead on a packed node ================"

export MAL_AFFINITY=0
EXP=overhead; KERN=spmv; SHARING=shared; PLACE=NA; NMODE=none; NTGT=NA; NUS=0; NN=0; YLD=0; EPOCH=$OVH_EPOCH

for iter in $(seq 1 "$OVH_ITERS"); do

	ITV=$iter

	for np in "${OVH_PROCS[@]}"; do

		NPV=$np

		CFG=normal
		run_one "--bind-to core" spmv_imbal_normal "--synthetic=$OVH_SYNTH_N --avg_deg=$OVH_AVG_DEG --pattern=uniform --reps=$OVH_REPS" ""

		CFG=malleable_nolb
		run_one "--bind-to core" spmv_imbal "--synthetic=$OVH_SYNTH_N --avg_deg=$OVH_AVG_DEG --pattern=uniform --reps=$OVH_REPS" "$(cfg_env malleable_nolb "$np" "$OVH_EPOCH")"

		CFG=malleable_lb
		run_one "--bind-to core" spmv_imbal "--synthetic=$OVH_SYNTH_N --avg_deg=$OVH_AVG_DEG --pattern=uniform --reps=$OVH_REPS" "$(cfg_env malleable_lb "$np" "$OVH_EPOCH")"

	done

done

echo "================ SECTION D: LB vs internal noise (busy ranks) ================"

export MAL_AFFINITY=0
EXP=noise; KERN=fp; SHARING=none; PLACE=NA; NMODE=internal; NTGT=compute; YLD=0; EPOCH=$NOI_EPOCH

noisy_list() {

	local np="$1" nn="$2" list="" k

	for ((k=0; k<nn && k<np; k++)); do

		if [[ -z "$list" ]]; then list="$k"; else list="$list,$k"; fi

	done

	echo "$list"

}

for iter in $(seq 1 "$NOI_ITERS"); do

	ITV=$iter

	for nu in "${NOI_US[@]}"; do

		NUS=$nu

		for np in "${NOI_PROCS[@]}"; do

			NPV=$np
			nn=$(( np / NOI_FRAC )); [[ $nn -lt 1 ]] && nn=1
			NN=$nn
			ranks=$(noisy_list "$np" "$nn")
			noi_n=$(( NOI_N_PER_RANK * np ))

			for config in malleable_nolb malleable_lb; do

				CFG=$config
				run_one "--bind-to none" noise_bench "--n=$noi_n --work=$NOI_WORK --noise_us=$nu --noise_ranks=$ranks" "$(cfg_env "$config" "$np" "$NOI_EPOCH")"

			done

		done

	done

done

echo "================ SECTION E: external stressor matrix (cpu/cache x sharing x target) ================"

export MAL_AFFINITY=1
EXP=noise_matrix; KERN=fp; NUS=0; NN=${#NOISY_RANKS[@]}; YLD=0; EPOCH=$NM_EPOCH; NPV=$NP32

for iter in $(seq 1 "$NM_ITERS"); do

	ITV=$iter

	for sharing in shared dedicated; do

		SHARING=$sharing

		if [[ "$sharing" == "shared" ]]; then

			nm_map="$MAP32S"
			nm_cores="$(noisy_cores lo32s)"
			PLACE=colocated

		else

			nm_map="$MAP32"
			nm_cores="$(noisy_cores lo32)"
			PLACE=separated

		fi

		for noise in "${NM_MODES[@]}"; do

			NMODE=$noise; NTGT=compute

			if [[ "$noise" != "none" ]]; then

				start_daemons "$noise" $nm_cores

			fi

			for config in normal malleable_nolb malleable_lb; do

				CFG=$config
				run_one "$nm_map" "$(cfg_bin "$config")" "--n=$NM_N --work=$NM_WORK --kernel=fp --noise_us=0" "$(cfg_env "$config" "$NP32" "$NM_EPOCH")"

			done

			if [[ "$noise" != "none" ]]; then

				stop_daemons

			fi

		done

	done

	SHARING=dedicated; PLACE=separated

	for noise in cpu cache; do

		NMODE=$noise; NTGT=worker
		start_daemons "$noise" $(noisy_cores hi32)

		for config in malleable_nolb malleable_lb; do

			CFG=$config
			run_one "$MAP32" noise_bench "--n=$NM_N --work=$NM_WORK --kernel=fp --noise_us=0" "$(cfg_env "$config" "$NP32" "$NM_EPOCH")"

		done

		stop_daemons

	done

done

echo "================ SECTION F: capacity crossover under noise (when paying the core WINS) ================"

export MAL_AFFINITY=1
EXP=capacity_noise; KERN=fp; SHARING=NA; NMODE=cpu; NTGT=compute; NUS=0; NN=${#NOISY_RANKS[@]}; YLD=0; EPOCH=$CAP_EPOCH

CAPN_ITERS=6

for iter in $(seq 1 "$CAPN_ITERS"); do

	ITV=$iter

	start_daemons cpu $(noisy_cores lo32)

	NPV=$NP64; PLACE=separated; CFG=normal
	run_one "$MAP64" noise_bench_normal "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "MAL_PIN_MAIN_ONLY=1"

	NPV=$NP32; PLACE=separated; CFG=normal
	run_one "$MAP32" noise_bench_normal "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "MAL_PIN_MAIN_ONLY=1"

	NPV=$NP32; PLACE=separated; CFG=malleable_nolb
	run_one "$MAP32" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_nolb "$NP32" "$CAP_EPOCH")"

	NPV=$NP32; PLACE=separated; CFG=malleable_lb
	run_one "$MAP32" noise_bench "--n=$CAP_N --work=$CAP_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_lb "$NP32" "$CAP_EPOCH")"

	stop_daemons

done

echo "================ SECTION G: epoch-collective latency (mechanism evidence, MAL_TIMING) ================"

export MAL_AFFINITY=1
EXP=epoch_timing; KERN=fp; SHARING=dedicated; NMODE=none; NTGT=compute; NUS=0; NN=0; YLD=0; EPOCH=$CONT_EPOCH; NPV=$NP32

TIM_ITERS=4

for iter in $(seq 1 "$TIM_ITERS"); do

	ITV=$iter

	for placement in separated colocated; do

		PLACE=$placement; CFG=malleable_lb
		run_one "$MAP32" noise_bench "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_lb "$NP32" "$CONT_EPOCH") MAL_TIMING=1"

		PLACE=$placement; CFG=malleable_nolb
		run_one "$MAP32" noise_bench "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env malleable_nolb "$NP32" "$CONT_EPOCH") MAL_TIMING=1"

	done

done

echo "================ SECTION H: SMT arm (worker on the hyperthread sibling) ================"

if [[ "$SMT_ON" == "1" ]]; then

	export MAL_AFFINITY=1
	EXP=contention; KERN=fp; SHARING=dedicated; PLACE=smt; NMODE=none; NTGT=compute; NUS=0; NN=0; YLD=0; EPOCH=$CONT_EPOCH; NPV=$NP32

	for iter in $(seq 1 4); do

		ITV=$iter

		for config in malleable_nolb malleable_lb; do

			CFG=$config
			run_one "$MAP32" noise_bench "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env "$config" "$NP32" "$CONT_EPOCH")"

		done

	done

else

	echo "[SMT] Thread(s) per core = 1 on this platform — SMT arm skipped (document as: middle option between colocated 2x and dedicated +0.6% not measurable here)"

fi

echo "================ SECTION I: multi-node generality (2 nodes, epoch collectives over the network) ================"

if [[ "$NUM_NODES" -ge 2 ]]; then

	export MAL_AFFINITY=1
	KERN=fp; SHARING=2nodes; NMODE=none; NTGT=compute; NUS=0; NN=0; YLD=0; EPOCH=$CONT_EPOCH; NPV=$NP32

	EXP=contention_2n

	for iter in $(seq 1 6); do

		ITV=$iter

		for placement in separated colocated; do

			PLACE=$placement

			for config in normal malleable_nolb malleable_lb; do

				CFG=$config
				run_one "$MAP_2N" "$(cfg_bin "$config")" "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env "$config" "$NP32" "$CONT_EPOCH")"

			done

		done

	done

	EXP=epoch_timing_2n

	for iter in $(seq 1 4); do

		ITV=$iter

		for placement in separated colocated; do

			PLACE=$placement

			for config in malleable_lb malleable_nolb; do

				CFG=$config
				run_one "$MAP_2N" noise_bench "--n=$CONT_N --work=$CONT_WORK --kernel=fp --noise_us=0" "$(cfg_env "$config" "$NP32" "$CONT_EPOCH") MAL_TIMING=1"

			done

		done

	done

else

	echo "[2N] single-node allocation — multi-node section skipped"

fi

echo "=============================================================="
echo " DONE."
echo " Per-run results:    $ALLCSV     (filter by 'experiment' column)"
echo " Per-rank/thread TM: $ALLTM      (filter by 'experiment' column)"
echo " Epoch timing:       $ALLTIMING  (experiment=epoch_timing; per-epoch latency = epoch_decision / epoch_decision_count)"
echo " Legacy contention:  $LEGACY     (python data/contention_analysis.py)"
echo " Crossover headline: time[capacity_noise normal np64] / time[capacity_noise lb np32] > 1 => paying the core WINS under noise"
echo "=============================================================="
