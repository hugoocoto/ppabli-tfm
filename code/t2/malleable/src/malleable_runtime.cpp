#include "malleable_resizer.cpp"

static void log_mpi_error(const char* where, int rc) {

	if (rc == MPI_SUCCESS) {

		return;

	}

	char err[MPI_MAX_ERROR_STRING] = {};
	int len = 0;
	MPI_Error_string(rc, err, &len);
	MAL_LOG_L(MAL_LOG_ERROR, "MPI", "%s failed rc=%d msg=%.*s", where, rc, len, err);

}

static bool parse_env_bool(const char* text, bool& out) {

	if (!text || !*text) {

		return false;

	}

	char* end = nullptr;
	long n = std::strtol(text, &end, 10);

	if (end != text && *end == '\0') {

		out = (n != 0);
		return true;

	}

	if (std::strcmp(text, "true") == 0 || std::strcmp(text, "TRUE") == 0 || std::strcmp(text, "on") == 0 || std::strcmp(text, "ON") == 0 || std::strcmp(text, "yes") == 0 || std::strcmp(text, "YES") == 0) {

		out = true;
		return true;

	}

	if (std::strcmp(text, "false") == 0 || std::strcmp(text, "FALSE") == 0 || std::strcmp(text, "off") == 0 || std::strcmp(text, "OFF") == 0 || std::strcmp(text, "no") == 0 || std::strcmp(text, "NO") == 0) {

		out = false;
		return true;

	}

	return false;

}

static bool parse_env_log_level(const char* text, MalLogLevel& out) {

	if (!text || !*text) {

		return false;

	}

	char* end = nullptr;
	long n = std::strtol(text, &end, 10);

	if (end != text && *end == '\0') {

		if (n < MAL_LOG_DEBUG || n > MAL_LOG_ERROR) {

			return false;

		}

		out = (MalLogLevel)n;
		return true;

	}

	if (std::strcmp(text, "DEBUG") == 0 || std::strcmp(text, "debug") == 0) {

		out = MAL_LOG_DEBUG;
		return true;

	}

	if (std::strcmp(text, "INFO") == 0 || std::strcmp(text, "info") == 0) {

		out = MAL_LOG_INFO;
		return true;

	}

	if (std::strcmp(text, "WARN") == 0 || std::strcmp(text, "warn") == 0 || std::strcmp(text, "WARNING") == 0 || std::strcmp(text, "warning") == 0) {

		out = MAL_LOG_WARN;
		return true;

	}

	if (std::strcmp(text, "ERROR") == 0 || std::strcmp(text, "error") == 0) {

		out = MAL_LOG_ERROR;
		return true;

	}

	return false;

}

void mal_set_epoch_interval_ms(int ms) {

	if (ms > 0) {

		g.cfg.epoch_ms.store(ms);

	} else {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid epoch interval ms=%d (must be > 0)", ms);

	}

	g.sync.notify();

}

void mal_set_resize_enabled(bool b) {

	g.cfg.enabled.store(b);

	if (!b) {

		clear_prepared_resize();

	}

	g.sync.notify();

}

void mal_set_attach_exec_mode(MalAttachExecMode mode) {

	if (mode != MAL_ATTACH_SYNC && mode != MAL_ATTACH_ASYNC) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid attach execution mode=%d", (int)mode);

		return;

	}

	g.cfg.attach_mode.store(mode);
	g.sync.notify();

}

MalAttachExecMode mal_get_attach_exec_mode() {

	return g.cfg.attach_mode.load();

}

bool apply_resize_sequence(const std::vector<int>& seq, const char* source) {

	g.cfg.sequence.clear();
	g.cfg.sequence.reserve(seq.size());

	for (size_t i = 0; i < seq.size(); i++) {

		const int target = seq[i];

		if (target <= 0) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s: invalid resize target seq[%zu]=%d (must be > 0)", source ? source : "CONFIG", i, target);
			return false;

		}

		g.cfg.sequence.push_back(target);

	}

	if (g.cfg.sequence.empty()) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s: resize sequence is empty", source ? source : "CONFIG");
		return false;

	}

	g.cfg.seq_idx.store(0, std::memory_order_relaxed);
	return true;

}

bool parse_resize_sequence(const char* text, std::vector<int>& seq_out, bool& found_invalid) {

	seq_out.clear();
	found_invalid = false;

	if (!text || !*text) {

		return false;

	}

	const char* p = text;

	while (*p) {

		char* end = nullptr;
		long n = std::strtol(p, &end, 10);

		if (end == p) {

			found_invalid = true;
			break;

		}

		if (n > 0) {

			seq_out.push_back((int)n);

		} else {

			found_invalid = true;

		}

		p = end;

		while (*p == ',' || *p == ' ') {

			p++;

		}

	}

	return !seq_out.empty();

}

void load_resize_sequence_or_abort() {

	const char* source_name = "MAL_RESIZE_SEQ";
	const char* source_value = std::getenv("MAL_RESIZE_SEQ");

	if (!source_value || !*source_value) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Missing resize sequence: set MAL_RESIZE_SEQ");
		std::abort();

	}

	std::vector<int> seq;
	bool found_invalid = false;

	if (!parse_resize_sequence(source_value, seq, found_invalid) || found_invalid) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s is invalid; expected comma-separated positive integers", source_name);
		std::abort();

	}

	if (!apply_resize_sequence(seq, source_name)) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Failed to apply resize sequence from %s", source_name);
		std::abort();

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "%s loaded (%zu resize points)", source_name, seq.size());

}

void validate_resize_sequence_against_universe_or_abort() {

	for (size_t i = 0; i < g.cfg.sequence.size(); i++) {

		const int target = g.cfg.sequence[i];

		if (target > g.comm.u_size) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Resize target seq[%zu]=%d exceeds universe size=%d", i, target, g.comm.u_size);
			std::abort();

		}

	}

}

void load_env_config() {

	if (const char* v = std::getenv("MAL_LOG_LEVEL")) {

		MalLogLevel level;

		if (parse_env_log_level(v, level)) {

			g.cfg.log_level.store(level, std::memory_order_relaxed);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_LOG_LEVEL=%s", mal_log_level_name(level));

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_LOG_LEVEL='%s' (valid: DEBUG/INFO/WARN/ERROR or 0..3)", v);

		}

	}

	if (const char* v = std::getenv("MAL_LOG_ALL_RANKS")) {

		bool all_ranks = false;

		if (parse_env_bool(v, all_ranks)) {

			g.cfg.log_all_ranks.store(all_ranks, std::memory_order_relaxed);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_LOG_ALL_RANKS=%d", (int)all_ranks);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_LOG_ALL_RANKS='%s' (valid bool)", v);

		}

	}

	if (const char* v = std::getenv("MAL_RESIZE_POLICY")) {

		MalResizePolicy policy = g.cfg.resize_policy;
		bool ok = true;

		if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {

			policy = MAL_RESIZE_POLICY_AUTO;

		} else if (std::strcmp(v, "throughput") == 0 || std::strcmp(v, "THROUGHPUT") == 0) {

			policy = MAL_RESIZE_POLICY_THROUGHPUT;

		} else if (std::strcmp(v, "energy") == 0 || std::strcmp(v, "ENERGY") == 0) {

			policy = MAL_RESIZE_POLICY_ENERGY;

		} else if (std::strcmp(v, "fixed") == 0 || std::strcmp(v, "FIXED") == 0 || std::strcmp(v, "fixed_sequence") == 0 || std::strcmp(v, "FIXED_SEQUENCE") == 0) {

			policy = MAL_RESIZE_POLICY_FIXED_SEQUENCE;

		} else if (std::strcmp(v, "cost") == 0 || std::strcmp(v, "COST") == 0) {

			policy = MAL_RESIZE_POLICY_COST;

		} else {

			ok = false;
			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_RESIZE_POLICY='%s' (valid: auto/throughput/energy/fixed/cost)", v);

		}

		if (ok) {

			g.cfg.resize_policy = policy;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_RESIZE_POLICY=%s", v);

		}

	}

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		load_resize_sequence_or_abort();

	}

	if (const char* v = std::getenv("MAL_EPOCH_INTERVAL_MS")) {

		long ms = std::strtol(v, nullptr, 10);

		if (ms > 0) {

			g.cfg.epoch_ms.store((int)ms);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_EPOCH_INTERVAL_MS=%ld", ms);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_EPOCH_INTERVAL_MS='%s' (must be > 0)", v);

		}

	}

	if (const char* v = std::getenv("MAL_EPOCH_CHANGE_MODE")) {

		char* end = nullptr;
		long mode = std::strtol(v, &end, 10);

		if (end == v || (mode != MAL_EPOCH_CHANGE_RECALCULATE && mode != MAL_EPOCH_CHANGE_USE_LAST_DECISION)) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_EPOCH_CHANGE_MODE='%s' (valid: 0=recalculate, 1=use last decision)", v);

		} else {

			g.cfg.epoch_change_mode.store((int)mode);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_EPOCH_CHANGE_MODE=%ld", mode);

		}

	}

	if (const char* v = std::getenv("MAL_RESIZE_ENABLED")) {

		bool val = false;

		if (parse_env_bool(v, val)) {

			g.cfg.enabled.store(val, std::memory_order_relaxed);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_RESIZE_ENABLED=%d", (int)val);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_RESIZE_ENABLED='%s' (valid bool)", v);

		}

	}

	if (const char* v = std::getenv("MAL_MALLEABILITY_ENABLED")) {

		bool val = false;

		if (parse_env_bool(v, val)) {

			g.cfg.malleability_enabled.store(val, std::memory_order_relaxed);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_MALLEABILITY_ENABLED=%d", (int)val);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_MALLEABILITY_ENABLED='%s' (valid bool)", v);

		}

	}

	if (const char* v = std::getenv("MAL_LOAD_BALANCING_ENABLED")) {

		bool val = false;

		if (parse_env_bool(v, val)) {

			g.cfg.load_balancing_enabled.store(val, std::memory_order_relaxed);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_LOAD_BALANCING_ENABLED=%d", (int)val);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_LOAD_BALANCING_ENABLED='%s' (valid bool)", v);

		}

	}

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		g.cfg.enabled.store(false, std::memory_order_relaxed);
		MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "Malleability disabled: forcing MAL_RESIZE_ENABLED=0");

	}

	if (const char* v = std::getenv("MAL_AFFINITY")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AFFINITY='%s' (invalid), using default (%d)", v, (int)kDefaultAffinityEnabled);

		} else {

			g.cfg.affinity_enabled = (val != 0);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AFFINITY=%ld to affinity %s", val, g.cfg.affinity_enabled ? "enabled" : "disabled");

		}

	}

	if (const char* v = std::getenv("MAL_MAIN_CORE")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v || val < 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_MAIN_CORE='%s' (must be >= 0), using default (%d)", v, kDefaultMainCore);

		} else {

			g.cfg.main_core = (int)val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_MAIN_CORE=%ld", val);

		}

	}

	if (const char* v = std::getenv("MAL_WORKER_CORE")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v || val < 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_WORKER_CORE='%s' (must be >= 0), using default (%d)", v, kDefaultWorkerCore);

		} else {

			g.cfg.worker_core = (int)val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_WORKER_CORE=%ld", val);

		}

	}

	if (const char* v = std::getenv("MAL_INITIAL_SIZE")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v || val <= 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_INITIAL_SIZE='%s' (must be > 0), using default (%d)", v, kDefaultInitialSize);

		} else {

			g.cfg.initial_size = (int)val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_INITIAL_SIZE=%ld", val);

		}

	}

}

inline bool active_comm_ready_or_stop() {

	return g.comm.active != MPI_COMM_NULL || g.sync.stop.load(std::memory_order_acquire);

}

inline bool attach_pending_cleared_or_stop() {

	return !g.sync.attach_pending.load(std::memory_order_acquire) || g.sync.stop.load(std::memory_order_acquire);

}

inline bool resize_pending_cleared_or_stop() {

	return !g.sync.resize_pending.load(std::memory_order_acquire) || g.sync.stop.load(std::memory_order_acquire);

}

void mal_init(MalResizePolicy policy) {

	g.cfg.resize_policy = policy;

	load_env_config();
	g.lb.auto_bw_est_bps = g.cfg.auto_bandwidth_bps;

	papi_init();

	int rc = MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.session);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Session_init", rc);
		std::abort();

	}

	rc = MPI_Group_from_session_pset(g.comm.session, "mpi://WORLD", &g.comm.world_group);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Group_from_session_pset", rc);
		std::abort();

	}

	rc = MPI_Comm_create_from_group(g.comm.world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.universe);

	if (rc != MPI_SUCCESS || g.comm.universe == MPI_COMM_NULL) {

		log_mpi_error("MPI_Comm_create_from_group(universe)", rc);
		std::abort();

	}

	rc = MPI_Comm_set_errhandler(g.comm.universe, MPI_ERRORS_RETURN);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Comm_set_errhandler(universe)", rc);
		std::abort();

	}

	rc = MPI_Comm_rank(g.comm.universe, &g.comm.u_rank);
	log_mpi_error("MPI_Comm_rank(universe)", rc);
	rc = MPI_Comm_size(g.comm.universe, &g.comm.u_size);
	log_mpi_error("MPI_Comm_size(universe)", rc);

	if (policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		validate_resize_sequence_against_universe_or_abort();

	}

	#if defined(__linux__) || defined(__APPLE__)

		pin_main_thread_to_pcore();

	#endif

	int effective_initial_size = g.cfg.initial_size > 0 ? std::min(g.cfg.initial_size, g.comm.u_size) : g.comm.u_size;

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		effective_initial_size = g.comm.u_size;

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "Initial active size: %d (universe=%d)", effective_initial_size, g.comm.u_size);

	int color = (g.comm.u_rank < effective_initial_size) ? 0 : MPI_UNDEFINED;
	rc = MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);
	log_mpi_error("MPI_Comm_split(active, init)", rc);

	g.worker = std::thread(progress_thread);

	#if defined(__linux__) || defined(__APPLE__)

		pin_worker_thread_to_ecore(g.worker);

	#endif

	if (g.comm.active != MPI_COMM_NULL) {

		rc = MPI_Comm_set_errhandler(g.comm.active, MPI_ERRORS_RETURN);
		log_mpi_error("MPI_Comm_set_errhandler(active, init)", rc);

		rc = MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		log_mpi_error("MPI_Comm_rank(active)", rc);
		rc = MPI_Comm_size(g.comm.active, &g.comm.a_size);
		log_mpi_error("MPI_Comm_size(active)", rc);

	} else {

		g.comm.a_rank = -1;
		g.comm.a_size = 0;

		g.sync.compute_wait(active_comm_ready_or_stop);

	}

}

void vec_scatter(MalVec& v, const void* root_data) {

	std::vector<int> sc, sd;

	if (g.comm.a_rank == 0) {

		const auto cuts = build_partition_cuts(v.total_N, g.comm.a_size);
		sc.resize(g.comm.a_size);

		for (int k = 0; k < g.comm.a_size; k++) {

			long ks = cuts[(size_t)k];
			long ke = cuts[(size_t)k + 1];
			long bytes = (ke - ks) * (long)v.elem_size;

			if (MAL_UNLIKELY(bytes > INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "SCATTER", "Per-rank send size overflow (%ld bytes) for rank=%d in vec_scatter", bytes, k);
				MPI_Abort(g.comm.universe, 1);

			}

			sc[k] = (int)bytes;

		}

		sd = make_displs(sc);

	}

	long rc_bytes = v.local_n * (long)v.elem_size;

	if (MAL_UNLIKELY(rc_bytes > INT_MAX)) {

		MAL_LOG_L(MAL_LOG_ERROR, "SCATTER", "Local receive size overflow (%ld bytes) in vec_scatter", rc_bytes);
		MPI_Abort(g.comm.universe, 1);

	}

	int rc = (int)rc_bytes;

	MPI_Scatterv(root_data, g.comm.a_rank == 0 ? sc.data() : nullptr, g.comm.a_rank == 0 ? sd.data() : nullptr, MPI_BYTE, v.buf, rc, MPI_BYTE, 0, g.comm.active);

}

void vec_gather(MalVec& v) {

	if (v.total_N == 0) {

		return;

	}

	if (v.local_n > v.done_n) {

		if (g.loop) {

			append_done_segments(v, *g.loop, v.plan_origin_n, v.done_n, v.local_n);

		} else {

			v.done_segs.push_back({v.buf_global_start + v.done_n, v.local_n - v.done_n});

		}

	}

	const int usiz = g.comm.u_size;
	const bool root = (g.comm.u_rank == v.gather_root);

	std::vector<long> my_seg_flat;
	my_seg_flat.reserve(v.done_segs.size() * 2);

	for (auto [s, c] : v.done_segs) {

		my_seg_flat.push_back(s);
		my_seg_flat.push_back(c);

	}

	int my_seg_count = (int)my_seg_flat.size();

	std::vector<int> seg_counts, seg_displs;

	if (root) {

		seg_counts.resize(usiz);

	}

	MPI_Gather(&my_seg_count, 1, MPI_INT, root ? seg_counts.data() : nullptr, 1, MPI_INT, v.gather_root, g.comm.universe);

	std::vector<long> all_segs;

	if (root) {

		seg_displs = make_displs(seg_counts);
		all_segs.resize(seg_displs.back() + seg_counts.back());

	}

	MPI_Gatherv(my_seg_flat.empty() ? nullptr : my_seg_flat.data(), my_seg_count, MPI_LONG, all_segs.empty() ? nullptr : all_segs.data(), root ? seg_counts.data() : nullptr, root ? seg_displs.data() : nullptr, MPI_LONG, v.gather_root, g.comm.universe);

	long my_done_elems = 0;

	for (size_t si = 1; si < my_seg_flat.size(); si += 2) {

		my_done_elems += my_seg_flat[si];

	}

	if (MAL_UNLIKELY(my_done_elems < 0)) {

		MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Negative done element count in vec_gather (%ld)", my_done_elems);
		MPI_Abort(g.comm.universe, 1);

	}

	if (MAL_UNLIKELY(my_done_elems > v.local_n)) {

		MAL_LOG_L(MAL_LOG_WARN, "GATHER", "done segments exceed local buffer (%ld > %ld), clamping", my_done_elems, v.local_n);
		my_done_elems = v.local_n;

	}

	long my_data_bytes = my_done_elems * (long)v.elem_size;

	if (MAL_UNLIKELY(my_data_bytes > INT_MAX)) {

		MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Local send size overflow in vec_gather (%ld bytes)", my_data_bytes);
		MPI_Abort(g.comm.universe, 1);

	}

	std::vector<int> data_counts, data_displs;
	int my_send_bytes = (int)my_data_bytes;
	std::vector<int> all_send_bytes;

	if (root) {

		all_send_bytes.resize(usiz);

	}

	MPI_Gather(&my_send_bytes, 1, MPI_INT, root ? all_send_bytes.data() : nullptr, 1, MPI_INT, v.gather_root, g.comm.universe);

	static thread_local void* tl_recv_raw = nullptr;
	static thread_local size_t tl_recv_cap = 0;
	void* recv_raw = nullptr;
	size_t recv_cap = 0;

	if (root) {

		data_counts.resize(usiz);

		for (int k = 0; k < usiz; k++) {

			if (MAL_UNLIKELY(all_send_bytes[k] < 0)) {

				MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Negative send size from rank=%d in vec_gather (%d)", k, all_send_bytes[k]);
				MPI_Abort(g.comm.universe, 1);

			}

			data_counts[k] = all_send_bytes[k];

		}

		data_displs = make_displs(data_counts);

		size_t total_recv = (size_t)(data_displs.back() + data_counts.back());
		pool_reserve(tl_recv_raw, tl_recv_cap, total_recv > 0 ? total_recv : 1, /*preserve_data=*/false);
		recv_raw = tl_recv_raw;
		recv_cap = tl_recv_cap;

	}

	MPI_Gatherv(my_data_bytes > 0 ? v.buf : nullptr, (int)my_data_bytes, MPI_BYTE, recv_raw, root ? data_counts.data() : nullptr, root ? data_displs.data() : nullptr, MPI_BYTE, v.gather_root, g.comm.universe);

	if (root && v.result_buf) {

		char* recv_buf = static_cast<char*>(recv_raw);

		for (int k = 0; k < usiz; k++) {

			long rank_used = 0;
			const long rank_bytes = data_counts[k];

			for (int s = 0; s < seg_counts[k] / 2; s++) {

				long gs = all_segs[seg_displs[k] + s * 2];
				long cnt = all_segs[seg_displs[k] + s * 2 + 1];
				const long seg_bytes = cnt * (long)v.elem_size;

				if (seg_bytes <= 0) {

					continue;

				}

				if (rank_used >= rank_bytes) {

					MAL_LOG_L(MAL_LOG_WARN, "GATHER", "Insufficient gathered bytes for rank=%d (segments exceed payload)", k);
					break;

				}

				const long available = rank_bytes - rank_used;
				const long copy_bytes = std::min(seg_bytes, available);

				std::memcpy(static_cast<char*>(v.result_buf) + gs * (long)v.elem_size, recv_buf + data_displs[k] + rank_used, copy_bytes);

				if (copy_bytes < seg_bytes) {

					MAL_LOG_L(MAL_LOG_WARN, "GATHER", "Truncated segment copy for rank=%d seg=%d (%ld/%ld bytes)", k, s, copy_bytes, seg_bytes);
					break;

				}

				rank_used += copy_bytes;

			}

		}

	}

	if (recv_raw) {

		tl_recv_raw = recv_raw;
		tl_recv_cap = recv_cap;

	}

}

void mal_finalize() {

	mal_wait_attach_tasks();

	// Do not force a local stop here: if one rank exits its worker early while others
	// are still inside resize-decision collectives, the job can deadlock/hard-abort.
	// Let the worker stop via the global runtime condition.
	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		g.sync.stop.store(true, std::memory_order_release);

	}

	g.sync.notify();
	g.worker.join();

	int rc = MPI_Barrier(g.comm.universe);
	log_mpi_error("MPI_Barrier(universe, pre-finalize)", rc);

	for (auto& vp : g.vecs) {

		MalVec& v = *vp;

		if (!v.buf) {

			continue;

		}

		if (v.gather_root >= 0) {

			vec_gather(v);

		}

		if (v.user_ptr) {

			*v.user_ptr = (g.comm.u_rank == v.gather_root && v.result_buf) ? v.result_buf : nullptr;

		}

		if (v.result_buf && !v.user_ptr) {

			std::free(v.result_buf);
			v.result_buf = nullptr;

		}

		v.free_resources();

	}

	g.vecs.clear();

	int naccs = (int)g.accs.size();
	int local_naccs = naccs;

	rc = MPI_Bcast(&naccs, 1, MPI_INT, 0, g.comm.universe);
	log_mpi_error("MPI_Bcast(naccs, finalize)", rc);

	if (MAL_UNLIKELY(naccs != local_naccs)) {

		MAL_LOG_L(MAL_LOG_WARN, "FINALIZE", "Acc count mismatch (local=%d root=%d); reducing identity for missing entries", local_naccs, naccs);

	}

	struct FinalAccGetter {

		MalAcc* operator()(int k) const {

			return (k >= 0 && (size_t)k < g.accs.size()) ? g.accs[(size_t)k].get() : nullptr;

		}

	};

	struct FinalAccSetter {

		void operator()(int k, const char* r, int) const {

			if (k < 0 || (size_t)k >= g.accs.size()) {

				return;

			}

			MalAcc* a = g.accs[(size_t)k].get();

			if (!a || !a->ptr) {

				return;

			}

			if (g.comm.u_rank == a->result_rank) {

				a->fn_set(a->ptr, r);

			} else {

				a->fn_reset(a->ptr);

			}

		}

	};

	batched_allreduce(naccs, FinalAccGetter{}, FinalAccSetter{});

	g.accs.clear();

	for (auto& sp : g.shared) {

		sp->free_resources();

	}

	g.shared.clear();

	if (g.pending) {

		g.pending.reset();

	}

	for (auto& e : g.gather_cache) {

		g_buffer_pool.release(e.ptr, e.bytes);

	}

	g.gather_cache.clear();

	if (g.comm.active != MPI_COMM_NULL) {

		rc = MPI_Barrier(g.comm.active);
		log_mpi_error("MPI_Barrier(active, pre-free)", rc);

		rc = MPI_Comm_free(&g.comm.active);
		log_mpi_error("MPI_Comm_free(active)", rc);
		g.comm.active = MPI_COMM_NULL;

	}

	rc = MPI_Barrier(g.comm.universe);
	log_mpi_error("MPI_Barrier(universe, pre-teardown)", rc);

	rc = MPI_Comm_free(&g.comm.universe);
	log_mpi_error("MPI_Comm_free(universe)", rc);
	g.comm.universe = MPI_COMM_NULL;

	rc = MPI_Group_free(&g.comm.world_group);
	log_mpi_error("MPI_Group_free(world_group)", rc);
	g.comm.world_group = MPI_GROUP_NULL;

	rc = MPI_Session_finalize(&g.comm.session);
	log_mpi_error("MPI_Session_finalize(session)", rc);
	g.comm.session = MPI_SESSION_NULL;

	papi_finalize();

}

inline bool can_enter_running_phase(const MalFor& f, bool ignore_attach_pending_gate = false) {

	if (g.sync.stop.load(std::memory_order_relaxed) || f.start >= f.end) {

		return false;

	}

	return ignore_attach_pending_gate || !g.sync.attach_pending.load(std::memory_order_relaxed);

}

inline void maybe_enter_running_phase(MalFor& f, bool ignore_attach_pending_gate = false) {

	if (can_enter_running_phase(f, ignore_attach_pending_gate)) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

}

MalFor mal_for(long total_iters, long& iter, long& limit) {

	MalFor f;

	f.user_iter = &iter;
	f.user_limit = &limit;

	if (g.sync.pending_has_ranges.load(std::memory_order_acquire)) {

		load_pending_ranges_into_loop(f);
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else if (g.comm.a_size > 0 && !g.sync.stop) {

		weighted_distribute(total_iters, g.comm.a_size, g.comm.a_rank, f.start, f.end);
		install_loop_plan(f, {{f.start, f.end}});

		g.lb.epoch_assigned = f.end - f.start;
		g.lb.epoch_start_time = MPI_Wtime();
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else {

		f.start = f.end = 0;
		f.plan_ranges.clear();
		f.plan_local_bases.clear();
		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);

	}

	set_iter(f, f.start);
	limit = f.end;

	{

		std::lock_guard lk(g.sync.mu);
		g.sync.compute_ready.store(false, std::memory_order_release);

	}

	g.loop = &f;

	const bool skip_idle_activation_wait = (f.start == f.end) && !g.sync.pending_has_ranges.load(std::memory_order_acquire) && (total_iters <= g.comm.a_size);

	while (!skip_idle_activation_wait && f.start == f.end && !g.sync.stop.load(std::memory_order_acquire)) {

		f.current = f.end;
		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);
		g.sync.compute_wait(has_work_or_stop);

		if (g.sync.stop.load(std::memory_order_acquire)) {

			break;

		}

		if (f.start == f.end && g.sync.pending_has_ranges.load(std::memory_order_acquire)) {

			load_pending_ranges_into_loop(f);
			f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

		}

	}

	maybe_enter_running_phase(f);

	return f;

}

void advance_next_range(MalFor& f) {

	f.plan_idx++;
	auto [a, b] = f.plan_ranges[f.plan_idx];

	f.start = a;

	sync_vec_mapping_for_current_range(f);

	set_limit(f, b);

	prime_range_start(f);

	MAL_LOG_L(MAL_LOG_DEBUG, "RANGE", "Next range [%ld, %ld) (base=%ld)", a, b, current_range_local_base(f));

}

void mal_check_for(MalFor& f) {

	g.sync.compute_epoch.fetch_add(1, std::memory_order_relaxed);

	if (g.sync.attach_pending.load(std::memory_order_acquire)) {

		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

		g.sync.compute_wait(attach_pending_cleared_or_stop);

		maybe_enter_running_phase(f, /*ignore_attach_pending_gate=*/true);

	}

	f.current = *f.user_iter;

	if (MAL_UNLIKELY(g.sync.resize_pending.load(std::memory_order_acquire))) {

		g.sync.compute_wait(resize_pending_cleared_or_stop);

		if (g.comm.active != MPI_COMM_NULL && f.start == f.end && g.pending && !g.pending->ranges.empty()) {

			load_pending_ranges_into_loop(f);

		}

		if (g.comm.active != MPI_COMM_NULL && f.start < f.end) {

			g.sync.loop_has_new_work.store(false, std::memory_order_relaxed);

			prime_range_start(f);

			return;

		}

	}

	if (MAL_LIKELY(*f.user_iter + 1 < f.end)) {

		return;

	}

	if (f.plan_idx + 1 < f.plan_ranges.size()) {

		advance_next_range(f);

		return;

	}

	if (g.sync.stop) {

		return;

	}

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		g.sync.stop.store(true, std::memory_order_release);
		g.sync.notify();
		return;

	}

	f.current = f.end;

	papi_rotate_epoch(g.lb.papi_prev_vals);
	g.sync.compute_wait(has_work_or_stop);

	if (!g.sync.stop) {

		prime_range_start(f);

	}

}

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_N, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	auto vp = std::make_unique<MalVec>();
	MalVec* v = vp.get();

	void* orig = user_ptr ? *user_ptr : nullptr;
	long n = f.end - f.start;
	long planned_total = total_range_iters(f.plan_ranges);

	if (planned_total > 0) {

		n = planned_total;

	}

	long buf_global_start = f.start;

	if (policy == MAL_ATTACH_PARTITIONED && g.comm.a_size > 0 && g.comm.a_rank >= 0) {

		const auto cuts = build_partition_cuts(total_N, g.comm.a_size);
		const long ds = cuts[(size_t)g.comm.a_rank];
		const long de = cuts[(size_t)g.comm.a_rank + 1];
		n = de - ds;
		buf_global_start = ds;

	}

	v->elem_size = elem_size;
	v->local_n = n;
	v->buf_global_start = buf_global_start;
	v->total_N = total_N;
	v->user_ptr = user_ptr;
	v->gather_root = result_rank;
	v->attach_policy = policy;
	v->access_mode = access_mode;
	v->cache_valid = false;

	if (result_rank >= 0 && g.comm.u_rank == result_rank) {

		v->result_buf = orig ? orig : checked_realloc(nullptr, total_N > 0 ? (size_t)total_N * elem_size : 1, "mal_attach_vec.result_buf");

	}

	const bool shared_active = (policy == MAL_ATTACH_SHARED_ACTIVE);
	bool once_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (MAL_UNLIKELY(elem_size == 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with elem_size=0");

	}

	if (MAL_UNLIKELY(total_N < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with negative total_N=%ld", total_N);

	}

	if ((once_all || shared_active) && result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "Shared vector policy ignores gather result_rank=%d", result_rank);

		if (v->result_buf != nullptr && v->result_buf != orig) {

			std::free(v->result_buf);

		}

		v->result_buf = nullptr;
		v->gather_root = -1;
		once_all = false;
		result_rank = -1;

	}

	if (once_all || shared_active) {

		n = total_N;
		v->local_n = n;
		v->done_n = 0;
		v->buf_global_start = 0;

	}

	v->buf = static_cast<char*>(g_buffer_pool.acquire((v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size));
	v->buf_bytes = (v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size;
	v->plan_origin_n = v->done_n;

	int idx = (int)f.vecs.size();

	if (g.pending) {

		StagedBuffer stash = take_pending_vec_slice(idx);

		if (stash.ptr && n > 0) {

			std::memcpy(v->buf, stash.ptr, (size_t)n * elem_size);
			g_buffer_pool.release(stash.ptr, stash.bytes);

		}

	}

	const bool inactive_no_pending = (g.comm.active == MPI_COMM_NULL && !g.pending && policy == MAL_ATTACH_PARTITIONED);

	if (inactive_no_pending && user_ptr) {

		*user_ptr = nullptr;

	} else {

		v->sync_user_ptr();

	}

	f.vecs.push_back(v);
	g.vecs.push_back(std::move(vp));

	const bool can_dispatch_attach = (!g.pending && g.comm.active != MPI_COMM_NULL);

	if (can_dispatch_attach) {

		const size_t total_bytes = (size_t)std::max(0L, total_N) * elem_size;

		if (once_all) {

			run_shared_all_attach_bcast(v->buf, orig, total_bytes, result_rank, exec_mode, "MAL_ATTACH_SHARED_ALL vector has null root pointer; broadcasting zero-initialized data");

		} else if (shared_active) {

			run_shared_active_attach_bcast(*v, orig, total_bytes, exec_mode);

		} else {

			run_partitioned_attach_scatter(*v, orig, result_rank, total_bytes, exec_mode);

		}

	}

	if (g.pending && idx + 1 == (int)g.pending->vec_slices.size()) {

		g.pending->vec_slices.clear();

	}

	maybe_enter_running_phase(f, /*ignore_attach_pending_gate=*/!async_attach);

}

void mal_attach_vec(MalForND& f, void** user_ptr, size_t elem_size, long total_N, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_vec(mal_for_nd_base(f), user_ptr, elem_size, total_N, result_rank, policy, exec_mode, access_mode);

}

void detail::acc_register(MalFor& f, detail::AccDesc d, int result_rank) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	auto ap = std::make_unique<MalAcc>();
	MalAcc* a = ap.get();

	a->result_rank = result_rank;
	a->ptr = d.ptr;
	a->dtype_idx = dtype_tag(d.dtype);
	a->dop_idx = dop_tag(d.dop);
	a->esz = d.esz;
	a->fn_get = d.fn_get;
	a->fn_set = d.fn_set;
	a->fn_add = d.fn_add;
	a->fn_reset = d.fn_reset;

	a->epoch_buf = take_pending_acc_epoch_buf(a->esz);

	a->fn_reset(a->ptr);

	f.accs.push_back(a);
	g.accs.push_back(std::move(ap));

	maybe_enter_running_phase(f, /*ignore_attach_pending_gate=*/true);

}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	if (MAL_UNLIKELY(elem_size == 0 || primary_n < 0 || secondary_n < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_mat called with invalid shape/size elem_size=%zu primary_n=%ld secondary_n=%ld", elem_size, primary_n, secondary_n);

	}

	if (policy == MAL_ATTACH_PARTITIONED) {

		mal_attach_vec(f, user_ptr, elem_size * (size_t)secondary_n, primary_n, result_rank, MAL_ATTACH_PARTITIONED, exec_mode, access_mode);

		return;

	}

	if (MAL_UNLIKELY(elem_size > 0 && secondary_n > 0 && (size_t)secondary_n > SIZE_MAX / elem_size)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_mat row stride overflow secondary_n=%ld elem_size=%zu", secondary_n, elem_size);
		MPI_Abort(g.comm.universe, 1);

	}

	const size_t row_bytes = (size_t)secondary_n * elem_size;

	if (MAL_UNLIKELY(row_bytes > 0 && primary_n > 0 && (size_t)primary_n > SIZE_MAX / row_bytes)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_mat total bytes overflow primary_n=%ld secondary_n=%ld elem_size=%zu", primary_n, secondary_n, elem_size);
		MPI_Abort(g.comm.universe, 1);

	}

	const size_t total_bytes = (size_t)primary_n * row_bytes;
	const bool shared_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (!shared_all && g.comm.active == MPI_COMM_NULL) {

		if (user_ptr) {

			*user_ptr = nullptr;

		}

		auto sp = std::make_unique<SharedMat>();
		sp->buf = nullptr;
		sp->total_bytes = total_bytes;
		sp->user_owned = false;
		sp->user_ptr = user_ptr;

		g.shared.push_back(std::move(sp));

		maybe_enter_running_phase(f, /*ignore_attach_pending_gate=*/!async_attach);

		return;

	}

	if (result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "result_rank=%d ignored for shared matrix policies", result_rank);

	}

	void* orig = user_ptr ? *user_ptr : nullptr;
	void* buf;

	if (g.pending && !shared_all) {

		StagedBuffer staged = take_pending_shared_mat();

		if (staged.ptr) {

			buf = staged.ptr;

		} else {

			buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);

		}

	} else {

		if (shared_all) {

			buf = g_buffer_pool.acquire(total_bytes > 0 ? total_bytes : 1);
			run_shared_all_attach_bcast(buf, orig, total_bytes, -1, exec_mode, "MAL_ATTACH_SHARED_ALL matrix has null root pointer; broadcasting zero-initialized data");

		} else {

			buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);

		}

	}

	if (user_ptr) {

		*user_ptr = buf;

	}

	auto sp = std::make_unique<SharedMat>();
	sp->buf = buf;
	sp->total_bytes = total_bytes;
	sp->user_owned = shared_all ? (g.comm.u_rank == 0 && orig != nullptr) : (g.comm.a_rank == 0 && orig != nullptr);
	sp->user_ptr = user_ptr;

	g.shared.push_back(std::move(sp));

	maybe_enter_running_phase(f, /*ignore_attach_pending_gate=*/!async_attach);

}

void mal_attach_mat(MalForND& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_mat(mal_for_nd_base(f), user_ptr, elem_size, primary_n, secondary_n, result_rank, policy, exec_mode, access_mode);

}
