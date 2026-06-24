#ifndef MALLEABLE_RESIZER_CPP_INCLUDED
#define MALLEABLE_RESIZER_CPP_INCLUDED

#include "malleable_types.cpp"

constexpr double kEpsDone = 1.0;
constexpr double kEpsElapsed = 1e-6;
constexpr double kEpsThroughput = 1e-9;
constexpr double kEpsWeight = 1e-12;
constexpr int kLbGatherFields = 2;
constexpr int kFusedGatherFields = 1 + kLbGatherFields;
constexpr int kMinResizeCooldownEpochs = 2;
constexpr int kMinBaselineEpochs = 3;

inline const double* lb_row_at(const std::vector<double>& all_lb_buf, int k) {

	return &all_lb_buf[(size_t)k * (size_t)kLbGatherFields];

}

inline double lb_row_tp(const double* row) {

	return (row[0] > 0.0 && row[1] > kEpsElapsed) ? (row[0] / row[1]) : 0.0;

}

inline bool reuse_flag_at(const std::vector<int>& all_reuse_flags, int n, int rank, int vi) {

	return all_reuse_flags[(size_t)rank * (size_t)n + (size_t)vi] != 0;

}

class Resizer {

	std::vector<std::pair<long,long>> remaining_;
	std::vector<long> remaining_offsets_;
	std::vector<long> target_cuts_;
	long total_rem_{0};

	struct VecTask {
		MalVec* v{nullptr};
		StagedBuffer gathered;
	};

	struct VecMeta {
		size_t esz{0};
		int shared_active{0};
	};

	std::vector<VecTask> vtasks_;
	std::vector<VecMeta> vmeta_;
	std::vector<long> rem_per_rank_;
	std::vector<std::vector<char>> new_epoch_bufs_;
	std::vector<std::pair<long,long>> scratch_assigned_;
	std::vector<int> scratch_reuse_flags_;
	std::vector<int> scratch_all_reuse_flags_;

	std::vector<long> flat_buf_;
	std::vector<int> flat_counts_buf_;
	std::vector<int> flat_displs_buf_;
	std::vector<double> all_lb_buf_;
	std::vector<double> fused_gather_buf_;

	std::vector<TransferPlanEntry> build_transfer_plan(const std::vector<long>& old_vs) const;
	void init_vec_tasks(int n, int nvecs, bool was_active);
	void reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) ;
	void exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags);

	void collect_ranges();
	void redistribute_vecs(int n_vecs);
	void reduce_accs(int n_accs);
	void apply_active();
	void apply_inactive();
	void broadcast_shared_vecs();
	void broadcast_shared_mats();
	void stash_gather_cache();

	int target_;
	int old_a_size_{0};
	long my_new_vs_{0};
	long my_new_count_{0};
	long confirmed_snapshot_{LONG_MIN};

public:

	explicit Resizer(int target) : target_(target) {}

	~Resizer() {

		for (auto& t : vtasks_) {

			if (t.gathered.ptr) {

				g_buffer_pool.release(t.gathered.ptr, t.gathered.bytes);

			}

		}

	}

	void prepare_phase();
	void commit_phase();

};

struct EpochMetrics {
	double global_thr{0.0};
	double global_remaining{0.0};
	int active_n{0};
	bool any_has_loop{false};
};

EpochMetrics gather_epoch_metrics() {

	EpochMetrics m;
	const int U = g.comm.u_size;
	static thread_local std::vector<double> per_rank_thr;
	per_rank_thr.assign((size_t)U, 0.0);

	double local_rem = 0.0;

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		const long cur = *g.loop->user_iter;

		if (cur + 1 < g.loop->end) {

			local_rem += (double)(g.loop->end - (cur + 1));

		}

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			local_rem += (double)(g.loop->plan_ranges[ri].second - g.loop->plan_ranges[ri].first);

		}

	}

	const double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	const long my_done = std::max(0L, g.lb.epoch_assigned - (long)local_rem);
	const double my_thr = (my_elapsed > kEpsElapsed && my_done > 0) ? (double)my_done / my_elapsed : 0.0;
	const double my_active_n = (double)g.comm.a_size;

	const double my_has_loop = (g.loop && g.loop->phase.load(std::memory_order_relaxed) != MAL_LOOP_WAITING_ACTIVATION) ? 1.0 : 0.0;
	constexpr int kGatherFields = 5;
	double send[kGatherFields] = {local_rem, my_thr, my_active_n, my_elapsed, my_has_loop};
	static thread_local std::vector<double> recv;
	recv.assign((size_t)U * kGatherFields, 0.0);
	MPI_Allgather(send, kGatherFields, MPI_DOUBLE, recv.data(), kGatherFields, MPI_DOUBLE, g.comm.universe);

	for (int k = 0; k < U; k++) {

		const double rem_k = recv[(size_t)k * kGatherFields + 0];
		const double thr_k = recv[(size_t)k * kGatherFields + 1];
		const int an_k = (int)std::lround(recv[(size_t)k * kGatherFields + 2]);

		m.active_n = std::max(m.active_n, an_k);
		m.global_remaining += rem_k;
		per_rank_thr[(size_t)k] = thr_k;
		if (recv[(size_t)k * kGatherFields + 4] > 0.5) m.any_has_loop = true;

	}

	for (int k = 0; k < m.active_n; k++) {

		m.global_thr += per_rank_thr[(size_t)k];

	}

	const double r0_elapsed = recv[0 * kGatherFields + 3];

	if (m.active_n == 1 && m.global_thr > kEpsThroughput) {

		const double min_valid_s = std::max(0.02, g.cfg.epoch_ms.load(std::memory_order_relaxed) * 0.0005);

		if (r0_elapsed >= min_valid_s) {

			using Phase = MalState::LoadBalance::Phase;
			const bool building = (g.lb.bs_phase == Phase::IDLE || g.lb.bs_phase == Phase::NEEDS_BASELINE);
			const double alpha = building ? 0.5 : 0.9;

			g.lb.thr_single_proc = (g.lb.thr_single_proc <= kEpsThroughput) ? m.global_thr : alpha * g.lb.thr_single_proc + (1.0 - alpha) * m.global_thr;

			if (building) {

				g.lb.bs_baseline_count++;

			}

		}

	}

	if (m.active_n > 0) {

		std::lock_guard<std::mutex> lk(g.lb.weights_mu);

		if ((int)g.lb.weights.size() < U) {

			g.lb.weights.assign((size_t)U, 0.0);

		}

		if (m.global_thr > kEpsWeight) {

			for (int k = 0; k < U; k++) {

				g.lb.weights[(size_t)k] = (k < m.active_n) ? per_rank_thr[(size_t)k] / m.global_thr : 0.0;

			}

		} else {

			const double w = 1.0 / (double)m.active_n;

			for (int k = 0; k < U; k++) {

				g.lb.weights[(size_t)k] = (k < m.active_n) ? w : 0.0;

			}

		}

	}

	return m;

}

static double get_efficiency_threshold() {

	switch (g.cfg.resize_policy) {
		case MAL_RESIZE_POLICY_THROUGHPUT: return 0.0;
		case MAL_RESIZE_POLICY_EFFICIENCY: return 0.8;
		case MAL_RESIZE_POLICY_COST: return 0.9;
		default: return 0.6;
	}

}


void Resizer::collect_ranges() {

	std::vector<long> local;
	local.reserve(16);

	confirmed_snapshot_ = (g.loop && g.comm.active != MPI_COMM_NULL) ? g.loop->confirmed_iter.load(std::memory_order_acquire) : LONG_MIN;

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		const long first_s = confirmed_snapshot_ + 1;
		const long first_e = g.loop->end;

		if (first_s < first_e) {

			local.push_back(first_s);
			local.push_back(first_e);

		}

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			const long s = g.loop->plan_ranges[ri].first;
			const long e = g.loop->plan_ranges[ri].second;

			if (s < e) {

				local.push_back(s);
				local.push_back(e);

			}

		}

	}

	int my_count = (int)local.size();

	long my_rem_local = 0;

	for (size_t i = 0; i + 1 < local.size(); i += 2) {

		my_rem_local += local[i + 1] - local[i];

	}

	double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	long my_done = std::max(0L, g.lb.epoch_assigned - my_rem_local);

	double fused_send[kFusedGatherFields] = {(double)my_count, (double)my_done, my_elapsed};
	fused_gather_buf_.resize((size_t)g.comm.u_size * (size_t)kFusedGatherFields);

	MPI_Allgather(fused_send, kFusedGatherFields, MPI_DOUBLE, fused_gather_buf_.data(), kFusedGatherFields, MPI_DOUBLE, g.comm.universe);

	flat_counts_buf_.resize(g.comm.u_size);
	all_lb_buf_.resize((size_t)g.comm.u_size * (size_t)kLbGatherFields);

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = &fused_gather_buf_[(size_t)k * (size_t)kFusedGatherFields];
		flat_counts_buf_[k] = (int)row[0];
		std::memcpy(&all_lb_buf_[(size_t)k * (size_t)kLbGatherFields], row + 1, kLbGatherFields * sizeof(double));

	}

	flat_displs_buf_ = make_displs(flat_counts_buf_);

	int total = flat_displs_buf_.back() + flat_counts_buf_.back();

	if (total == 0) {

		remaining_.clear();
		remaining_offsets_.assign(1, 0);
		total_rem_ = 0;
		rem_per_rank_.assign(g.comm.u_size, 0);

		return;

	}

	flat_buf_.resize(total > 0 ? (size_t)total : 1);

	MPI_Allgatherv(local.empty() ? nullptr : local.data(), my_count, MPI_LONG, flat_buf_.data(), flat_counts_buf_.data(), flat_displs_buf_.data(), MPI_LONG, g.comm.universe);

	remaining_.clear();
	remaining_.reserve(total / 2 + 1);
	remaining_offsets_.clear();
	remaining_offsets_.reserve(total / 2 + 2);
	remaining_offsets_.push_back(0);

	total_rem_ = 0;
	rem_per_rank_.assign(g.comm.u_size, 0);

	for (int k = 0; k < g.comm.u_size; k++) {

		int disp = flat_displs_buf_[k];
		int nranges = flat_counts_buf_[k] / 2;

		for (int p = 0; p < nranges; p++) {

			long s = flat_buf_[disp + p * 2];
			long e = flat_buf_[disp + p * 2 + 1];
			long len = e - s;

			remaining_.push_back({s, e});
			remaining_offsets_.push_back(total_rem_ + len);
			rem_per_rank_[k] += len;
			total_rem_ += len;

		}

	}

	{

		std::lock_guard<std::mutex> lk(g.lb.weights_mu);

		if ((int)g.lb.weights.size() < g.comm.u_size) {

			g.lb.weights.assign((size_t)g.comm.u_size, 0.0);

		}

		const bool is_scale_up = (target_ > old_a_size_);

		if (is_scale_up) {

			const double w = 1.0 / (double)std::max(1, target_);

			for (int k = 0; k < g.comm.u_size; k++) {

				g.lb.weights[(size_t)k] = (k < target_) ? w : 0.0;

			}

		} else {

			double total_tp = 0.0;

			for (int k = 0; k < target_; k++) {

				total_tp += lb_row_tp(lb_row_at(all_lb_buf_, k));

			}

			if (total_tp > kEpsWeight) {

				for (int k = 0; k < g.comm.u_size; k++) {

					g.lb.weights[(size_t)k] = (k < target_) ? lb_row_tp(lb_row_at(all_lb_buf_, k)) / total_tp : 0.0;

				}

			} else {

				const double w = 1.0 / (double)std::max(1, target_);

				for (int k = 0; k < g.comm.u_size; k++) {

					g.lb.weights[(size_t)k] = (k < target_) ? w : 0.0;

				}

			}

		}

	}

	const double* my_row = lb_row_at(all_lb_buf_, g.comm.u_rank);
	MAL_LOG(MAL_LOG_DEBUG, "LB: epoch done=%.0f elapsed=%.3fs thr=%.1f iters/s weight=%.4f", my_row[0], my_row[1], lb_row_tp(my_row), g.comm.u_rank < (int)g.lb.weights.size() ? g.lb.weights[(size_t)g.comm.u_rank] : 0.0);

}


std::vector<TransferPlanEntry> Resizer::build_transfer_plan(const std::vector<long>& old_vs) const {

	std::vector<TransferPlanEntry> plan;
	plan.reserve((size_t)g.comm.u_size + (size_t)target_);

	if (target_ <= 0) {

		return plan;

	}

	int oi = 0;
	int ni = 0;
	long nv_s = target_cuts_[0];
	long nv_e = target_cuts_[1];

	while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

		oi++;

	}

	while (oi < g.comm.u_size && ni < target_) {

		long ov_e = old_vs[(size_t)oi + 1];
		long seg_s = std::max(old_vs[(size_t)oi], nv_s);
		long seg_e = std::min(ov_e, nv_e);

		if (seg_s < seg_e) {

			plan.push_back({oi, ni, seg_s, seg_e - seg_s});

		}

		if (ov_e <= nv_e) {

			oi++;

			while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

				oi++;

			}

		}

		if (nv_e <= ov_e) {

			ni++;

			if (ni < target_) {

				nv_s = target_cuts_[(size_t)ni];
				nv_e = target_cuts_[(size_t)ni + 1];

			}

		}

	}

	return plan;

}

void Resizer::init_vec_tasks(int n, int nvecs, bool was_active) {

	vtasks_.resize(n);

	if (g.gather_cache.size() < (size_t)n) {

		g.gather_cache.resize((size_t)n);

	}

	for (int vi = 0; vi < n; vi++) {

		auto& t = vtasks_[vi];

		t.v = (vi < nvecs) ? g.loop->vecs[vi] : nullptr;
		t.gathered = g.gather_cache[(size_t)vi];
		g.gather_cache[(size_t)vi] = {};

		if (vmeta_[vi].shared_active) {

			if (t.v) {

				t.v->done_n = 0;

			}

			continue;

		}

		if (!t.v) {

			continue;

		}

		long old_done = t.v->done_n;
		long new_done = old_done;

		if (was_active) {

			new_done = std::clamp(confirmed_snapshot_ + 1 - t.v->buf_global_start, 0L, t.v->local_n);

		}

		if (new_done > old_done) {

			append_done_segments(*t.v, *g.loop, t.v->plan_origin_n, old_done, new_done);

		}

		t.v->done_n = new_done;
		advance_read_only_cache_after_progress(*t.v, old_done, new_done);

	}

}

void Resizer::reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) {

	if (!am_receiver || my_new_count <= 0) {

		return;

	}

	bool has_local_assignment = false;

	for (const auto& tr : plan) {

		if (tr.new_rank != g.comm.u_rank) {

			continue;

		}

		has_local_assignment = true;
		break;

	}

	if (!has_local_assignment) {

		return;

	}

	for (int vi = 0; vi < n; vi++) {

		if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, g.comm.u_rank, vi)) {

			continue;

		}

		size_t bytes = (size_t)my_new_count * vmeta_[vi].esz;
		void* gp = vtasks_[vi].gathered.ptr;
		size_t gc = vtasks_[vi].gathered.bytes;

		pool_reserve(gp, gc, bytes, /*preserve_data=*/false);
		vtasks_[vi].gathered = {gp, gc};

	}

}

void Resizer::exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags) {

	std::vector<MPI_Request> reqs;
	reqs.reserve(plan.size() * 2);

	std::vector<StagedBuffer> packed_sends;
	std::vector<StagedBuffer> packed_recvs;
	packed_sends.reserve(plan.size());
	packed_recvs.reserve(plan.size());

	struct PendingPackedRecv {

		const TransferPlanEntry* tr{nullptr};
		void* buf{nullptr};
		size_t bytes{0};

	};

	std::vector<PendingPackedRecv> pending_packed_recvs;
	pending_packed_recvs.reserve(plan.size());

	const int packed_tag = n;

	for (const auto& tr : plan) {

		const bool local_sender = (tr.old_rank == g.comm.u_rank);
		const bool local_recv = (tr.new_rank == g.comm.u_rank);

		if (!local_sender && !local_recv) {

			continue;

		}

		if (tr.old_rank == tr.new_rank) {

			if (!local_sender) {

				continue;

			}

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				long bytes64 = tr.v_count * (long)esz;

				if (MAL_UNLIKELY(bytes64 > INT_MAX)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
					MPI_Abort(g.comm.universe, 1);

				}

				int byte_count = (int)bytes64;

				if (byte_count == 0) {

					continue;

				}

				const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (send_base && t.gathered.ptr) {

					long src_off = (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz;
					long dst_off = (tr.v_start - my_new_vs_) * (long)esz;

					std::memmove(static_cast<char*>(t.gathered.ptr) + dst_off, send_base + src_off, byte_count);

				}

			}

			continue;

		}

		size_t packed_bytes = 0;
		int eligible_vecs = 0;

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

				continue;

			}

			const size_t b = (size_t)tr.v_count * vmeta_[vi].esz;

			if (b > 0) {

				packed_bytes += b;
				eligible_vecs++;

			}

		}

		if (packed_bytes == 0) {

			continue;

		}

		if (packed_bytes > (size_t)INT_MAX || eligible_vecs == 1) {

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				long bytes64 = tr.v_count * (long)esz;

				if (MAL_UNLIKELY(bytes64 > INT_MAX)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
					MPI_Abort(g.comm.universe, 1);

				}

				int byte_count = (int)bytes64;

				if (byte_count == 0) {

					continue;

				}

				const char* send_base = ((tr.old_rank == g.comm.u_rank) && t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (tr.old_rank == g.comm.u_rank) {

					if (MAL_UNLIKELY(!send_base)) {

						MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing sender buffer for old_rank=%d vec=%d", tr.old_rank, vi);
						MPI_Abort(g.comm.universe, 1);

					}

					MPI_Request req;

					MPI_Isend(send_base + (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz, byte_count, MPI_BYTE, tr.new_rank, vi, g.comm.universe, &req);
					reqs.push_back(req);

				}

				if (tr.new_rank == g.comm.u_rank) {

					if (MAL_UNLIKELY(!t.gathered.ptr)) {

						MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing receiver buffer for new_rank=%d vec=%d", tr.new_rank, vi);
						MPI_Abort(g.comm.universe, 1);

					}

					char* dst = static_cast<char*>(t.gathered.ptr) + (tr.v_start - my_new_vs_) * (long)esz;
					MPI_Request req;

					MPI_Irecv(dst, byte_count, MPI_BYTE, tr.old_rank, vi, g.comm.universe, &req);
					reqs.push_back(req);

				}

			}

			continue;

		}

		if (local_sender) {

			void* send_buf = g_buffer_pool.acquire(packed_bytes);
			size_t off = 0;
			char* dst = static_cast<char*>(send_buf);

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				const size_t bytes = (size_t)tr.v_count * esz;

				if (bytes == 0) {

					continue;

				}

				const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (MAL_UNLIKELY(!send_base)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing sender buffer while packing old_rank=%d vec=%d", tr.old_rank, vi);
					MPI_Abort(g.comm.universe, 1);

				}

				long src_off = (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz;
				std::memcpy(dst + off, send_base + src_off, bytes);
				off += bytes;

			}

			packed_sends.push_back({send_buf, packed_bytes});

			MPI_Request req;

			MPI_Isend(send_buf, (int)packed_bytes, MPI_BYTE, tr.new_rank, packed_tag, g.comm.universe, &req);
			reqs.push_back(req);

		}

		if (local_recv) {

			void* recv_buf = g_buffer_pool.acquire(packed_bytes);
			packed_recvs.push_back({recv_buf, packed_bytes});

			pending_packed_recvs.push_back({&tr, recv_buf, packed_bytes});

			MPI_Request req;

			MPI_Irecv(recv_buf, (int)packed_bytes, MPI_BYTE, tr.old_rank, packed_tag, g.comm.universe, &req);
			reqs.push_back(req);

		}

	}

	if (!reqs.empty()) {

		MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

	}

	for (const auto& pr : pending_packed_recvs) {

		if (pr.tr && pr.buf && pr.bytes > 0) {

			size_t off = 0;
			const TransferPlanEntry& tr = *pr.tr;
			const char* src = static_cast<const char*>(pr.buf);

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				const size_t bytes = (size_t)tr.v_count * esz;

				if (bytes == 0) {

					continue;

				}

				if (MAL_UNLIKELY(!t.gathered.ptr)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing receiver buffer while unpacking new_rank=%d vec=%d", tr.new_rank, vi);
					MPI_Abort(g.comm.universe, 1);

				}

				long dst_off = (tr.v_start - my_new_vs_) * (long)esz;
				std::memcpy(static_cast<char*>(t.gathered.ptr) + dst_off, src + off, bytes);
				off += bytes;

			}

		}

	}

	for (auto& b : packed_sends) {

		g_buffer_pool.release(b.ptr, b.bytes);

	}

	for (auto& b : packed_recvs) {

		g_buffer_pool.release(b.ptr, b.bytes);

	}

}

void Resizer::redistribute_vecs(int n) {

	if (n == 0) {

		return;

	}

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;

	std::vector<long> old_vs(g.comm.u_size + 1, 0);
	std::inclusive_scan(rem_per_rank_.begin(), rem_per_rank_.end(), old_vs.begin() + 1);
	target_cuts_.clear();

	if (target_ > 0) {

		target_cuts_ = build_partition_cuts(total_rem_, target_);

	}

	const auto plan = build_transfer_plan(old_vs);

	bool was_active = (g.comm.active != MPI_COMM_NULL);
	init_vec_tasks(n, nvecs, was_active);

	my_new_vs_ = 0;
	my_new_count_ = 0;
	bool am_receiver = (g.comm.u_rank < target_);

	if (am_receiver) {

		long my_nv_e = target_cuts_[(size_t)g.comm.u_rank + 1];
		my_new_vs_ = target_cuts_[(size_t)g.comm.u_rank];
		my_new_count_ = my_nv_e - my_new_vs_;

	}

	scratch_assigned_.clear();

	if (am_receiver && my_new_count_ > 0) {

		scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, my_new_vs_, my_new_vs_ + my_new_count_);

	}

	scratch_reuse_flags_.assign((size_t)n, 0);

	if (am_receiver && my_new_count_ > 0) {

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || !vtasks_[vi].v) {

				continue;

			}

			if (vec_can_reuse_assigned_ranges(*vtasks_[vi].v, scratch_assigned_)) {

				scratch_reuse_flags_[(size_t)vi] = 1;

			}

		}

	}

	scratch_all_reuse_flags_.assign((size_t)g.comm.u_size * (size_t)n, 0);

	MPI_Allgather(scratch_reuse_flags_.data(), n, MPI_INT, scratch_all_reuse_flags_.data(), n, MPI_INT, g.comm.universe);

	reserve_receiver_buffers(n, am_receiver, plan, my_new_count_, scratch_all_reuse_flags_);

	int n_sources = 0;
	int source_rank = -1;

	for (int k = 0; k < g.comm.u_size; k++) {

		if (rem_per_rank_[k] > 0) {

			n_sources++;
			source_rank = k;

		}

	}

	if (n_sources == 1 && target_ > 1) {

		std::vector<int> scounts(g.comm.u_size, 0);
		std::vector<int> sdispls(g.comm.u_size, 0);

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active) {

				continue;

			}

			const size_t esz = vmeta_[vi].esz;
			auto& t = vtasks_[vi];

			const char* sendbuf = nullptr;

			if (g.comm.u_rank == source_rank && t.v && was_active) {

				sendbuf = static_cast<char*>(t.v->buf) + t.v->done_n * (long)esz;

				for (int j = 0; j < target_; j++) {

					if (reuse_flag_at(scratch_all_reuse_flags_, n, j, vi)) {

						scounts[j] = 0;
						sdispls[j] = 0;

					} else {

						long count = target_cuts_[(size_t)j + 1] - target_cuts_[(size_t)j];
						long bytes = count * (long)esz;

						scounts[j] = (bytes <= INT_MAX) ? (int)bytes : 0;
						sdispls[j] = (int)(target_cuts_[(size_t)j] * (long)esz);

					}

				}

				for (int j = target_; j < g.comm.u_size; j++) {

					scounts[j] = 0;
					sdispls[j] = 0;

				}

			}

			int recvcount = (am_receiver && my_new_count_ > 0 && t.gathered.ptr) ? (int)(my_new_count_ * (long)esz) : 0;

			MPI_Scatterv(sendbuf, scounts.data(), sdispls.data(), MPI_BYTE, t.gathered.ptr, recvcount, MPI_BYTE, source_rank, g.comm.universe);

		}

	} else {

		exchange_vec_data(n, was_active, old_vs, plan, scratch_all_reuse_flags_);

	}

}

void Resizer::reduce_accs(int n) {

	if (n == 0) {

		return;

	}

	int naccs = g.loop ? (int)g.loop->accs.size() : 0;

	new_epoch_bufs_.resize(n);
	std::vector<MalAcc*>& loop_accs = g.loop->accs;

	struct AccGetter {

		int naccs;
		const std::vector<MalAcc*>* accs;

		MalAcc* operator()(int k) const {

			return k < naccs ? (*accs)[(size_t)k] : nullptr;

		}

	};

	struct AccResultSetter {

		int naccs;
		const std::vector<MalAcc*>* accs;
		std::vector<std::vector<char>>* epoch_bufs;

		void operator()(int k, const char* r, int esz) const {

			(*epoch_bufs)[(size_t)k].assign(r, r + esz);

			if (k >= naccs) {

				return;

			}

			MalAcc* a = (*accs)[(size_t)k];

			a->epoch_buf.assign(r, r + esz);
			write_identity(static_cast<char*>(a->ptr), a->dtype_idx, a->dop_idx, (int)a->esz);

		}

	};

	batched_allreduce(n, AccGetter{naccs, &loop_accs}, AccResultSetter{naccs, &loop_accs, &new_epoch_bufs_});

}

void Resizer::apply_active() {

	if (target_cuts_.size() != (size_t)g.comm.a_size + 1) {

		target_cuts_ = build_partition_cuts(total_rem_, g.comm.a_size);

	}

	const std::vector<long>& active_cuts = target_cuts_;
	long vstart = active_cuts[(size_t)g.comm.a_rank];
	long vend = active_cuts[(size_t)g.comm.a_rank + 1];
	scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, vstart, vend);
	auto& assigned = scratch_assigned_;

	long new_asgn = vend - vstart;
	const bool has_assigned_ranges = (new_asgn > 0 && !assigned.empty());

	const bool waiting_for_activation = g.loop && g.loop->phase.load(std::memory_order_acquire) == MAL_LOOP_WAITING_ACTIVATION;
	bool publish_pending_after_broadcast = false;
	std::vector<std::pair<long,long>> deferred_pending_ranges;

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "a_rank=%d assigned %zu range(s) (%ld iters, weight=%.4f)", g.comm.a_rank, assigned.size(), new_asgn, g.comm.a_rank < (int)g.lb.weights.size() ? g.lb.weights[g.comm.a_rank] : 1.0 / g.comm.a_size);

	g.lb.epoch_assigned = new_asgn;
	g.lb.epoch_start_time = MPI_Wtime();

	if (g.loop && !waiting_for_activation) {

		for (size_t ti = 0; ti < vtasks_.size(); ti++) {

			auto& t = vtasks_[ti];

			if (!t.v) {

				continue;

			}

			if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE || t.v->attach_policy == MAL_ATTACH_SHARED_ALL) {

				configure_shared_active_vec(*t.v, (size_t)std::max(1L, t.v->total_N) * vmeta_[ti].esz);

				continue;

			}

			long new_local = t.v->done_n + new_asgn;
			bool reused_local = false;

			if (has_assigned_ranges && !t.gathered.ptr) {

				reused_local = vec_reuse_local_copy(*t.v, assigned, t.v->done_n);

			}

			size_t buf_need = (size_t)std::max(1L, new_local) * vmeta_[ti].esz;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);

			if (has_assigned_ranges && !reused_local && t.gathered.ptr) {

				long buf_off = t.v->done_n;
				long gathered_off = 0;

				for (auto [g_start, g_end] : assigned) {

					long len = g_end - g_start;

					if (len > 0 && gathered_off + len <= my_new_count_) {

						std::memcpy(static_cast<char*>(t.v->buf) + buf_off * vmeta_[ti].esz, static_cast<char*>(t.gathered.ptr) + gathered_off * vmeta_[ti].esz, len * vmeta_[ti].esz);

					}

					buf_off += len;
					gathered_off += len;

				}

			}

			long new_buf_global_start = assigned.empty() ? t.v->buf_global_start : (assigned[0].first - t.v->done_n);
			set_partitioned_layout(*t.v, new_local, t.v->done_n, new_buf_global_start);

			if (!set_read_only_cache_from_ranges(*t.v, assigned, t.v->done_n) &&
				t.v->access_mode != MAL_ACCESS_READ_ONLY) {

				t.v->cache_valid = false;

			}

			t.v->sync_user_ptr();

		}

		if (!assigned.empty()) {

			install_loop_plan(*g.loop, assigned);
			set_limit(*g.loop, g.loop->end);
			set_iter (*g.loop, g.loop->start - 1);

			g.loop->confirmed_iter.store(g.loop->start - 1, std::memory_order_release);
			g.sync.loop_has_new_work.store(true, std::memory_order_release);

		} else {

			freeze_loop_at_current(*g.loop);

		}

	} else {

		auto pa = std::make_unique<PendingActivation>();

		deferred_pending_ranges = std::move(assigned);
		pa->ranges.clear();
		pa->vec_slices.resize(vtasks_.size());

		for (int vi = 0; vi < (int)vtasks_.size(); vi++) {

			auto& t = vtasks_[vi];

			if (new_asgn > 0 && t.gathered.ptr && vmeta_[vi].esz > 0) {

				pa->vec_slices[(size_t)vi] = t.gathered;
				t.gathered = {};

			}

		}

		pa->acc_epoch_bufs = std::move(new_epoch_bufs_);
		g.pending = std::move(pa);
		publish_pending_after_broadcast = true;

	}

	if (target_ > old_a_size_) {

		broadcast_shared_vecs();
		broadcast_shared_mats();

	}

	if (publish_pending_after_broadcast && g.pending) {

		g.pending->ranges = std::move(deferred_pending_ranges);

		if (!g.pending->ranges.empty()) {

			g.sync.pending_has_ranges.store(true, std::memory_order_release);

		}

		g.sync.notify();

	}

}

void Resizer::broadcast_shared_mats() {

	if (MAL_UNLIKELY(g.comm.active == MPI_COMM_NULL || target_ <= old_a_size_)) {

		return;

	}

	int n_shared = (int)g.shared.size();

	MPI_Bcast(&n_shared, 1, MPI_INT, 0, g.comm.active);

	if (n_shared == 0) {

		return;

	}

	const bool is_new = (g.comm.u_rank >= old_a_size_ && g.comm.u_rank < target_);

	std::vector<size_t> tots(n_shared, 0);

	if (!is_new) {

		for (int si = 0; si < n_shared; si++) {

			tots[si] = get_shared_mat_or_abort(si)->total_bytes;

		}

	}

	mpi_bcast_bytes(tots.data(), (size_t)n_shared * sizeof(size_t), 0, g.comm.active);

	constexpr int kSharedMatTagBase = 0x2000;

	auto entry_needed = [](unsigned long long mask, int si) {

		return si >= 64 || ((mask >> si) & 1ull) != 0;

	};

	unsigned long long my_need = 0;

	if (is_new) {

		for (int si = 0; si < n_shared && si < 64; si++) {

			SharedMat* sm = (si < (int)g.shared.size()) ? g.shared[(size_t)si].get() : nullptr;

			if (!(sm && sm->buf && sm->total_bytes == tots[si])) {

				my_need |= (1ull << si);

			}

		}

	}

	std::vector<unsigned long long> all_need((size_t)g.comm.a_size, 0);
	MPI_Allgather(&my_need, 1, MPI_UNSIGNED_LONG_LONG, all_need.data(), 1, MPI_UNSIGNED_LONG_LONG, g.comm.active);

	if (is_new) {

		for (int si = 0; si < n_shared; si++) {

			const size_t tot = tots[si];
			const size_t cap = tot > 0 ? tot : 1;
			SharedMat* sm = (si < (int)g.shared.size()) ? g.shared[(size_t)si].get() : nullptr;

			if (sm && sm->buf && sm->total_bytes == tot) {

				if (tot > 0 && entry_needed(my_need, si)) {

					mpi_recv_bytes(sm->buf, tot, 0, kSharedMatTagBase + si, g.comm.active);

				}

				continue;

			}

			PendingActivation& pa = ensure_pending_activation();
			void* buf = g_buffer_pool.acquire(cap);

			if (tot > 0) {

				mpi_recv_bytes(buf, tot, 0, kSharedMatTagBase + si, g.comm.active);

			}

			pa.shared_mats.push_back({buf, cap});

		}

		return;

	}

	if (g.comm.a_rank != 0) {

		return;

	}

	for (int nr = old_a_size_; nr < target_; nr++) {

		for (int si = 0; si < n_shared; si++) {

			const size_t tot = tots[si];

			if (tot == 0 || !entry_needed(all_need[(size_t)nr], si)) {

				continue;

			}

			mpi_send_bytes(get_shared_mat_or_abort(si)->buf, tot, nr, kSharedMatTagBase + si, g.comm.active);

		}

	}

}

void Resizer::broadcast_shared_vecs() {

	if (MAL_UNLIKELY(g.comm.active == MPI_COMM_NULL || target_ <= old_a_size_)) {

		return;

	}

	struct SharedVecBroadcast {
		int index;
		int bytes;
	};

	std::vector<SharedVecBroadcast> shared_meta;

	if (g.comm.a_rank == 0) {

		shared_meta.reserve(g.vecs.size());

		for (int i = 0; i < (int)g.vecs.size(); i++) {

			MalVec* v = g.vecs[i].get();

			if (!v || (v->attach_policy != MAL_ATTACH_SHARED_ACTIVE && v->attach_policy != MAL_ATTACH_SHARED_ALL)) {

				continue;

			}

			size_t b = (size_t)std::max(0L, v->total_N) * v->elem_size;

			if (MAL_UNLIKELY(b > (size_t)INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector size overflow (%zu bytes)", b);
				MPI_Abort(g.comm.universe, 1);

			}

			shared_meta.push_back({i, (int)b});

		}

	}

	int n = (int)shared_meta.size();
	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.active);

	if (n == 0) {

		return;

	}

	shared_meta.resize(n);
	MPI_Bcast(shared_meta.data(), n * (int)sizeof(SharedVecBroadcast), MPI_BYTE, 0, g.comm.active);

	const bool is_new = (g.comm.u_rank >= old_a_size_ && g.comm.u_rank < target_);
	constexpr int kSharedVecTagBase = 0x1000;

	auto entry_needed = [](unsigned long long mask, int mi) {

		return mi >= 64 || ((mask >> mi) & 1ull) != 0;

	};

	unsigned long long my_need = 0;

	if (is_new) {

		for (int mi = 0; mi < n; mi++) {

			if (mi >= 64) {

				break;

			}

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;
			MalVec* v = (vi >= 0 && vi < (int)g.vecs.size()) ? g.vecs[(size_t)vi].get() : nullptr;

			const bool reusable = v && v->attach_policy == MAL_ATTACH_SHARED_ALL && v->access_mode == MAL_ACCESS_READ_ONLY && v->buf && v->local_n == v->total_N && (size_t)std::max(0, nbytes) <= (size_t)std::max(1L, v->total_N) * v->elem_size;

			if (!reusable) {

				my_need |= (1ull << mi);

			}

		}

	}

	std::vector<unsigned long long> all_need((size_t)g.comm.a_size, 0);
	MPI_Allgather(&my_need, 1, MPI_UNSIGNED_LONG_LONG, all_need.data(), 1, MPI_UNSIGNED_LONG_LONG, g.comm.active);

	if (is_new) {

		for (int mi = 0; mi < n; mi++) {

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;

			MalVec* v = (vi >= 0 && vi < (int)g.vecs.size()) ? g.vecs[(size_t)vi].get() : nullptr;
			const bool have_vec = v && (v->attach_policy == MAL_ATTACH_SHARED_ACTIVE || v->attach_policy == MAL_ATTACH_SHARED_ALL);

			if (have_vec) {

				const size_t buf_need = (size_t)std::max(1L, v->total_N) * v->elem_size;

				if (MAL_UNLIKELY((size_t)std::max(0, nbytes) > buf_need)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared vector size mismatch at index %d (%d > %zu)", vi, nbytes, buf_need);
					MPI_Abort(g.comm.universe, 1);

				}

				configure_shared_active_vec(*v, buf_need);

				if (nbytes > 0 && entry_needed(my_need, mi)) {

					mpi_recv_bytes(v->buf, (size_t)nbytes, 0, kSharedVecTagBase + mi, g.comm.active);

				}

				continue;

			}

			PendingActivation& pa = ensure_pending_activation();
			const size_t cap = nbytes > 0 ? (size_t)nbytes : 1;
			void* buf = g_buffer_pool.acquire(cap);

			if (nbytes > 0) {

				mpi_recv_bytes(buf, (size_t)nbytes, 0, kSharedVecTagBase + mi, g.comm.active);

			}

			pa.shared_vecs.push_back({buf, cap});

		}

		return;

	}

	if (g.comm.a_rank != 0) {

		return;

	}

	for (int nr = old_a_size_; nr < target_; nr++) {

		for (int mi = 0; mi < n; mi++) {

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;

			if (nbytes <= 0 || !entry_needed(all_need[(size_t)nr], mi)) {

				continue;

			}

			if (MAL_UNLIKELY(vi < 0 || vi >= (int)g.vecs.size() || !g.vecs[(size_t)vi])) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector index out of range: %d", vi);
				MPI_Abort(g.comm.universe, 1);

			}

			mpi_send_bytes(g.vecs[(size_t)vi]->buf, (size_t)nbytes, nr, kSharedVecTagBase + mi, g.comm.active);

		}

	}

}

void Resizer::apply_inactive() {

	g.comm.a_rank = -1;
	g.comm.a_size = 0;

	g.lb.epoch_assigned = 0;
	g.lb.epoch_start_time = 0.0;

	for (auto& t : vtasks_) {

		if (!t.v) {

			continue;

		}

		if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE) {

			release_shared_active_vec(*t.v);

			continue;

		}

		if (t.v->attach_policy == MAL_ATTACH_SHARED_ALL) {

			configure_shared_active_vec(*t.v, (size_t)std::max(1L, t.v->total_N) * t.v->elem_size);

			continue;

		}

		refresh_inactive_read_only_cache(*t.v);

		if (t.v->access_mode != MAL_ACCESS_READ_ONLY) {

			size_t buf_need = (size_t)std::max(1L, t.v->done_n) * t.v->elem_size;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);
			t.v->local_n = t.v->done_n;

		}

		t.v->sync_user_ptr();

	}

	if (g.loop) {

		freeze_loop_at_current(*g.loop);

	}

	g.sync.compute_ready.store(true, std::memory_order_release);

}

void Resizer::stash_gather_cache() {

	if (g.gather_cache.size() < vtasks_.size()) {

		g.gather_cache.resize(vtasks_.size());

	}

	for (size_t i = 0; i < vtasks_.size(); i++) {

		auto& t = vtasks_[i];
		auto& c = g.gather_cache[i];

		if (c.ptr) {

			g_buffer_pool.release(c.ptr, c.bytes);

		}

		c = t.gathered;
		t.gathered = {};

	}

}

void Resizer::prepare_phase() {

	const double t0 = MPI_Wtime();
	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Prepare phase start target=%d (active=%d)", target_, g.comm.a_size);

	old_a_size_ = g.comm.a_size;
	MPI_Bcast(&old_a_size_, 1, MPI_INT, 0, g.comm.universe);

	collect_ranges();

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;
 	int naccs = g.loop ? (int)g.loop->accs.size() : 0;

	int header[3] = {nvecs, naccs, 0};
	header[2] = (int)(3 * sizeof(int) + (size_t)nvecs * sizeof(VecMeta));
	int bcast_meta_bytes = header[2];
	std::vector<char> bcast_buf((size_t)bcast_meta_bytes);

	std::memcpy(bcast_buf.data(), header, 3 * sizeof(int));

	if (nvecs > 0) {

		vmeta_.resize(nvecs);

		for (int vi = 0; vi < nvecs; vi++) {

			vmeta_[vi].esz = g.loop->vecs[vi]->elem_size;
			vmeta_[vi].shared_active = (g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ACTIVE || g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ALL) ? 1 : 0;

		}

		std::memcpy(bcast_buf.data() + 3 * sizeof(int), vmeta_.data(), (size_t)nvecs * sizeof(VecMeta));

	}

	MPI_Bcast(header, 3, MPI_INT, 0, g.comm.universe);

	bcast_meta_bytes = header[2];
	bcast_buf.resize((size_t)bcast_meta_bytes);

	if (bcast_meta_bytes > (int)(3 * sizeof(int))) {

		MPI_Bcast(bcast_buf.data() + 3 * sizeof(int), bcast_meta_bytes - (int)(3 * sizeof(int)), MPI_BYTE, 0, g.comm.universe);

	}

	int n_vecs = header[0];
	int n_accs = header[1];

	if (n_vecs > 0) {

		vmeta_.resize(n_vecs);
		std::memcpy(vmeta_.data(), bcast_buf.data() + 3 * sizeof(int), (size_t)n_vecs * sizeof(VecMeta));

	}

	redistribute_vecs(n_vecs);
	reduce_accs(n_accs);

	if (g.comm.u_rank == 0 && !target_cuts_.empty() && total_rem_ > 0) {

		char buf[4096];
		int pos = 0;
		pos += snprintf(buf + pos, (int)sizeof(buf) - pos, "Distribution after resize %d->%d (total=%ld iters):", old_a_size_, target_, total_rem_);

		for (int k = 0; k < target_ && pos < (int)sizeof(buf) - 64; k++) {

			long iters = target_cuts_[(size_t)k + 1] - target_cuts_[(size_t)k];
			double w = k < (int)g.lb.weights.size() ? g.lb.weights[(size_t)k] : 1.0 / target_;
			pos += snprintf(buf + pos, (int)sizeof(buf) - pos, "\n R%-2d: %6ld iters weight=%.4f", k, iters, w);

		}

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "%s", buf);

	}

	const double prepare_elapsed = MPI_Wtime() - t0;
	g.timing.resize_prepare += prepare_elapsed;
	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Prepare phase done target=%d in %.4f s", target_, prepare_elapsed);

}

void Resizer::commit_phase() {

	{
		const double t_wfc = MPI_Wtime();
		g.sync.wait_for_compute();
		g.timing.wait_for_compute += MPI_Wtime() - t_wfc;
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Commit phase start target=%d (current=%d)", target_, g.comm.a_size);

	double t0 = MPI_Wtime();

	const bool same_size_rebalance = (target_ == old_a_size_);

	if (!same_size_rebalance) {

		if (g.comm.active != MPI_COMM_NULL) {

			int rc = MPI_Comm_free(&g.comm.active);

			if (rc != MPI_SUCCESS) {

				char err[MPI_MAX_ERROR_STRING] = {};
				int len = 0;
				MPI_Error_string(rc, err, &len);
				MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_free(active, commit) failed rc=%d msg=%.*s", rc, len, err);

			}

			g.comm.active = MPI_COMM_NULL;

		}

		int color = (g.comm.u_rank < target_) ? 0 : MPI_UNDEFINED;
		int rc = MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);

		if (rc != MPI_SUCCESS) {

			char err[MPI_MAX_ERROR_STRING] = {};
			int len = 0;
			MPI_Error_string(rc, err, &len);
			MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_split(active, commit) failed rc=%d msg=%.*s", rc, len, err);

		}

	}

	if (g.comm.active != MPI_COMM_NULL) {

		int rc = MPI_Comm_set_errhandler(g.comm.active, MPI_ERRORS_RETURN);

		if (rc != MPI_SUCCESS) {

			char err[MPI_MAX_ERROR_STRING] = {};
			int len = 0;
			MPI_Error_string(rc, err, &len);
			MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_set_errhandler(active, commit) failed rc=%d msg=%.*s", rc, len, err);

		}

		MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		MPI_Comm_size(g.comm.active, &g.comm.a_size);
		apply_active();

	} else {

		apply_inactive();

	}

	stash_gather_cache();

	const double commit_elapsed = MPI_Wtime() - t0;
	const double epoch_secs = std::max(kEpsElapsed, g.cfg.epoch_ms.load() / 1000.0);
	const bool fast_resp = g.cfg.fast_response.load(std::memory_order_relaxed);

	const int adaptive_cooldown = fast_resp ? 0 : std::max(0, (int)std::ceil(commit_elapsed / epoch_secs));
	const int min_cooldown = fast_resp ? 0 : kMinResizeCooldownEpochs;
	const int base_resize_cooldown = std::max(adaptive_cooldown, min_cooldown);

	if (same_size_rebalance) {

		g.lb.same_size_rebalance_cooldown = adaptive_cooldown;

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Rebalance on %d active ranks done in %.4f s (cooldown=%d)", target_, commit_elapsed, g.lb.same_size_rebalance_cooldown);

	} else {

		const bool is_oscillation = (g.lb.prev_resize_to > 0 && target_ == g.lb.prev_resize_from && old_a_size_ == g.lb.prev_resize_to);

		g.lb.prev_resize_from = old_a_size_;
		g.lb.prev_resize_to = target_;

		g.lb.resize_cooldown = (is_oscillation && !fast_resp) ? std::max(base_resize_cooldown * 2, 4) : base_resize_cooldown;

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Resize %d->%d done in %.4f s (thr_1=%.1f iters/s, cooldown=%d%s)", old_a_size_, target_, commit_elapsed, g.lb.thr_single_proc, g.lb.resize_cooldown, is_oscillation ? ", oscillation" : "");

	}

	g.timing.resize_commit += commit_elapsed;
	g.timing.resize_count++;

}

ResizeDecision decide_resize_fixed_sequence() {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return out;

	}

	size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);

	if (seq_idx >= g.cfg.sequence.size()) {

		return out;

	}

	int target = g.cfg.sequence[seq_idx];

	if (target <= 0 || target > g.comm.u_size || target == g.comm.a_size) {

		return out;

	}

	out.should_resize = true;
	out.target_active_size = target;

	return out;

}

ResizeDecision decide_resize_auto(const EpochMetrics& m) {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) return out;

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		out.done = true;
		return out;

	}

	if (g.lb.resize_cooldown > 0) {

		g.lb.resize_cooldown--;
		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Resize skipped: resize_cooldown=%d", g.lb.resize_cooldown);
		return out;

	}

	if (g.lb.same_size_rebalance_cooldown > 0) g.lb.same_size_rebalance_cooldown--;

	const int U = g.comm.u_size;

	if (U == 1) return out;

	using Phase = MalState::LoadBalance::Phase;
	const double threshold = get_efficiency_threshold();

	auto compute_efficiency = [&]() -> double {

		const double thr_1 = g.lb.thr_single_proc;

		if (thr_1 <= kEpsThroughput || m.global_thr <= kEpsThroughput || m.active_n <= 0) {

			return -1.0;

		}

		const double speedup = m.global_thr / thr_1;
		return speedup / (double)m.active_n;

	};

	auto log_probe = [&](const char* tag, int probe_n, double eff) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs-%s probe=%d active=%d thr=%.1f thr_1=%.1f speedup=%.2f eff=%.3f threshold=%.2f [lo=%d hi=%d]", tag, probe_n, m.active_n, m.global_thr, g.lb.thr_single_proc, (g.lb.thr_single_proc > kEpsThroughput ? m.global_thr / g.lb.thr_single_proc : 0.0), eff, threshold, g.lb.bs_lo, g.lb.bs_hi);

		}

	};

	switch (g.lb.bs_phase) {

	case Phase::IDLE:

		if (g.lb.thr_single_proc <= kEpsThroughput) {

			g.lb.bs_phase = Phase::NEEDS_BASELINE;
			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: no baseline, going to N=1");

			if (m.active_n != 1) {

				out.should_resize = true;
				out.target_active_size = 1;

			}

			return out;

		}

		g.lb.bs_lo = 1;
		g.lb.bs_hi = U;
		g.lb.bs_phase = Phase::EXPLORE_MAX;

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline thr_1=%.1f, exploring N=%d", g.lb.thr_single_proc, U);

		if (m.active_n != U) {

			out.should_resize = true;
			out.target_active_size = U;

		}

		return out;

	case Phase::NEEDS_BASELINE:


		if (g.lb.bs_baseline_count < kMinBaselineEpochs || g.lb.thr_single_proc <= kEpsThroughput) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline %d/%d epochs thr_1=%.1f (waiting)", g.lb.bs_baseline_count, kMinBaselineEpochs, g.lb.thr_single_proc);
			return out;

		}


		g.lb.bs_lo = 1;
		g.lb.bs_hi = U;
		g.lb.bs_baseline_count = 0;
		g.lb.bs_phase = Phase::EXPLORE_MAX;

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline ready thr_1=%.1f (%d epochs), going to N=%d", g.lb.thr_single_proc, kMinBaselineEpochs, U);

		out.should_resize = true;
		out.target_active_size = U;

		return out;

	case Phase::EXPLORE_MAX:

		if (m.active_n != U) {

			out.should_resize = true;
			out.target_active_size = U;

			return out;

		}

		{

			const double eff = compute_efficiency();
			if (eff < 0.0) return out;

			log_probe("max", U, eff);

			if (eff >= threshold) {

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: N=%d meets threshold (eff=%.3f >= %.2f), entering PROBING", U, eff, threshold);
				g.lb.bs_phase = Phase::PROBING;

				return out;

			}

			g.lb.bs_lo = 1;
			g.lb.bs_hi = U;
			const int probe = (g.lb.bs_lo + g.lb.bs_hi) / 2;
			g.lb.bs_phase = Phase::SEARCHING;
			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: N=%d below threshold (eff=%.3f < %.2f), searching [%d,%d] probe=%d", U, eff, threshold, g.lb.bs_lo, g.lb.bs_hi, probe);

			out.should_resize = true;
			out.target_active_size = probe;

			return out;

		}

	case Phase::SEARCHING:
		{

			const double eff = compute_efficiency();

			if (eff < 0.0) return out;

			log_probe("search", m.active_n, eff);

			if (eff >= threshold) {

				g.lb.bs_lo = m.active_n;

			} else {

				g.lb.bs_hi = m.active_n;

			}

			if (g.lb.bs_hi - g.lb.bs_lo <= 1) {

				const int best = g.lb.bs_lo;
				g.lb.bs_phase = Phase::PROBING;

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: converged, PROBING from N=%d", best);

				if (best != m.active_n) {

					out.should_resize = true;
					out.target_active_size = best;

				}

				return out;

			}

			const int probe = (g.lb.bs_lo + g.lb.bs_hi) / 2;

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: [%d,%d] next probe=%d", g.lb.bs_lo, g.lb.bs_hi, probe);

			out.should_resize = true;
			out.target_active_size = probe;

			return out;

		}

	case Phase::PROBING:
		{

			const double eff = compute_efficiency();
			if (eff < 0.0) {

				return out;

			}

			log_probe("probe", m.active_n, eff);

			if (eff >= threshold) {

				if (m.active_n > g.lb.bs_lo) {

					g.lb.bs_lo = m.active_n;
					MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: improved to bs_lo=%d", g.lb.bs_lo);

				}

				if (m.active_n < U) {

					out.should_resize = true;
					out.target_active_size = m.active_n + 1;

					return out;

				}

				const bool lb_enabled = g.cfg.load_balancing_enabled.load(std::memory_order_relaxed);
				if (lb_enabled && g.lb.same_size_rebalance_cooldown == 0 && m.active_n > 1) {

					const double ideal_w = 1.0 / (double)m.active_n;
					double max_w = 0.0;

					for (int k = 0; k < m.active_n; k++) {

						if (k < (int)g.lb.weights.size()) {

							max_w = std::max(max_w, g.lb.weights[(size_t)k]);

						}

					}

					if (max_w > 1.3 * ideal_w) {

						out.should_resize = true;
						out.target_active_size = m.active_n;

					}

				}

				return out;
			}

			if (m.active_n > g.lb.bs_lo) {

				g.lb.bs_hi = m.active_n;
				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: N=%d failed (eff=%.3f < %.2f), returning to bs_lo=%d", m.active_n, eff, threshold, g.lb.bs_lo);
				out.should_resize = true;
				out.target_active_size = g.lb.bs_lo;

			} else {

				g.lb.bs_hi = m.active_n;
				g.lb.bs_lo = std::max(1, m.active_n / 2);
				g.lb.bs_phase = Phase::SEARCHING;
				const int probe = (g.lb.bs_lo + g.lb.bs_hi) / 2;

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: home N=%d degraded (eff=%.3f < %.2f), re-searching [%d,%d] probe=%d", m.active_n, eff, threshold, g.lb.bs_lo, g.lb.bs_hi, probe);

				if (probe != m.active_n) {

					out.should_resize = true;
					out.target_active_size = probe;

				}

			}

			return out;

		}

	}

	return out;

}

ResizeDecision run_local_resize_decision(const EpochMetrics& m) {

	ResizeDecision decision;

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		decision.done = true;
		return decision;

	}

	switch (g.cfg.resize_policy) {

		case MAL_RESIZE_POLICY_AUTO:
		case MAL_RESIZE_POLICY_THROUGHPUT:
		case MAL_RESIZE_POLICY_EFFICIENCY:
		case MAL_RESIZE_POLICY_COST:

			decision = decide_resize_auto(m);
			break;

		case MAL_RESIZE_POLICY_FIXED_SEQUENCE:

			decision = decide_resize_fixed_sequence();
			break;

		default:

			decision = decide_resize_auto(m);
			break;

	}

	if (!decision.should_resize) {

		decision.target_active_size = -1;
		return decision;

	}

	if (decision.target_active_size <= 0 || decision.target_active_size > g.comm.u_size) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Decision returned invalid target=%d (valid range 1..%d)", decision.target_active_size, g.comm.u_size);
		decision.should_resize = false;
		decision.target_active_size = -1;

	}

	return decision;

}

struct ResizeConsensus {

	bool unanimous{false};
	bool should_resize{false};
	int target{-1};
	int active_size{-1};
	unsigned long long local_decision_epoch{0};

};

ResizeConsensus unanimous_resize_decision() {

	const double t_decision_start = MPI_Wtime();

	ResizeConsensus out;

	EpochMetrics m = gather_epoch_metrics();

	if (!m.any_has_loop) {

		g.timing.epoch_decision += MPI_Wtime() - t_decision_start;
		g.timing.epoch_decision_count++;
		return out;

	}

	const unsigned long long pre_decision_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);
	const bool is_active = (g.comm.active != MPI_COMM_NULL);
	ResizeDecision local_decision = is_active ? run_local_resize_decision(m) : ResizeDecision{};
	const unsigned long long post_decision_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);
	const unsigned long long decision_epoch = std::max(pre_decision_epoch, post_decision_epoch);
	out.local_decision_epoch = decision_epoch;

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Distributed resize evaluation: active=%d universe=%d epoch=%llu", g.comm.a_size, g.comm.u_size, decision_epoch);

	}

	const long long local_should = local_decision.should_resize ? 1LL : 0LL;
	const long long local_target = local_decision.should_resize ? (long long)local_decision.target_active_size : -1LL;
	const long long local_active = (long long)g.comm.a_size;
	const long long local_finalize = g.sync.finalize_requested.load(std::memory_order_acquire) ? 1LL : 0LL;
	const long long local_done = local_decision.done ? 1LL : 0LL;
	const long long any_active_flag = is_active ? 1LL : 0LL;

	const long long send_should = is_active ? local_should : 0LL;
	const long long send_neg_should = is_active ? -local_should : LLONG_MIN / 2;
	const long long send_target = is_active ? local_target : -1LL;
	const long long send_neg_target = is_active ? -local_target : LLONG_MIN / 2;

	long long reduce_in[8] = {send_should, send_target, send_neg_should, send_neg_target, local_active, local_finalize, local_done, any_active_flag};
	long long reduce_out[8] = {};

	MPI_Allreduce(reduce_in, reduce_out, 8, MPI_LONG_LONG, MPI_MAX, g.comm.universe);

	const long long max_should = reduce_out[0];
	const long long max_target = reduce_out[1];
	const long long min_should = -reduce_out[2];
	const long long min_target = -reduce_out[3];
	const long long any_finalize = reduce_out[5];
	const long long any_done = reduce_out[6];
	const bool any_active_voted = (reduce_out[7] != 0);

	if (any_finalize || any_done) {

		g.sync.stop.store(true, std::memory_order_release);
		g.sync.notify();
		out.unanimous = true;
		out.should_resize = false;
		out.target = -1;
		out.active_size = (int)reduce_out[4];
		return out;

	}

	out.unanimous = any_active_voted && (min_should == max_should) && (min_should == 0 || min_target == max_target);
	out.should_resize = out.unanimous && min_should != 0;
	out.target = out.should_resize ? (int)min_target : -1;
	out.active_size = (int)reduce_out[4];

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Consensus from distributed eval: unanimous=%d should=%d target=%d", (int)out.unanimous, (int)out.should_resize, out.target);

	}

	g.timing.epoch_decision += MPI_Wtime() - t_decision_start;
	g.timing.epoch_decision_count++;

	return out;

}

void advance_default_sequence_after_commit() {

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return;

	}

	size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);

	if (seq_idx < g.cfg.sequence.size()) {

		seq_idx++;
		g.cfg.seq_idx.store(seq_idx, std::memory_order_relaxed);

	}

	if (seq_idx >= g.cfg.sequence.size()) {

		g.cfg.enabled.store(false, std::memory_order_relaxed);

	}

}

static bool prepare_with_consistent_snapshot(Resizer& resizer) {

	if (g.sync.stop.load(std::memory_order_acquire)) {

		return false;

	}

	resizer.prepare_phase();
	return true;

}

bool prepare_resize_if_needed() {

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		return false;

	}

	if (g.prepared_resize_ready.load(std::memory_order_acquire)) {

		return false;

	}

	{

		std::lock_guard lk(g.resize_mu);

		if (g.prepared_resize.ready()) {

			return false;

		}

	}

	ResizeConsensus consensus = unanimous_resize_decision();
	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Consensus: should=%d target=%d active=%d", (int)consensus.should_resize, consensus.target, consensus.active_size);

	const bool seq_policy = (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE);

	if (seq_policy && !consensus.unanimous) {

		advance_default_sequence_after_commit();
		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Sequence divergence detected (non-unanimous decision); advancing sequence index to seek next common point");

		return false;

	}

	if (seq_policy && consensus.unanimous && !consensus.should_resize) {

		size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);
		bool skipped = g.cfg.enabled.load(std::memory_order_relaxed) && seq_idx < g.cfg.sequence.size() && g.cfg.sequence[seq_idx] == consensus.active_size;

		if (skipped) {

			advance_default_sequence_after_commit();

			MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Skipping no-op resize target=%d", consensus.active_size);

		}

		return false;

	}

	if (!consensus.unanimous || !consensus.should_resize) {

		return false;

	}

	auto prepared = std::make_unique<Resizer>(consensus.target);

	if (!prepare_with_consistent_snapshot(*prepared)) {

		return false;

	}

	{

		std::lock_guard lk(g.resize_mu);

		if (g.prepared_resize.ready()) {

			g.sync.resize_pending.store(false, std::memory_order_release);
			g.sync.notify();
			return false;

		}

		g.prepared_resize.target = consensus.target;
		g.prepared_resize.local_decision_epoch = consensus.local_decision_epoch;
		g.prepared_resize.work = std::move(prepared);
		g.prepared_resize_ready.store(true, std::memory_order_release);

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Prepared resize candidate target=%d", consensus.target);

	return true;

}

void clear_prepared_resize() {

	std::lock_guard lk(g.resize_mu);
	g.prepared_resize.reset();
	g.prepared_resize_ready.store(false, std::memory_order_release);

}

bool commit_prepared_resize_if_ready() {

	if (!g.prepared_resize_ready.load(std::memory_order_acquire)) {

		return false;

	}

	int prep_target = -1;
	unsigned long long prep_local_epoch = 0;
	std::unique_ptr<Resizer> prepared_work;

	{

		std::lock_guard lk(g.resize_mu);

		if (!g.prepared_resize.ready()) {

			g.prepared_resize_ready.store(false, std::memory_order_release);
			g.sync.resize_pending.store(false, std::memory_order_release);
			g.sync.notify();

			return false;

		}

		prep_target = g.prepared_resize.target;
		prep_local_epoch = g.prepared_resize.local_decision_epoch;
		prepared_work = std::move(g.prepared_resize.work);
		g.prepared_resize.target = -1;
		g.prepared_resize.local_decision_epoch = 0;
		g.prepared_resize_ready.store(false, std::memory_order_release);

	}

	unsigned long long epoch_now = g.sync.compute_epoch.load(std::memory_order_acquire);
	int local_changed = (epoch_now > prep_local_epoch) ? 1 : 0;
	int any_changed = 0;

	MPI_Allreduce(&local_changed, &any_changed, 1, MPI_INT, MPI_MAX, g.comm.universe);

	if (any_changed != 0) {

		const int mode = g.cfg.epoch_change_mode.load(std::memory_order_relaxed);

		if (mode == MAL_EPOCH_CHANGE_USE_LAST_DECISION) {

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Epoch changed (prep_epoch=%llu, current_epoch=%llu) — STALE decision, but mode=1 (USE_LAST_DECISION): reusing old decision target=%d without recalculating", prep_local_epoch, epoch_now, prep_target);

		} else {

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Epoch changed (prep_epoch=%llu, current_epoch=%llu) — STALE decision, mode=0 (RECALCULATE): discarding old target=%d and recalculating", prep_local_epoch, epoch_now, prep_target);

			prepared_work.reset();

			ResizeConsensus refreshed = unanimous_resize_decision();

			if (!refreshed.unanimous || !refreshed.should_resize) {

				MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Recalculated decision: no valid resize needed (old_target=%d) — resize aborted, releasing compute", prep_target);
				g.sync.resize_pending.store(false, std::memory_order_release);
				g.sync.notify();
				return false;

			}

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Recalculated decision: new_target=%d (old_target=%d)", refreshed.target, prep_target);

			prepared_work = std::make_unique<Resizer>(refreshed.target);

			if (!prepare_with_consistent_snapshot(*prepared_work)) {

				return false;

			}

			prep_target = refreshed.target;
			prep_local_epoch = refreshed.local_decision_epoch;

		}

	} else {

		MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Epoch unchanged (prep_epoch=%llu == current_epoch=%llu) — FRESH decision: using target=%d as-is", prep_local_epoch, epoch_now, prep_target);

	}

	g.sync.resize_pending.store(true, std::memory_order_release);
	g.sync.notify();

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Committing resize target=%d", prep_target);

	prepared_work->commit_phase();

	g.sync.resize_pending.store(false, std::memory_order_release);
	clear_prepared_resize();

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		advance_default_sequence_after_commit();

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Commit complete (active=%d)", g.comm.a_size);

	g.sync.notify();

	return true;

}

inline int effective_epoch_interval_ms() {

	const int wait_ms = g.cfg.epoch_ms.load(std::memory_order_relaxed);
	return wait_ms > 0 ? wait_ms : kDefaultEpochIntervalMs;

}

void progress_thread() {

	#ifdef __APPLE__

		if (g.cfg.affinity_enabled) {

			#if defined(__arm64__) || defined(__aarch64__)

				pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "worker: QoS set to E-Core");

			#elif defined(__x86_64__) || defined(__i386__)

				mach_port_t self = mach_thread_self();
				thread_affinity_policy_data_t policy = { 1 };
				thread_policy_set(self, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
				mach_port_deallocate(mach_task_self(), self);
				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "worker: affinity hint set to E-Core");

			#endif

		}

	#endif

	std::vector<std::function<void()>> batch;
	int epoch_backoff_multiplier = 1;
	constexpr int kMaxBackoffMultiplier = 8;
	const bool needs_initial_rampup = g.comm.a_size > 0 && g.comm.a_size < g.comm.u_size && g.cfg.resize_policy != MAL_RESIZE_POLICY_FIXED_SEQUENCE;
	auto next_resize_check = std::chrono::steady_clock::now() + std::chrono::milliseconds(needs_initial_rampup ? 1 : effective_epoch_interval_ms());

	while (!g.sync.stop.load(std::memory_order_relaxed)) {

		for (;;) {

			{

				std::lock_guard lk(g.attach_mu);

				if (g.attach_tasks.empty()) {

					g.sync.attach_pending.store(false, std::memory_order_release);
					g.sync.notify();
					break;

				}

				batch.swap(g.attach_tasks);

			}

			for (auto& fn : batch) {

				if (fn) fn();

			}

			batch.clear();

		}

		const int epoch_snapshot_ms = effective_epoch_interval_ms();

		{

			std::unique_lock lk(g.sync.mu);

			for (;;) {

				const bool should_wake =
					g.sync.stop.load(std::memory_order_relaxed) ||
					g.sync.finalize_requested.load(std::memory_order_relaxed) ||
					g.sync.attach_pending.load(std::memory_order_relaxed) ||
					g.sync.compute_ready.load(std::memory_order_relaxed) ||
					g.prepared_resize_ready.load(std::memory_order_relaxed) ||
					effective_epoch_interval_ms() != epoch_snapshot_ms;

				if (should_wake) {

					break;

				}

				if (g.sync.cv.wait_until(lk, next_resize_check) == std::cv_status::timeout) {

					break;

				}

			}

		}

		if (g.sync.stop.load(std::memory_order_relaxed)) {

			break;

		}

		if (g.sync.attach_pending.load(std::memory_order_acquire)) {

			continue;

		}

		const int epoch_ms = effective_epoch_interval_ms();

		if (epoch_ms != epoch_snapshot_ms) {

			next_resize_check = std::chrono::steady_clock::now() + std::chrono::milliseconds(epoch_ms);
			continue;

		}

		const bool finalize_now = g.sync.finalize_requested.load(std::memory_order_acquire);
		const bool compute_ready = g.sync.compute_ready.load(std::memory_order_acquire);
		const bool prepared_ready = g.prepared_resize_ready.load(std::memory_order_acquire);
		const auto now = std::chrono::steady_clock::now();

		if (!finalize_now && !compute_ready && !prepared_ready && now < next_resize_check) {

			continue;

		}

		const bool should_try_prepare = finalize_now || compute_ready || now >= next_resize_check;

		if (should_try_prepare) {

			const bool did_prepare = prepare_resize_if_needed();

			const bool fast_resp = g.cfg.fast_response.load(std::memory_order_relaxed);

			if (did_prepare) {

				epoch_backoff_multiplier = 1;

			} else if (needs_initial_rampup && g.timing.resize_count == 0) {

				epoch_backoff_multiplier = 1;

			} else if (fast_resp) {

				epoch_backoff_multiplier = 1;

			} else if (epoch_backoff_multiplier < kMaxBackoffMultiplier) {

				epoch_backoff_multiplier = std::min(epoch_backoff_multiplier * 2, kMaxBackoffMultiplier);

			}

			next_resize_check = now + std::chrono::milliseconds(epoch_ms * epoch_backoff_multiplier);

		}

		if (g.sync.stop.load(std::memory_order_acquire)) {

			break;

		}

		if (g.prepared_resize_ready.load(std::memory_order_acquire)) {

			commit_prepared_resize_if_ready();
			epoch_backoff_multiplier = 1;

		}

		g.sync.notify();

	}

	g.sync.notify();

}

#endif
