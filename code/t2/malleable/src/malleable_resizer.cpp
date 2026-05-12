#ifndef MALLEABLE_RESIZER_CPP_INCLUDED
#define MALLEABLE_RESIZER_CPP_INCLUDED

#include "malleable_types.cpp"
#include "malleable_papi.cpp"

constexpr double kEpsElapsed = 1e-6;
constexpr double kEpsThroughput = 1e-9;
constexpr double kEpsWeight = 1e-12;
constexpr double kEpsDone = 1.0;
constexpr int kLbGatherFields = 5;

inline const double* lb_row_at(const std::vector<double>& all_lb_buf, int k) {

	return &all_lb_buf[(size_t)k * (size_t)kLbGatherFields];

}

inline double lb_row_tp(const double* row) {

	return (row[0] > 0.0 && row[1] > kEpsElapsed) ? (row[0] / row[1]) : 0.0;

}

inline double lb_row_ee(const double* row) {

	return (row[2] > kEpsWeight) ? (1.0 / row[2]) : 0.0;

}

inline bool reuse_flag_at(const std::vector<int>& all_reuse_flags, int n, int rank, int vi) {

	return all_reuse_flags[(size_t)rank * (size_t)n + (size_t)vi] != 0;

}

class Resizer {

	std::vector<std::pair<long,long>> remaining_;
	std::vector<long> remaining_offsets_;
	std::vector<long> target_cuts_;
	long total_rem_{0};

	// gathered: pool buffer with received slice. Owned by VecTask until either
	// (a) consumed in apply_active() (memcpy then released by ~Resizer), or
	// (b) moved to PendingActivation::vec_slices in waiting_for_activation path
	// (then released by ~PendingActivation). After move, t.gathered = {} prevents double-free.
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

	void init_vec_meta(int nvecs, int n);
	std::vector<TransferPlanEntry> build_transfer_plan(const std::vector<long>& old_vs) const;
	void init_vec_tasks(int n, int nvecs, bool was_active);
	void reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) ;
	void exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags);

	void collect_ranges();
	void redistribute_vecs();
	void reduce_accs();
	void apply_active();
	void apply_inactive();
	void broadcast_shared_vecs();
	void broadcast_shared_mats();
	void stash_gather_cache();

	int target_;
	int old_a_size_{0};
	long my_new_vs_{0};
	long my_new_count_{0};

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

struct AutoResizeMetrics {

	static constexpr int kIdxLocalRem = 0;
	static constexpr int kIdxThr = 1;
	static constexpr int kIdxDataBytes = 2;
	static constexpr int kIdxActiveN = 3;
	static constexpr int kIdxMemBound = 4;
	static constexpr int kIdxEnergyEff = 5;
	static constexpr int kDecFields = 6;
	double local_rem{0.0};
	double my_thr{0.0};
	double my_data_bytes{0.0};
	double my_mem_bound_local{0.0};
	double my_energy_eff{0.0};
	std::vector<double> per_rank_all;
	std::vector<long> all_local_rem;
	double global_remaining{0.0};
	double global_throughput_inst{0.0};
	double max_data_bytes{0.0};
	double global_energy_eff_mass{0.0};
	double sum_mem_bound_w{0.0};
	double sum_thr_w{0.0};
	int active_n{0};
	double mem_bound_fresh{0.0};
	double global_throughput{0.0};

};

double energy_mix_from_metrics(double mem_bound_fresh, double throughput_mass, double energy_mass, double ipc_ewma, double ipc_peak_ref) {

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_THROUGHPUT) {

		return 0.0;

	}

	const double throughput_pressure = std::clamp(throughput_mass / (throughput_mass + energy_mass + kEpsWeight), 0.0, 1.0);
	const double ipc_pressure = std::clamp(1.0 - (ipc_peak_ref > kEpsWeight ? ipc_ewma / ipc_peak_ref : 0.0), 0.0, 1.0);
	const double energy_pressure = std::clamp(mem_bound_fresh + ipc_pressure - mem_bound_fresh * ipc_pressure, 0.0, 1.0);
	return energy_pressure / (energy_pressure + throughput_pressure + kEpsWeight);

}

struct ResizeCandidateEval {

	int target_n{-1};
	double T_next{0.0};
	double transfer_cost{0.0};
	double net_gain{0.0};
	double rel_gain{0.0};
	double score{0.0};
	double predicted_cost{0.0};
	double cost_rel_gain{0.0};
	bool worthwhile{false};

};

void log_top_resize_candidates(const std::vector<ResizeCandidateEval>& all_candidates, int current_n) {

	if (g.comm.u_rank != 0 || all_candidates.empty()) {

		return;

	}

	const int top_n = std::min(3, (int)all_candidates.size());
	std::array<int, 3> top_idx{-1, -1, -1};

	for (int i = 0; i < (int)all_candidates.size(); i++) {

		const double s = all_candidates[(size_t)i].score;

		for (int t = 0; t < top_n; t++) {

			if (top_idx[t] < 0 || s > all_candidates[(size_t)top_idx[t]].score) {

				for (int u = top_n - 1; u > t; u--) top_idx[u] = top_idx[u - 1];
				top_idx[t] = i;
				break;

			}

		}

	}

	for (int t = 0; t < top_n; t++) {

		if (top_idx[t] < 0) break;

		const auto& c = all_candidates[(size_t)top_idx[t]];
		const char* mode = (c.target_n == current_n) ? "rebalance" : ((c.target_n > current_n) ? "scale-up" : "scale-down");

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Top candidate #%d: target=%d mode=%s score=%.4f rel=%.2f%% gain=%.4fs cost=%.4fs C(p)=%.4fps cost_rel=%.2f%%", t + 1, c.target_n, mode, c.score, c.rel_gain * 100.0, c.net_gain, c.transfer_cost, c.predicted_cost, c.cost_rel_gain * 100.0);

	}

}

inline void push_candidate_target(std::vector<int>& out, std::vector<char>& seen, int target) {

	if (target < 1 || target >= (int)seen.size()) {

		return;

	}

	if (seen[(size_t)target]) {

		return;

	}

	seen[(size_t)target] = 1;
	out.push_back(target);

}

AutoResizeMetrics gather_auto_resize_metrics() {

	AutoResizeMetrics m;
	m.per_rank_all.assign((size_t)g.comm.u_size * AutoResizeMetrics::kDecFields, 0.0);
	m.all_local_rem.assign((size_t)g.comm.u_size, 0);

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		long cur = *g.loop->user_iter;

		if (cur + 1 < g.loop->end) {

			m.local_rem += g.loop->end - (cur + 1);

		}

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			m.local_rem += g.loop->plan_ranges[ri].second - g.loop->plan_ranges[ri].first;

		}

	}

	const double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	const long my_done = std::max(0L, g.lb.epoch_assigned - (long)m.local_rem);
	m.my_thr = (my_elapsed > kEpsElapsed && my_done > 0) ? (double)my_done / my_elapsed : 0.0;

	if (g.loop) {

		for (MalVec* v : g.loop->vecs) {

			if (v && v->attach_policy == MAL_ATTACH_PARTITIONED) {

				const long rem_local = std::max(0L, v->local_n - v->done_n);
				m.my_data_bytes += (double)rem_local * (double)v->elem_size;

			}

		}

	}

	m.my_mem_bound_local = papi_mem_bound_fraction(g.lb.papi_prev_vals);
	const double my_energy_per_iter = papi_energy_per_iter(g.lb.papi_prev_vals, my_done);
	m.my_energy_eff = (my_energy_per_iter > kEpsWeight) ? (1.0 / my_energy_per_iter) : 0.0;
	double per_rank_send[AutoResizeMetrics::kDecFields] = { m.local_rem, m.my_thr, m.my_data_bytes, (double)g.comm.a_size, m.my_mem_bound_local, m.my_energy_eff };
	MPI_Allgather(per_rank_send, AutoResizeMetrics::kDecFields, MPI_DOUBLE, m.per_rank_all.data(), AutoResizeMetrics::kDecFields, MPI_DOUBLE, g.comm.universe);

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = &m.per_rank_all[(size_t)k * AutoResizeMetrics::kDecFields];
		m.all_local_rem[(size_t)k] = (long)std::llround(row[AutoResizeMetrics::kIdxLocalRem]);
		m.global_remaining += row[AutoResizeMetrics::kIdxLocalRem];
		const double thr_k = row[AutoResizeMetrics::kIdxThr];
		m.global_throughput_inst += thr_k;
		m.max_data_bytes = std::max(m.max_data_bytes, row[AutoResizeMetrics::kIdxDataBytes]);
		m.active_n = std::max(m.active_n, (int)std::lround(row[AutoResizeMetrics::kIdxActiveN]));
		m.global_energy_eff_mass += std::max(0.0, row[AutoResizeMetrics::kIdxEnergyEff]);

		if (thr_k > kEpsWeight) {

			m.sum_mem_bound_w += row[AutoResizeMetrics::kIdxMemBound] * thr_k;
			m.sum_thr_w += thr_k;

		}

	}

	m.mem_bound_fresh = (m.sum_thr_w > kEpsWeight) ? std::clamp(m.sum_mem_bound_w / m.sum_thr_w, 0.0, 1.0) : std::clamp(g.lb.global_mem_bound, 0.0, 1.0);

	if (m.active_n > 0) {

		std::lock_guard<std::mutex> lk(g.lb.weights_mu);

		if ((int)g.lb.weights.size() < g.comm.u_size) {

			g.lb.weights.assign((size_t)g.comm.u_size, 0.0);
			const double init_w = 1.0 / (double)m.active_n;

			for (int k = 0; k < m.active_n; k++) {

				g.lb.weights[(size_t)k] = init_w;

			}

		}

		double thr_mass_active = 0.0;

		for (int k = 0; k < m.active_n; k++) {

			const double thr_k = std::max(0.0, m.per_rank_all[(size_t)k * AutoResizeMetrics::kDecFields + AutoResizeMetrics::kIdxThr]);
			thr_mass_active += thr_k;

		}

		if (thr_mass_active > kEpsWeight) {

			const double w_alpha = std::clamp(g.lb.alpha, kEpsElapsed, 1.0);

			for (int k = 0; k < g.comm.u_size; k++) {

				const double target_w =
					(k < m.active_n)
					? (std::max(0.0, m.per_rank_all[(size_t)k * AutoResizeMetrics::kDecFields + AutoResizeMetrics::kIdxThr]) / thr_mass_active)
					: 0.0;

				g.lb.weights[(size_t)k] = w_alpha * target_w + (1.0 - w_alpha) * g.lb.weights[(size_t)k];

			}

			double wsum = 0.0;

			for (int k = 0; k < m.active_n; k++) {

				wsum += std::max(0.0, g.lb.weights[(size_t)k]);

			}

			if (wsum > kEpsWeight) {

				for (int k = 0; k < m.active_n; k++) {

					g.lb.weights[(size_t)k] = std::max(0.0, g.lb.weights[(size_t)k]) / wsum;

				}

			} else {

				const double uniform_w = 1.0 / (double)m.active_n;

				for (int k = 0; k < m.active_n; k++) {

					g.lb.weights[(size_t)k] = uniform_w;

				}

			}

			for (int k = m.active_n; k < g.comm.u_size; k++) {

				g.lb.weights[(size_t)k] = 0.0;

			}

		}

	}

	const double thr_ewma_alpha = std::clamp(g.cfg.auto_thr_ewma_alpha, kEpsElapsed, 1.0);

	if (m.global_throughput_inst > kEpsWeight) {

		if (g.lb.auto_thr_ewma <= kEpsWeight) {

			g.lb.auto_thr_ewma = m.global_throughput_inst;

		} else {

			g.lb.auto_thr_ewma = thr_ewma_alpha * m.global_throughput_inst + (1.0 - thr_ewma_alpha) * g.lb.auto_thr_ewma;

		}

	}

	m.global_throughput = (g.lb.auto_thr_ewma > kEpsWeight) ? g.lb.auto_thr_ewma : m.global_throughput_inst;

	return m;

}

void Resizer::collect_ranges() {

	std::vector<long> local;
	local.reserve(16);

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		const long first_s = *g.loop->user_iter + 1;
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

	flat_counts_buf_.resize(g.comm.u_size);

	MPI_Allgather(&my_count, 1, MPI_INT, flat_counts_buf_.data(), 1, MPI_INT, g.comm.universe);

	flat_displs_buf_ = make_displs(flat_counts_buf_);

	int total = flat_displs_buf_.back() + flat_counts_buf_.back();

	if (total == 0) {

		remaining_.clear();
		remaining_offsets_.assign(1, 0);
		total_rem_ = 0;
		rem_per_rank_.assign(g.comm.u_size, 0);
		g.sync.stop = true;

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

	if (total_rem_ == 0) {

		g.sync.stop.store(true, std::memory_order_release);

	}

	double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	long my_done = std::max(0L, g.lb.epoch_assigned - rem_per_rank_[g.comm.u_rank]);

	double lbdata[kLbGatherFields] = {(double)my_done, my_elapsed, papi_energy_per_iter(g.lb.papi_prev_vals, my_done), papi_mem_bound_fraction(g.lb.papi_prev_vals), papi_ipc(g.lb.papi_prev_vals)};
	all_lb_buf_.resize((size_t)g.comm.u_size * (size_t)kLbGatherFields);

	MPI_Allgather(lbdata, kLbGatherFields, MPI_DOUBLE, all_lb_buf_.data(), kLbGatherFields, MPI_DOUBLE, g.comm.universe);

	double total_tp = 0.0;
	double total_ee = 0.0;

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = lb_row_at(all_lb_buf_, k);
		total_tp += lb_row_tp(row);
		total_ee += lb_row_ee(row);

	}

	if (total_tp > 0.0) {

		std::lock_guard<std::mutex> lk_w(g.lb.weights_mu);
		const double thr_ewma_alpha_cr = std::clamp(g.cfg.auto_thr_ewma_alpha, kEpsElapsed, 1.0);

		if (g.lb.auto_thr_ewma <= kEpsWeight) {

			g.lb.auto_thr_ewma = total_tp;

		} else {

			g.lb.auto_thr_ewma = thr_ewma_alpha_cr * total_tp + (1.0 - thr_ewma_alpha_cr) * g.lb.auto_thr_ewma;

		}

		if (target_ > old_a_size_ && target_ > 0) {

			if ((int)g.lb.weights.size() < g.comm.u_size) {

				g.lb.weights.assign((size_t)g.comm.u_size, 0.0);

			}

			double done_total = 0.0;

			for (int k = 0; k < target_; k++) {

				done_total += std::max(0.0, lb_row_at(all_lb_buf_, k)[0]);

			}

			const double target_work = (done_total + (double)total_rem_) / (double)target_;
			double quota_sum = 0.0;

			for (int k = 0; k < g.comm.u_size; k++) {

				double w_boot = 0.0;

				if (k < target_) {

					const double done_k = std::max(0.0, lb_row_at(all_lb_buf_, k)[0]);
					w_boot = std::max(0.0, target_work - done_k);
					quota_sum += w_boot;

				}

				g.lb.weights[(size_t)k] = w_boot;

			}

			if (quota_sum > kEpsWeight) {

				for (int k = 0; k < target_; k++) {

					g.lb.weights[(size_t)k] /= quota_sum;

				}

			} else {

				const double uniform_w = 1.0 / (double)target_;

				for (int k = 0; k < target_; k++) {

					g.lb.weights[(size_t)k] = uniform_w;

				}

			}

		} else {

			if ((int)g.lb.weights.size() < g.comm.u_size) {

				double fill;

				if (g.lb.weights.empty()) {

					fill = 1.0 / g.comm.u_size;

				} else {

					double sum = 0.0;

					for (double wk : g.lb.weights) sum += wk;

					fill = sum / (double)g.lb.weights.size();

				}

				g.lb.weights.resize(g.comm.u_size, fill);

			}

			const double energy_coeff = energy_mix_from_metrics(g.lb.global_mem_bound, total_tp, total_ee, g.lb.global_ipc_ewma, std::max({kEpsWeight, g.lb.ipc_peak_ref, g.lb.global_ipc_ewma}));
			const bool use_energy = (energy_coeff > kEpsWeight && total_ee > kEpsWeight && papi_is_available());

			for (int k = 0; k < g.comm.u_size; k++) {

				const double* row = lb_row_at(all_lb_buf_, k);
				const double tp_k = lb_row_tp(row);

				if (tp_k > 0.0) {

					const double norm_tp = tp_k / total_tp;
					double blended = norm_tp;

					if (use_energy) {

						const double ee_k = lb_row_ee(row);

						if (ee_k > 0.0) {

							blended = (1.0 - energy_coeff) * norm_tp + energy_coeff * (ee_k / total_ee);

						}

					}

					g.lb.weights[k] = g.lb.alpha * blended + (1.0 - g.lb.alpha) * g.lb.weights[k];

				}

			}

		}

		double mem_bound_sum = 0.0;
		double ipc_sum = 0.0;
		double papi_w_sum = 0.0;

		for (int k = 0; k < g.comm.u_size; k++) {

			const double* row = lb_row_at(all_lb_buf_, k);
			const double tp_k = lb_row_tp(row);
			const double mb = row[3];
			const double ipc = row[4];

			if (tp_k > kEpsWeight && (mb > kEpsWeight || ipc > kEpsWeight)) {

				mem_bound_sum += mb * tp_k;
				ipc_sum += ipc * tp_k;
				papi_w_sum += tp_k;

			}

		}

		if (papi_w_sum > kEpsWeight) {

			g.lb.global_mem_bound = std::clamp(mem_bound_sum / papi_w_sum, 0.0, 1.0);
			const double new_ipc = ipc_sum / papi_w_sum;
			const double thr_alpha = std::clamp(g.cfg.auto_thr_ewma_alpha, kEpsElapsed, 1.0);

			if (g.lb.ipc_peak_ref <= kEpsWeight) {

				g.lb.ipc_peak_ref = new_ipc;

			} else {

				g.lb.ipc_peak_ref = std::max(new_ipc, thr_alpha * new_ipc + (1.0 - thr_alpha) * g.lb.ipc_peak_ref);

			}

			if (g.lb.global_ipc_ewma > kEpsWeight && new_ipc > kEpsWeight) {

				const double rel_drop = (g.lb.global_ipc_ewma - new_ipc) / g.lb.global_ipc_ewma;
				const double rel_drop_abs = std::fabs(rel_drop);
				g.lb.ipc_drop_ewma = thr_alpha * rel_drop_abs + (1.0 - thr_alpha) * g.lb.ipc_drop_ewma;
				const double drop_trigger = std::max(g.lb.ipc_drop_ewma, g.lb.auto_alpha_est_sec / (g.lb.auto_alpha_est_sec + std::max(my_elapsed, kEpsElapsed)));

				if (rel_drop > drop_trigger) {

					MAL_LOG_L(MAL_LOG_DEBUG, "LB", "Phase change detected: IPC %.2f to %.2f (drop=%.1f%%) — decaying weights", g.lb.global_ipc_ewma, new_ipc, rel_drop * 100.0);

					const double decay = 1.0 - std::min(0.9, rel_drop);

					for (auto& wk : g.lb.weights) {

						wk *= decay;

					}

				}

			}

			if (g.lb.global_ipc_ewma <= kEpsWeight) {

				g.lb.global_ipc_ewma = new_ipc;

			} else {

				g.lb.global_ipc_ewma = thr_alpha * new_ipc + (1.0 - thr_alpha) * g.lb.global_ipc_ewma;

			}

		}

		const double* my_row = lb_row_at(all_lb_buf_, g.comm.u_rank);
		const double my_mem_bound = my_row[3];
		const double my_ipc = my_row[4];
		MAL_LOG(MAL_LOG_DEBUG, "LB: epoch done=%ld elapsed=%.3fs thr=%.1f iters/s ipc=%.2f mem_bound=%.2f weight=%.4f", my_done, my_elapsed, my_done > 0 && my_elapsed > kEpsElapsed ? (double)my_done / my_elapsed : 0.0, my_ipc, my_mem_bound, g.comm.u_rank < (int)g.lb.weights.size() ? g.lb.weights[g.comm.u_rank] : 0.0);

	}

}

ResizeCandidateEval evaluate_resize_candidate(const AutoResizeMetrics& m, int current_n, int target_n, const std::vector<double>& rank_weight_fill, const std::vector<double>& rank_weight_prefix, bool active_weights_ok, double sum_w_active, double global_throughput, double global_remaining, double max_data_bytes, double bw, double sync_cost_base, double T_current, double mem_bound, double non_llc_stall, double energy_mix) {

	ResizeCandidateEval eval;
	eval.target_n = target_n;
	const bool lb_enabled = g.cfg.load_balancing_enabled.load(std::memory_order_relaxed);

	if (current_n <= 0 || target_n <= 0) {

		return eval;

	}

	if (!lb_enabled && target_n == current_n) {

		return eval;

	}

	const double target_sync_cost = sync_cost_base * std::log2(std::max(2.0, (double)target_n));
	const int compare_n = std::max(current_n, target_n);
	const double migration_cost_est = (bw > 0.0) ? max_data_bytes * (double)std::abs(target_n - current_n) / (double)compare_n / bw : 0.0;
	const double migration_pressure = std::clamp((target_sync_cost + migration_cost_est) / (T_current + target_sync_cost + migration_cost_est + kEpsWeight), 0.0, 1.0);
	const double utility_energy_mix = (g.cfg.resize_policy == MAL_RESIZE_POLICY_THROUGHPUT) ? 0.0 : ((g.cfg.resize_policy == MAL_RESIZE_POLICY_ENERGY) ? 1.0 : std::clamp(energy_mix, 0.0, 1.0));
	double utility_throughput = 1.0 - utility_energy_mix;
	double utility_energy = utility_energy_mix;
	double utility_migration = std::clamp(migration_pressure, 0.0, 1.0) * std::max(kEpsWeight, g.lb.migration_aversion);

	const double utility_norm = utility_throughput + utility_energy + utility_migration;

	if (utility_norm > kEpsWeight) {

		utility_throughput /= utility_norm;
		utility_energy /= utility_norm;
		utility_migration /= utility_norm;

	}

	const double roofline_exponent = std::clamp(energy_mix, 0.0, 1.0);

	if (target_n == current_n && g.lb.same_size_rebalance_cooldown > 0) {

		return eval;

	}

	double next_time = T_current;
	double transfer_cost = target_sync_cost + migration_cost_est;

	if (target_n == current_n) {

		double T_bal = 0.0;
		double moved_iters_l1 = 0.0;

		for (int k = 0; k < current_n; k++) {

			const long rk = m.all_local_rem[(size_t)k];
			const double share = (active_weights_ok && sum_w_active > kEpsThroughput) ? (rank_weight_fill[(size_t)k] / sum_w_active) : (1.0 / (double)current_n);
			const double speed = std::max(kEpsWeight, global_throughput * share);
			const double expected = global_remaining * share;

			T_bal = std::max(T_bal, expected / speed);
			moved_iters_l1 += std::fabs((double)rk - expected);

		}

		const double moved_iters = 0.5 * moved_iters_l1;
		const double moved_bytes = (global_remaining > kEpsThroughput) ? (max_data_bytes * moved_iters / global_remaining) : 0.0;
		next_time = T_bal;
		transfer_cost = target_sync_cost + ((bw > 0.0) ? (moved_bytes / bw) : 0.0);

	} else {

		const double candidate_weight_mass = rank_weight_prefix[(size_t)target_n];
		const double speedup_ideal = (active_weights_ok && sum_w_active > kEpsThroughput) ? candidate_weight_mass / sum_w_active : (double)target_n / (double)current_n;
		const double mb_target = mem_bound * std::pow((double)current_n / (double)target_n, roofline_exponent);
		const double eff_stall = std::min(1.0, mb_target + non_llc_stall);
		const double speedup = 1.0 + (speedup_ideal - 1.0) * std::max(0.0, 1.0 - eff_stall);

		next_time = (speedup > kEpsWeight) ? T_current / speedup : T_current;

		if (active_weights_ok && sum_w_active > kEpsWeight) {

			double bottleneck = 0.0;
			const int load_n = std::min(current_n, target_n);
			const double rem_share = global_remaining / (double)target_n;

			for (int k = 0; k < load_n; k++) {

				const double speed_k = global_throughput * rank_weight_fill[(size_t)k] / sum_w_active;

				if (speed_k <= kEpsWeight) {

					continue;

				}

				const double mb_k = std::clamp(m.per_rank_all[(size_t)k * AutoResizeMetrics::kDecFields + AutoResizeMetrics::kIdxMemBound], 0.0, 1.0);
				const double mb_k_target = mb_k * std::pow((double)current_n / (double)target_n, roofline_exponent);
				const double eff_k = std::min(1.0, mb_k_target + non_llc_stall);
				const double speedup_k = 1.0 + (speedup_ideal - 1.0) * std::max(0.0, 1.0 - eff_k);
				const double speed_k_target = speed_k * std::max(1.0, speedup_k);

				bottleneck = std::max(bottleneck, rem_share / speed_k_target);

			}

			if (bottleneck > kEpsElapsed) {

				next_time = bottleneck;

			}

		}

	}

	const double current_energy_proxy = std::max((double)current_n * T_current, kEpsThroughput);
	const double next_energy_proxy = std::max((double)target_n * next_time, kEpsThroughput);
	const double throughput_gain = std::max(0.0, (T_current - next_time) / std::max(T_current, kEpsThroughput));
	const double energy_gain = std::max(0.0, (current_energy_proxy - next_energy_proxy) / current_energy_proxy);
	const double transfer_norm = transfer_cost / std::max(T_current, kEpsThroughput);
	const double phase_factor = T_current / (T_current + transfer_cost + kEpsThroughput);

	eval.T_next = next_time;
	eval.transfer_cost = transfer_cost;
	eval.net_gain = T_current - (eval.T_next + eval.transfer_cost);
	eval.rel_gain = (T_current > kEpsThroughput) ? (eval.net_gain / T_current) : 0.0;
	eval.score = phase_factor * (utility_throughput * throughput_gain + utility_energy * energy_gain - utility_migration * transfer_norm);

	// Cost model C(p) = p * T_par(p). Add transfer_cost into prediction so policy
	// pays the migration overhead like real wall time would.
	const double current_cost = std::max(kEpsThroughput, (double)current_n * T_current);
	eval.predicted_cost = (double)target_n * (eval.T_next + eval.transfer_cost);
	eval.cost_rel_gain = (current_cost - eval.predicted_cost) / current_cost;

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_COST) {

		// Replace utility-based score with cost reduction ratio. Cap migration penalty
		// implicitly because predicted_cost already includes transfer_cost.
		eval.score = phase_factor * eval.cost_rel_gain;
		eval.worthwhile = eval.cost_rel_gain > 0.0 && eval.net_gain > -transfer_cost;

	} else {

		eval.worthwhile = eval.score > 0.0 && eval.net_gain > 0.0;

	}

	return eval;


}

void maybe_report_realized_benefit(const AutoResizeMetrics& m) {

	if (!g.lb.post_eval_pending) {

		return;

	}

	const double elapsed = std::max(kEpsElapsed, MPI_Wtime() - g.lb.post_eval_decision_time);
	const double baseline_thr = std::max(kEpsThroughput, g.lb.post_eval_baseline_thr);
	const double baseline_remaining = std::max(0.0, g.lb.post_eval_baseline_remaining);

	const double actual_progress = std::max(0.0, baseline_remaining - m.global_remaining);
	const double expected_progress_no_change = baseline_thr * elapsed;
	const double extra_progress = actual_progress - expected_progress_no_change;
	const double realized_gain_sec = extra_progress / std::max(m.global_throughput_inst, kEpsThroughput);
	const double realized_thr_gain = (m.global_throughput_inst - baseline_thr) / baseline_thr;

	const double lr = std::clamp(g.cfg.auto_calibration_alpha, kEpsElapsed, 1.0);
	const double pred_gain = g.lb.post_eval_pred_net_gain;
	const double pred_abs = std::max(std::fabs(pred_gain), kEpsElapsed);
	const double trust_sample = std::clamp(0.5 + 0.5 * realized_gain_sec / pred_abs, 0.0, 1.0);

	g.lb.benefit_trust = lr * trust_sample + (1.0 - lr) * g.lb.benefit_trust;

	const double predicted_positive = std::max(0.0, pred_gain);
	const double realized_positive = std::max(0.0, realized_gain_sec);
	const double rel_error = (predicted_positive - realized_positive) / (predicted_positive + realized_positive + kEpsElapsed);
	const double aversion_step = std::exp(lr * rel_error);

	g.lb.migration_aversion = std::clamp(g.lb.migration_aversion * aversion_step, kEpsWeight, 5.0);

	MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Post-commit eval %d->%d: pred(net=%.4fs rel=%.2f%% score=%.4f) realized(thr=%.2f%% gain=%.4fs progress=%.2f/%.2f iters elapsed=%.3fs trust=%.3f mig_aversion=%.3f)", g.lb.post_eval_from_n, g.lb.post_eval_to_n, g.lb.post_eval_pred_net_gain, g.lb.post_eval_pred_rel_gain * 100.0, g.lb.post_eval_pred_score, realized_thr_gain * 100.0, realized_gain_sec, actual_progress, expected_progress_no_change, elapsed, g.lb.benefit_trust, g.lb.migration_aversion);

	g.lb.post_eval_pending = false;

}

void Resizer::init_vec_meta(int nvecs, int n) {

	vmeta_.resize(n);

	for (int vi = 0; vi < nvecs; vi++) {

		vmeta_[vi].esz = g.loop->vecs[vi]->elem_size;
		vmeta_[vi].shared_active = (g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ACTIVE) ? 1 : 0;

	}

	MPI_Bcast(vmeta_.data(), n * (int)sizeof(VecMeta), MPI_BYTE, 0, g.comm.universe);

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

			new_done = std::clamp(*g.loop->user_iter - t.v->buf_global_start + 1, 0L, t.v->local_n);

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

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

				continue;

			}

			packed_bytes += (size_t)tr.v_count * vmeta_[vi].esz;

		}

		if (packed_bytes == 0) {

			continue;

		}

		if (packed_bytes > (size_t)INT_MAX) {

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

void Resizer::redistribute_vecs() {

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;
	int n = nvecs;

	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.universe);

	if (n == 0) {

		return;

	}

	init_vec_meta(nvecs, n);

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
	exchange_vec_data(n, was_active, old_vs, plan, scratch_all_reuse_flags_);

}

void Resizer::reduce_accs() {

	int naccs = g.loop ? (int)g.loop->accs.size() : 0;
	int n = naccs;

	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.universe);

	if (n == 0) {

		return;

	}

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
			a->fn_reset(a->ptr);

		}

	};

	batched_allreduce(n, AccGetter{naccs, &loop_accs}, AccResultSetter{naccs, &loop_accs, &new_epoch_bufs_});

}

void Resizer::apply_active() {

	const std::vector<long>& active_cuts = (g.comm.a_size == target_ && target_cuts_.size() == (size_t)g.comm.a_size + 1) ? target_cuts_ : (target_cuts_ = build_partition_cuts(total_rem_, g.comm.a_size));
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

			if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE) {

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
			set_iter (*g.loop, g.loop->start);
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

	if (is_new) {

		PendingActivation& pa = ensure_pending_activation();
		pa.shared_mats.reserve(pa.shared_mats.size() + (size_t)n_shared);

		for (int si = 0; si < n_shared; si++) {

			const size_t tot = tots[si];
			const size_t cap = tot > 0 ? tot : 1;
			void* buf = g_buffer_pool.acquire(cap);

			mpi_bcast_bytes(buf, tot, 0, g.comm.active);
			pa.shared_mats.push_back({buf, cap});

		}

		return;

	}

	for (int si = 0; si < n_shared; si++) {

		const size_t tot = tots[si];
		const size_t cap = tot > 0 ? tot : 1;
		SharedMat* sm = get_shared_mat_or_abort(si);

		if (!sm->buf || sm->total_bytes != tot) {

			if (!sm->user_owned && sm->buf) {

				g_buffer_pool.release(sm->buf, sm->total_bytes > 0 ? sm->total_bytes : 1);

			}

			sm->buf = g_buffer_pool.acquire(cap);
			sm->user_owned = false;

		}

		sm->total_bytes = tot;

		if (sm->user_ptr) {

			*sm->user_ptr = sm->buf;

		}

		mpi_bcast_bytes(sm->buf, tot, 0, g.comm.active);

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

			if (!v || v->attach_policy != MAL_ATTACH_SHARED_ACTIVE) {

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

	for (const auto& meta : shared_meta) {

		int vi = meta.index;
		int nbytes = meta.bytes;

		if (MAL_UNLIKELY(vi < 0 || vi >= (int)g.vecs.size())) {

			MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector index out of range: %d", vi);
			MPI_Abort(g.comm.universe, 1);

		}

		MalVec* v = g.vecs[vi].get();

		if (MAL_UNLIKELY(!v)) {

			MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector missing at index %d", vi);
			MPI_Abort(g.comm.universe, 1);

		}

		configure_shared_active_vec(*v, (size_t)std::max(1L, v->total_N) * v->elem_size);

		if (nbytes > 0) {

			mpi_bcast_bytes(v->buf, (size_t)nbytes, 0, g.comm.active);

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

	collect_ranges();
	redistribute_vecs();
	reduce_accs();

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Prepare phase done target=%d in %.4f s", target_, MPI_Wtime() - t0);

}

void Resizer::commit_phase() {

	g.sync.wait_for_compute();

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Commit phase start target=%d (current=%d)", target_, g.comm.a_size);

	double t0 = MPI_Wtime();
	const bool same_size_rebalance = (target_ == g.comm.a_size);

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
	const int adaptive_cooldown = std::max(0, (int)std::ceil(commit_elapsed / epoch_secs));

	if (same_size_rebalance) {

		g.lb.same_size_rebalance_cooldown = adaptive_cooldown;

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Rebalance on %d active ranks done in %.4f s (cooldown=%d)", target_, commit_elapsed, g.lb.same_size_rebalance_cooldown);

	} else {

		if (old_a_size_ > 0 && target_ > 0 && g.lb.auto_thr_ewma > kEpsWeight) {

			g.lb.auto_thr_ewma *= (double)target_ / (double)old_a_size_;

		}

		MPI_Bcast(&g.lb.auto_thr_ewma, 1, MPI_DOUBLE, 0, g.comm.universe);

		g.lb.resize_cooldown = adaptive_cooldown;

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Resize %d to %d done in %.4f s (thr_ewma=%.1f iters/s synced, cooldown=%d)", old_a_size_, target_, commit_elapsed, g.lb.auto_thr_ewma, g.lb.resize_cooldown);

	}

}

ResizeDecisionContext make_resize_decision_context() {

	ResizeDecisionContext ctx;

	ctx.universe_size = g.comm.u_size;
	ctx.active_size = g.comm.a_size;
	ctx.compute_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);

 	return ctx;

}

ResizeDecision decide_resize_fixed_sequence(const ResizeDecisionContext& ctx) {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return out;

	}

	size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);

	if (seq_idx >= g.cfg.sequence.size()) {

		return out;

	}

	int target = g.cfg.sequence[seq_idx];

	if (target <= 0 || target > ctx.universe_size || target == ctx.active_size) {

		return out;

	}

	out.should_resize = true;
	out.target_active_size = target;

	return out;

}

ResizeDecision decide_resize_auto(const ResizeDecisionContext& ctx) {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return out;

	}

	AutoResizeMetrics m = gather_auto_resize_metrics();
	maybe_report_realized_benefit(m);

	if (g.lb.same_size_rebalance_cooldown > 0) g.lb.same_size_rebalance_cooldown--;

	if (g.lb.resize_cooldown > 0) {

		g.lb.resize_cooldown--;
		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Resize skipped: resize_cooldown=%d", g.lb.resize_cooldown);
		return out;

	}

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		g.sync.stop.store(true, std::memory_order_release);
		g.sync.notify();

		return out;

	}

	if (m.global_throughput < kEpsThroughput) {

		return out;

	}

	const int U = ctx.universe_size;
	const double bw = (g.lb.auto_bw_est_bps > kEpsWeight) ? g.lb.auto_bw_est_bps : 0.0;
	const double epoch_secs = g.cfg.epoch_ms.load() / 1000.0;
	const double base_sync = std::max(std::max(0.0, g.lb.auto_alpha_est_sec + g.lb.sync_wait_est_sec), epoch_secs * g.cfg.auto_sync_overhead_frac);

	const auto& w = g.lb.weights;
	const bool active_weights_ok = ((int)w.size() >= m.active_n);
	double sum_w_active = active_weights_ok ? 0.0 : ((double)m.active_n / (double)U);

	if (active_weights_ok) {

		for (int k = 0; k < m.active_n; k++) {

			sum_w_active += w[k];

		}

	}

	double T_current = m.global_remaining / m.global_throughput;

	if (active_weights_ok && sum_w_active > kEpsThroughput && m.global_throughput > kEpsWeight) {

		for (int k = 0; k < m.active_n; k++) {

			const double speed_k = m.global_throughput * w[k] / sum_w_active;

			if (speed_k > kEpsWeight) {

				T_current = std::max(T_current, (double)m.all_local_rem[(size_t)k] / speed_k);

			}

		}

	}

	const double mem_bound = m.mem_bound_fresh;
	const double ipc_peak = std::max({kEpsWeight, g.lb.ipc_peak_ref, g.lb.global_ipc_ewma});
	const double energy_mix = energy_mix_from_metrics(m.mem_bound_fresh, m.global_throughput_inst, m.global_energy_eff_mass, g.lb.global_ipc_ewma, ipc_peak);
	const double ipc_total_stall = (g.lb.global_ipc_ewma > kEpsWeight) ? std::clamp(1.0 - g.lb.global_ipc_ewma / ipc_peak, 0.0, 1.0) : 0.0;
	const double non_llc_stall = std::max(0.0, ipc_total_stall - mem_bound);
	const int denom = std::max(1, m.active_n);
	std::vector<double> rank_weight_fill((size_t)std::max(0, U), 1.0 / (double)denom);

	if (active_weights_ok) {

		for (int i = 0; i < U; i++) {

			rank_weight_fill[(size_t)i] = (i < (int)w.size()) ? w[(size_t)i] : (1.0 / (double)denom);

		}

	}

	std::vector<double> rank_weight_prefix((size_t)std::max(0, U) + 1, 0.0);

	for (int i = 0; i < U; i++) {

		rank_weight_prefix[(size_t)i + 1] = rank_weight_prefix[(size_t)i] + rank_weight_fill[(size_t)i];

	}

	std::vector<int> candidate_targets;

	if (U <= 16) {

		candidate_targets.reserve((size_t)U);

		for (int target_n = 1; target_n <= U; target_n++) {

			candidate_targets.push_back(target_n);

		}

	} else {

		std::vector<char> seen((size_t)U + 1, 0);
		candidate_targets.reserve((size_t)std::min(U, 16));

		push_candidate_target(candidate_targets, seen, 1);
		push_candidate_target(candidate_targets, seen, U);
		push_candidate_target(candidate_targets, seen, m.active_n);

		const int radius = std::max(0, g.cfg.auto_candidate_radius);

		for (int d = 1; d <= radius; d++) {

			push_candidate_target(candidate_targets, seen, m.active_n - d);
			push_candidate_target(candidate_targets, seen, m.active_n + d);

		}

		for (int p2 = 1; p2 <= U; p2 <<= 1) {

			push_candidate_target(candidate_targets, seen, p2);

			if (p2 > U / 2) {

				break;

			}

		}

		const int stride = std::max(1, g.cfg.auto_candidate_stride);

		if (stride > 1) {

			for (int target_n = stride; target_n < U; target_n += stride) {

				push_candidate_target(candidate_targets, seen, target_n);

			}

		}

		std::sort(candidate_targets.begin(), candidate_targets.end());

	}

	std::vector<ResizeCandidateEval> all_candidates;
	all_candidates.reserve(candidate_targets.size());

	for (int target_n : candidate_targets) {

		all_candidates.push_back(evaluate_resize_candidate(m, m.active_n, target_n, rank_weight_fill, rank_weight_prefix, active_weights_ok, sum_w_active, m.global_throughput, m.global_remaining, m.max_data_bytes, bw, base_sync, T_current, mem_bound, non_llc_stall, energy_mix));

	}

	log_top_resize_candidates(all_candidates, m.active_n);

	ResizeCandidateEval selected;
	selected.score = 0.0;

	for (const auto& candidate : all_candidates) {

		if (!candidate.worthwhile || candidate.score <= selected.score) {

			continue;

		}

		selected = candidate;

	}

	const double transfer_pressure = std::clamp((base_sync + g.lb.sync_wait_est_sec) / (T_current + base_sync + g.lb.sync_wait_est_sec + kEpsWeight), 0.0, 1.0);
	const double thr_noise = std::clamp(std::fabs(m.global_throughput_inst - m.global_throughput) / std::max(m.global_throughput, kEpsWeight), 0.0, 1.0);
	const double trust_guard_cap = std::max(1.0, g.cfg.auto_trust_guard_cap);
	const double raw_trust_guard = (g.lb.benefit_trust > kEpsWeight) ? (1.0 / g.lb.benefit_trust) : trust_guard_cap;
	const double trust_guard = std::clamp(raw_trust_guard, 0.0, trust_guard_cap);
	const double min_rel_gain_model = (base_sync + g.lb.auto_alpha_est_sec) / (T_current + base_sync + g.lb.auto_alpha_est_sec + kEpsWeight) * trust_guard;
	const double rel_gain_floor = std::clamp(g.cfg.auto_min_rel_gain_floor, 0.0, 1.0);
	const double rel_gain_cap = std::clamp(std::max(rel_gain_floor, g.cfg.auto_min_rel_gain_cap), rel_gain_floor, 1.0);
	const double min_rel_gain = std::clamp(min_rel_gain_model, rel_gain_floor, rel_gain_cap);
	const double gain_margin = 1.0 + transfer_pressure + thr_noise;

	const ResizeCandidateEval* rebalance = nullptr;

	for (const auto& candidate : all_candidates) {

		if (candidate.target_n == m.active_n) {

			rebalance = &candidate;
			break;

		}

	}

	const bool rebalance_passes_rel_gain = rebalance && rebalance->worthwhile && rebalance->rel_gain >= min_rel_gain;

	if (selected.worthwhile && selected.target_n != m.active_n && rebalance_passes_rel_gain) {

		const double required_score = rebalance->score * gain_margin;

		if (selected.score < required_score) {

			if (g.comm.u_rank == 0) {

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Scale candidate downgraded to same-size rebalance (scale_score=%.4f < rebalance_score=%.4f * margin=%.2f)", selected.score, rebalance->score, gain_margin);

			}
			selected = *rebalance;

		}

	}

	if (!selected.worthwhile || selected.score <= 0.0 || selected.target_n <= 0) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "No resize candidate clears the score bar: active=%d rem=%.0f T=%.3fs mem=%.2f ipc=%.2f", m.active_n, m.global_remaining, T_current, mem_bound, g.lb.global_ipc_ewma);

		}
		return out;

	}

	const bool selected_passes_rel_gain = selected.worthwhile && selected.rel_gain >= min_rel_gain;

	if (!selected_passes_rel_gain) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Candidate rejected by min rel gain: target=%d rel=%.2f%% < min=%.2f%%", selected.target_n, selected.rel_gain * 100.0, min_rel_gain * 100.0);

		}
		return out;

	}

	out.should_resize = true;
	out.target_active_size = selected.target_n;
	out.post_eval_valid = true;
	out.post_eval_from_n = m.active_n;
	out.post_eval_to_n = selected.target_n;
	out.post_eval_decision_time = MPI_Wtime();
	out.post_eval_pred_net_gain = selected.net_gain;
	out.post_eval_pred_rel_gain = selected.rel_gain;
	out.post_eval_pred_score = selected.score;
	out.post_eval_baseline_thr = m.global_throughput_inst;
	out.post_eval_baseline_remaining = m.global_remaining;

	if (selected.target_n == m.active_n) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Same-size rebalance: active=%d T_curr=%.3fs T_new=%.3fs gain=%.3fs cost=%.3fs score=%.3fs rel=%.2f%%", m.active_n, T_current, selected.T_next, selected.net_gain, selected.transfer_cost, selected.score, selected.rel_gain * 100.0);

		}

	} else if (selected.target_n < m.active_n) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Scale down: rem=%.0f T_curr=%.3fs T_new=%.3fs gain=%.3fs cost=%.3fs score=%.3fs to target=%d", m.global_remaining, T_current, selected.T_next, selected.net_gain, selected.transfer_cost, selected.score, selected.target_n);

		}

	} else {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Scale up: rem=%.0f T_curr=%.3fs T_new=%.3fs gain=%.3fs cost=%.3fs score=%.3fs to target=%d", m.global_remaining, T_current, selected.T_next, selected.net_gain, selected.transfer_cost, selected.score, selected.target_n);

		}

	}

	return out;

}

ResizeDecision run_local_resize_decision(ResizeDecisionContext& ctx) {

	ResizeDecision decision;

	switch (g.cfg.resize_policy) {

		case MAL_RESIZE_POLICY_AUTO:
		case MAL_RESIZE_POLICY_THROUGHPUT:
		case MAL_RESIZE_POLICY_ENERGY:
		case MAL_RESIZE_POLICY_COST:
			decision = decide_resize_auto(ctx);
			break;

		case MAL_RESIZE_POLICY_FIXED_SEQUENCE:
			decision = decide_resize_fixed_sequence(ctx);
			break;

		default:
			decision = decide_resize_auto(ctx);
			break;

	}

	if (!decision.should_resize) {

		decision.target_active_size = -1;
		decision.post_eval_valid = false;
		return decision;

	}

	if (decision.target_active_size <= 0 || decision.target_active_size > g.comm.u_size) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Decision returned invalid target=%d (valid range 1..%d)", decision.target_active_size, g.comm.u_size);
		decision.should_resize = false;
		decision.target_active_size = -1;
		decision.post_eval_valid = false;

	}

	return decision;

}

struct ResizeConsensus {

	bool unanimous{false};
	bool should_resize{false};
	int target{-1};
	int active_size{-1};
	unsigned long long decision_epoch{0};
	unsigned long long local_decision_epoch{0};
	ResizeDecision local_decision{};

};

ResizeConsensus unanimous_resize_decision() {

	ResizeConsensus out;
	ResizeDecisionContext local_ctx = make_resize_decision_context();
	const bool is_active = (g.comm.active != MPI_COMM_NULL);
	ResizeDecision local_decision = is_active ? run_local_resize_decision(local_ctx) : ResizeDecision{};
	out.local_decision = local_decision;

	const unsigned long long post_decision_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);
	out.local_decision_epoch = std::max(local_ctx.compute_epoch, post_decision_epoch);
	local_ctx.compute_epoch = out.local_decision_epoch;

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Distributed resize evaluation: active=%d universe=%d epoch=%llu", local_ctx.active_size, local_ctx.universe_size, local_ctx.compute_epoch);

	}

	const long long any_active_flag = is_active ? 1LL : 0LL;
	const long long local_should = local_decision.should_resize ? 1LL : 0LL;
	const long long local_target = local_decision.should_resize ? (long long)local_decision.target_active_size : -1LL;
	const long long local_active = (long long)local_ctx.active_size;
	const long long local_epoch = (long long)local_ctx.compute_epoch;

	const long long send_should = is_active ? local_should : 0LL;
	const long long send_neg_should = is_active ? -local_should : LLONG_MIN / 2;
	const long long send_target = is_active ? local_target : -1LL;
	const long long send_neg_target = is_active ? -local_target : LLONG_MIN / 2;

	long long reduce_in[7] = { send_should, send_target, send_neg_should, send_neg_target, local_active, local_epoch, any_active_flag};
	long long reduce_out[7] = {0, 0, 0, 0, 0, 0, 0};

	MPI_Allreduce(reduce_in, reduce_out, 7, MPI_LONG_LONG, MPI_MAX, g.comm.universe);

	const long long max_should = reduce_out[0];
	const long long max_target = reduce_out[1];
	const long long min_should = -reduce_out[2];
	const long long min_target = -reduce_out[3];
	const bool any_active_voted = (reduce_out[6] != 0);

	out.unanimous = any_active_voted && (min_should == max_should) && (min_should == 0 || min_target == max_target);
	out.should_resize = out.unanimous && min_should != 0;
	out.target = out.should_resize ? (int)min_target : -1;
	out.active_size = (int)reduce_out[4];
	out.decision_epoch = (unsigned long long)reduce_out[5];

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Consensus from distributed eval: unanimous=%d should=%d target=%d", (int)out.unanimous, (int)out.should_resize, out.target);

	}

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

bool prepare_resize_if_needed() {

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

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
	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Consensus: should=%d target=%d active=%d epoch=%llu", (int)consensus.should_resize, consensus.target, consensus.active_size, consensus.decision_epoch);

	const bool seq_policy = (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE);

	if (seq_policy && !consensus.unanimous) {

		advance_default_sequence_after_commit();
		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Sequence divergence detected (non-unanimous decision); advancing sequence index to seek next common point");

		return false;

	}

	if (seq_policy && consensus.unanimous && !consensus.should_resize) {

		size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);
		bool skipped = g.cfg.enabled.load(std::memory_order_relaxed) &&
			seq_idx < g.cfg.sequence.size() &&
			g.cfg.sequence[seq_idx] == consensus.active_size;

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
	prepared->prepare_phase();

	{

		std::lock_guard lk(g.resize_mu);

		if (g.prepared_resize.ready()) {

			return false;

		}

		g.prepared_resize.target = consensus.target;
		g.prepared_resize.decision_epoch = consensus.decision_epoch;
		g.prepared_resize.local_decision_epoch = consensus.local_decision_epoch;
		g.prepared_resize.post_eval_valid = consensus.local_decision.post_eval_valid;
		g.prepared_resize.post_eval_from_n = consensus.local_decision.post_eval_from_n;
		g.prepared_resize.post_eval_to_n = consensus.local_decision.post_eval_to_n;
		g.prepared_resize.post_eval_decision_time = consensus.local_decision.post_eval_decision_time;
		g.prepared_resize.post_eval_pred_net_gain = consensus.local_decision.post_eval_pred_net_gain;
		g.prepared_resize.post_eval_pred_rel_gain = consensus.local_decision.post_eval_pred_rel_gain;
		g.prepared_resize.post_eval_pred_score = consensus.local_decision.post_eval_pred_score;
		g.prepared_resize.post_eval_baseline_thr = consensus.local_decision.post_eval_baseline_thr;
		g.prepared_resize.post_eval_baseline_remaining = consensus.local_decision.post_eval_baseline_remaining;

		g.prepared_resize.work = std::move(prepared);
		g.prepared_resize_ready.store(true, std::memory_order_release);

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Prepared resize candidate target=%d (epoch=%llu)", consensus.target, consensus.decision_epoch);

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
	bool prep_post_eval_valid = false;
	int prep_post_eval_from_n = 0;
	int prep_post_eval_to_n = 0;
	double prep_post_eval_decision_time = 0.0;
	double prep_post_eval_pred_net_gain = 0.0;
	double prep_post_eval_pred_rel_gain = 0.0;
	double prep_post_eval_pred_score = 0.0;
	double prep_post_eval_baseline_thr = 0.0;
	double prep_post_eval_baseline_remaining = 0.0;
	std::unique_ptr<Resizer> prepared_work;

	{

		std::lock_guard lk(g.resize_mu);

		if (!g.prepared_resize.ready()) {

			g.prepared_resize_ready.store(false, std::memory_order_release);

			return false;

		}

		prep_target = g.prepared_resize.target;
		prep_local_epoch = g.prepared_resize.local_decision_epoch;
		prep_post_eval_valid = g.prepared_resize.post_eval_valid;
		prep_post_eval_from_n = g.prepared_resize.post_eval_from_n;
		prep_post_eval_to_n = g.prepared_resize.post_eval_to_n;
		prep_post_eval_decision_time = g.prepared_resize.post_eval_decision_time;
		prep_post_eval_pred_net_gain = g.prepared_resize.post_eval_pred_net_gain;
		prep_post_eval_pred_rel_gain = g.prepared_resize.post_eval_pred_rel_gain;
		prep_post_eval_pred_score = g.prepared_resize.post_eval_pred_score;
		prep_post_eval_baseline_thr = g.prepared_resize.post_eval_baseline_thr;
		prep_post_eval_baseline_remaining = g.prepared_resize.post_eval_baseline_remaining;
		prepared_work = std::move(g.prepared_resize.work);
		g.prepared_resize.target = -1;
		g.prepared_resize.decision_epoch = 0;
		g.prepared_resize.local_decision_epoch = 0;
		g.prepared_resize.post_eval_valid = false;
		g.prepared_resize_ready.store(false, std::memory_order_release);

	}

	unsigned long long epoch_now = g.sync.compute_epoch.load(std::memory_order_acquire);
	int local_changed = (epoch_now > prep_local_epoch) ? 1 : 0;
	int any_changed = 0;

	MPI_Allreduce(&local_changed, &any_changed, 1, MPI_INT, MPI_MAX, g.comm.universe);

	if (any_changed != 0) {

		const int mode = g.cfg.epoch_change_mode.load(std::memory_order_relaxed);

		if (mode == MAL_EPOCH_CHANGE_USE_LAST_DECISION) {

			MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Epoch changed; reusing prepared decision target=%d with existing data (mode=1)", prep_target);

		} else {

			MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Epoch changed; discarding prepared data and recalculating (mode=0)");

			prepared_work.reset();

			ResizeConsensus refreshed = unanimous_resize_decision();

			if (!refreshed.unanimous || !refreshed.should_resize) {

				MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Recalculated decision: no valid resize needed (old_target=%d)", prep_target);
				return false;

			}

			prepared_work = std::make_unique<Resizer>(refreshed.target);
			prepared_work->prepare_phase();

			prep_target = refreshed.target;
			prep_local_epoch = refreshed.local_decision_epoch;
			prep_post_eval_valid = refreshed.local_decision.post_eval_valid;
			prep_post_eval_from_n = refreshed.local_decision.post_eval_from_n;
			prep_post_eval_to_n = refreshed.local_decision.post_eval_to_n;
			prep_post_eval_decision_time = refreshed.local_decision.post_eval_decision_time;
			prep_post_eval_pred_net_gain = refreshed.local_decision.post_eval_pred_net_gain;
			prep_post_eval_pred_rel_gain = refreshed.local_decision.post_eval_pred_rel_gain;
			prep_post_eval_pred_score = refreshed.local_decision.post_eval_pred_score;
			prep_post_eval_baseline_thr = refreshed.local_decision.post_eval_baseline_thr;
			prep_post_eval_baseline_remaining = refreshed.local_decision.post_eval_baseline_remaining;

		}

	}

	double my_data_bytes = 0.0;

	if (g.loop) {

		for (MalVec* v : g.loop->vecs) {

			if (v && v->attach_policy == MAL_ATTACH_PARTITIONED) {

				const long rem_local = std::max(0L, v->local_n - v->done_n);
				my_data_bytes += (double)rem_local * (double)v->elem_size;

			}

		}

	}

	double max_data_bytes = 0.0;
	MPI_Allreduce(&my_data_bytes, &max_data_bytes, 1, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	const int old_n = std::max(1, g.comm.a_size);
	const int new_n = std::max(1, prep_target);
	const int denom_n = std::max(old_n, new_n);
	const double model_moved_bytes = max_data_bytes * (double)std::abs(new_n - old_n) / (double)denom_n;
	const double model_logp = std::log2(std::max(2.0, (double)new_n));

	g.sync.resize_pending.store(true, std::memory_order_release);
	g.sync.notify();

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Committing resize target=%d", prep_target);

	const double commit_t0 = MPI_Wtime();
	prepared_work->commit_phase();
	const double commit_elapsed_local = MPI_Wtime() - commit_t0;

	g.lb.post_eval_pending = prep_post_eval_valid;

	if (prep_post_eval_valid) {

		g.lb.post_eval_from_n = prep_post_eval_from_n;
		g.lb.post_eval_to_n = prep_post_eval_to_n;
		g.lb.post_eval_decision_time = prep_post_eval_decision_time;
		g.lb.post_eval_pred_net_gain = prep_post_eval_pred_net_gain;
		g.lb.post_eval_pred_rel_gain = prep_post_eval_pred_rel_gain;
		g.lb.post_eval_pred_score = prep_post_eval_pred_score;
		g.lb.post_eval_baseline_thr = prep_post_eval_baseline_thr;
		g.lb.post_eval_baseline_remaining = prep_post_eval_baseline_remaining;

	}

	double commit_elapsed = 0.0;
	MPI_Allreduce(&commit_elapsed_local, &commit_elapsed, 1, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	if (model_logp > kEpsThroughput) {

		const double cal_alpha = std::clamp(g.cfg.auto_calibration_alpha, kEpsElapsed, 1.0);
		const double bw_ref = (g.lb.auto_bw_est_bps > kEpsWeight) ? g.lb.auto_bw_est_bps : g.cfg.auto_bandwidth_bps;
		const double data_term = (bw_ref > kEpsWeight) ? (model_moved_bytes / bw_ref) : 0.0;

		double alpha_sample = (commit_elapsed - data_term) / model_logp;
		if (!std::isfinite(alpha_sample) || alpha_sample < 0.0) {

			alpha_sample = 0.0;

		}

		g.lb.auto_alpha_est_sec = cal_alpha * alpha_sample + (1.0 - cal_alpha) * g.lb.auto_alpha_est_sec;

		if (model_moved_bytes > kEpsElapsed && commit_elapsed > kEpsThroughput) {

			double beta_sample = (commit_elapsed - g.lb.auto_alpha_est_sec * model_logp) / model_moved_bytes;

			if (std::isfinite(beta_sample) && beta_sample > 0.0) {

				const double bw_sample = 1.0 / beta_sample;
				g.lb.auto_bw_est_bps = cal_alpha * bw_sample + (1.0 - cal_alpha) * g.lb.auto_bw_est_bps;

			}

		}

		const double model_comm_est = g.lb.auto_alpha_est_sec * model_logp + ((g.lb.auto_bw_est_bps > kEpsWeight) ? (model_moved_bytes / g.lb.auto_bw_est_bps) : 0.0);
		double sync_wait_sample = commit_elapsed - model_comm_est;

		if (!std::isfinite(sync_wait_sample) || sync_wait_sample < 0.0) {

			sync_wait_sample = 0.0;

		}

		g.lb.sync_wait_est_sec = cal_alpha * sync_wait_sample + (1.0 - cal_alpha) * g.lb.sync_wait_est_sec;

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Calibrated comm model: alpha=%.6fs bw=%.3e B/s sync_wait=%.6fs (elapsed=%.4fs moved=%.3eB logp=%.3f)", g.lb.auto_alpha_est_sec, g.lb.auto_bw_est_bps, g.lb.sync_wait_est_sec, commit_elapsed, model_moved_bytes, model_logp);

	}

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
	auto next_resize_check = std::chrono::steady_clock::now() + std::chrono::milliseconds(effective_epoch_interval_ms());

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
					g.sync.stop.load(std::memory_order_acquire) ||
					g.sync.attach_pending.load(std::memory_order_acquire) ||
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

		const auto now = std::chrono::steady_clock::now();

		if (now < next_resize_check) {

			continue;

		}

		next_resize_check = now + std::chrono::milliseconds(epoch_ms);

		prepare_resize_if_needed();

		if (g.prepared_resize_ready.load(std::memory_order_acquire)) {

			commit_prepared_resize_if_ready();

		}

		g.sync.notify();

	}

	g.sync.notify();

}

#endif
