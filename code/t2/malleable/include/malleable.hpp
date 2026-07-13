#pragma once

#include <mpi.h>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#define MAL_ALWAYS_INLINE __attribute__((always_inline)) inline
#define MAL_LIKELY(x) __builtin_expect(!!(x), 1)
#define MAL_UNLIKELY(x) __builtin_expect(!!(x), 0)

int mal_rank();
int mal_size();
int mal_active_size();
double mal_t_origin();

enum MalLogLevel {

	MAL_LOG_DEBUG, MAL_LOG_INFO, MAL_LOG_WARN, MAL_LOG_ERROR,

};

const char* mal_log_level_name(MalLogLevel level);
bool mal_should_log(MalLogLevel level);

#define MAL_LOG(level, fmt, ...) do { MalLogLevel _mal_level = (level); if (mal_should_log(_mal_level)) printf("[t=%10.4f][%-5s][R%d] | " fmt "\n", MPI_Wtime() - mal_t_origin(), mal_log_level_name(_mal_level), mal_rank(), ##__VA_ARGS__); } while (0)
#define MAL_LOG_L(level, tag, fmt, ...) do { MalLogLevel _mal_level = (level); if (mal_should_log(_mal_level)) printf("[t=%10.4f][%-5s][%-6s][R%d] | " fmt "\n", MPI_Wtime() - mal_t_origin(), mal_log_level_name(_mal_level), (tag), mal_rank(), ##__VA_ARGS__); } while (0)

struct MalVec;
struct MalAcc;

enum MalLoopPhase {

	MAL_LOOP_WAITING_ACTIVATION, MAL_LOOP_ATTACHING, MAL_LOOP_RUNNING, MAL_LOOP_FINISHED,

};

enum MalAttachPolicy {

	MAL_ATTACH_PARTITIONED, MAL_ATTACH_SHARED_ACTIVE, MAL_ATTACH_SHARED_ALL,

};

enum MalAttachExecMode {

	MAL_ATTACH_INHERIT, MAL_ATTACH_SYNC, MAL_ATTACH_ASYNC,

};

enum MalDataAccessMode {

	MAL_ACCESS_READ_WRITE, MAL_ACCESS_READ_ONLY,

};

enum MalResizePolicy {

	MAL_RESIZE_POLICY_AUTO, MAL_RESIZE_POLICY_THROUGHPUT, MAL_RESIZE_POLICY_EFFICIENCY, MAL_RESIZE_POLICY_FIXED_SEQUENCE, MAL_RESIZE_POLICY_COST,

};

template<typename T> struct MpiType;
template<> struct MpiType<int> { static MPI_Datatype value() { return MPI_INT; } };
template<> struct MpiType<long> { static MPI_Datatype value() { return MPI_LONG; } };
template<> struct MpiType<long long> { static MPI_Datatype value() { return MPI_LONG_LONG; } };
template<> struct MpiType<unsigned> { static MPI_Datatype value() { return MPI_UNSIGNED; } };
template<> struct MpiType<unsigned long> { static MPI_Datatype value() { return MPI_UNSIGNED_LONG; } };
template<> struct MpiType<float> { static MPI_Datatype value() { return MPI_FLOAT; } };
template<> struct MpiType<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };

struct alignas(64) MalFor {

	long start{0};
	long end{0};
	long current{0};
	long* user_iter{nullptr};
	long* user_limit{nullptr};
	std::atomic<MalLoopPhase> phase{MAL_LOOP_WAITING_ACTIVATION};
	std::atomic<long> confirmed_iter{LONG_MIN};
	size_t plan_idx{0};
	size_t check_counter{0};

	std::vector<std::pair<long,long>> plan_ranges;
	std::vector<long> plan_local_bases;
	std::vector<MalVec*> vecs;
	std::vector<MalAcc*> accs;

	MalFor() = default;
	~MalFor();
	MalFor(const MalFor&) = delete;
	MalFor& operator=(const MalFor&) = delete;
	MalFor(MalFor&& other) noexcept;
	MalFor& operator=(MalFor&&) = delete;

};

struct MalCollapseSpec {

	std::vector<long> extents;
	std::vector<long> strides;
	long total_iters{0};

};

void mal_init(MalResizePolicy policy = MAL_RESIZE_POLICY_AUTO);
void mal_finalize();

void mal_set_epoch_interval_ms(int ms);
void mal_set_resize_enabled(bool enabled);
void mal_set_attach_exec_mode(MalAttachExecMode mode);
[[nodiscard]] MalAttachExecMode mal_get_attach_exec_mode();
void mal_wait_attach_tasks();

[[nodiscard]] MalFor mal_for(long total_iters, long& iter, long& limit);
[[nodiscard]] MalCollapseSpec mal_make_collapse_spec(const long* extents, size_t ndims);
[[nodiscard]] MalFor mal_for_collapse(const MalCollapseSpec& spec, long& iter, long& limit);
void mal_collapse_decode(const MalCollapseSpec& spec, long flat_iter, long* indices_out);

struct MalForND {

	std::unique_ptr<MalFor> base;
	MalCollapseSpec spec;
	std::vector<long*> iter_vars;
	std::vector<long*> limit_vars;
	std::vector<long> starts;
	std::vector<long> limits;
	std::vector<long> decoded_idx;
	long flat{0};
	long flat_limit{0};
	long last_flat{LONG_MIN};
	bool done{true};

	MalForND() = default;
	MalForND(const MalForND&) = delete;
	MalForND& operator=(const MalForND&) = delete;

	MalForND(MalForND&& other) noexcept {

		*this = std::move(other);

	}

	MalForND& operator=(MalForND&& other) noexcept {

		if (this == &other) {

			return *this;

		}

		base = std::move(other.base);
		spec = std::move(other.spec);
		iter_vars = std::move(other.iter_vars);
		limit_vars = std::move(other.limit_vars);
		starts = std::move(other.starts);
		limits = std::move(other.limits);
		decoded_idx = std::move(other.decoded_idx);
		flat = other.flat;
		flat_limit = other.flat_limit;
		last_flat = other.last_flat;
		done = other.done;

		if (base) {

			base->user_iter = &flat;
			base->user_limit = &flat_limit;

		}

		return *this;

	}

};

[[nodiscard]] MalForND mal_for_nd_begin(long* const* vars, const long* starts, const long* limits, size_t ndims);
[[nodiscard]] MalForND mal_for_nd_begin(long* const* iter_vars, long* const* limit_vars, const long* starts, const long* limits, size_t ndims);
[[nodiscard]] bool mal_for_nd_done(const MalForND& f);
MalFor& mal_for_nd_base(MalForND& f);

void mal_check_for(MalFor& f);
void mal_check_for(MalForND& f);

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_N, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE);
void mal_attach_vec(MalForND& f, void** user_ptr, size_t elem_size, long total_N, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE);

void mal_attach_vec_ragged(MalFor& f, void** user_ptr, size_t elem_size, long total_inner, const long* row_offsets, long n_rows, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_ONLY);

void mal_attach_csr(MalFor& f, void** values, size_t value_elem_size, void** col_indices, size_t index_elem_size, long* row_ptr, long n_rows, long nnz);

void mal_allgather_replicated(MalFor& f, void* full_buf, size_t elem_size, long total_n);

void mal_bcast_impl(void* buf, int count, MPI_Datatype dtype, int root);

void mal_sync_impl(MalFor& f, void* buf, int count, MPI_Datatype dtype, MPI_Op op);

void halo_exchange_field(MalFor& f, void* buf, size_t elem, long total);

void mal_loop_horizon(long steps_remaining);

void mal_step_sync(MalFor& f, void* full_buf, size_t elem_size, long total_n);

void mal_step(MalFor& f, void* full_buf, size_t elem_size, long total_n);

namespace detail {

	struct AccDesc {

		void* ptr;
		MPI_Datatype dtype;
		MPI_Op dop;
		size_t esz;
		void (*fn_get) (const void* p, void* dst);
		void (*fn_set) (void* p, const void* src);
		void (*fn_add) (void* p, const void* src);
		void (*fn_reset)(void* p);

	};

	void acc_register(MalFor& f, AccDesc d, int result_rank);

	template<typename T> inline void acc_get_t(const void* p, void* d) {

		*static_cast<T*>(d) = *static_cast<const T*>(p);

	}

	template<typename T> inline void acc_set_t(void* p, const void* s) {

		*static_cast<T*>(p) = *static_cast<const T*>(s);

	}

	template<typename T> inline void acc_add_t(void* p, const void* s) {

		*static_cast<T*>(p) += *static_cast<const T*>(s);

	}

	template<typename T> inline void acc_reset_t(void* p) {

		*static_cast<T*>(p) = T{};

	}

}

template<typename T> inline void mal_attach_acc(MalFor& f, T& acc, MPI_Datatype dtype, MPI_Op op, int result_rank = 0) {

	detail::acc_register(f, {

		&acc, dtype, op, sizeof(T), detail::acc_get_t<T>, detail::acc_set_t<T>, detail::acc_add_t<T>, detail::acc_reset_t<T>, }, result_rank);

}

template<typename T> inline void mal_attach_acc(MalFor& f, T& acc, int result_rank = 0) {

	mal_attach_acc(f, acc, MpiType<T>::value(), MPI_SUM, result_rank);

}

template<typename T> inline void mal_attach_acc(MalForND& f, T& acc, int result_rank = 0) {

	mal_attach_acc(mal_for_nd_base(f), acc, result_rank);

}

template<typename T> inline void mal_sync(MalFor& f, T& value, MPI_Op op = MPI_SUM) {

	mal_sync_impl(f, &value, 1, MpiType<T>::value(), op);

}

template<typename T> inline void mal_sync(MalFor& f, T* values, int count, MPI_Op op = MPI_SUM) {

	mal_sync_impl(f, values, count, MpiType<T>::value(), op);

}

template<typename T> inline void mal_sync(MalForND& f, T& value, MPI_Op op = MPI_SUM) {

	mal_sync_impl(mal_for_nd_base(f), &value, 1, MpiType<T>::value(), op);

}

template<typename T> inline void mal_bcast(T& value, int root = 0) {

	mal_bcast_impl(&value, 1, MpiType<T>::value(), root);

}

template<typename T> inline void mal_bcast(T* values, int count, int root = 0) {

	mal_bcast_impl(values, count, MpiType<T>::value(), root);

}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE);
void mal_attach_mat(MalForND& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE);
