/* montecarlo_plugin_policy.cpp – resize policy plugin for montecarlo_plugin.
 *
 * Compile as a shared library; the symbol 'montecarlo_policy' is loaded at
 * runtime by montecarlo_plugin via mal_set_decide_resize_plugin().
 *
 * Policy:
 *   - Scale up by 1 rank when throughput imbalance ratio exceeds 1.3 and
 *     there is room to grow.
 *   - Scale down by 1 rank when all ranks report "settled" throughput and
 *     we are above 2 ranks.
 *   - Do nothing otherwise.
 */

#include <algorithm>
#include "malleable.hpp"

extern "C" ResizeDecision montecarlo_policy(const EpochMetrics& m) {

	ResizeDecision d;

	// Scale up: load is imbalanced and we still have room to grow.
	if (m.imbalance_ratio() > 1.3) {

		d.should_resize = true;
		d.target_active_size = m.active_n + 1; // runtime clamps to [1, u_size]
		return d;

	}

	// Scale down: all ranks have settled but we still have spare ranks.
	if (m.any_settled && m.active_n > 2) {

		d.should_resize = true;
		d.target_active_size = m.active_n - 1; // runtime clamps to [1, u_size]
		return d;

	}

	return d; // should_resize = false

}
