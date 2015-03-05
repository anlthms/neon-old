# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.
"""

# import numpy as np
# import pycuda.driver as drv
from matplotlib import pyplot as plt
plt.interactive(1)
import numpy as np


def print_performance_stats(backend, logger):


    call_list = backend.flop_dict.keys()
    used_call_list = []
    timed_calls = []
    timed_times = []
    total_time = 0
    total_gflop = 0
    num_bins = 30
    for call in call_list:
        logger.info("Performed %2.2f GFLOP \tin %2.2fs \t(%d %s calls)",
                    sum(backend.flop_dict[call])/ 1e9,
                    sum(backend.time_dict[call]),
                    len(backend.flop_dict[call]),
                    call)

        # Histogram of where the time is spent.
        gflop_array = np.array(backend.flop_dict[call]) / 1e9
        time_array = np.array(backend.time_dict[call])
        total_time += time_array.sum()
        total_gflop += gflop_array.sum()
        flop_per_s = gflop_array / time_array  # in GFLOP/s
        # plot only the biggest contributors
        if time_array.sum() > .2:
            used_call_list.append(call)
            timed_calls.append(flop_per_s)
            timed_times.append(time_array)
    #import pdb; pdb.set_trace()
    plt.figure(1)
    n, bins, patches = plt.hist(timed_calls, num_bins,
                                weights=timed_times, #flop_array/ 1e9,
                                range=(0, 5000),
                                histtype='barstacked',
                                normed=0, alpha=0.5)


    plt.title(r'Time vs. Compute, total %2.2fs %2.2fGF average %2.2fGFLOP/S'  % (total_time, total_gflop, total_gflop/total_time))
    plt.xlabel('GFLOP/s')
    plt.ylabel('op count / GFLOP')
    plt.xlim((0, 5000))
    plt.ylabel('time / s')
    plt.legend(used_call_list)
    plt.show()
    #
    # TODO: Within a bin, figure out how many FLOP were performed, or how many seconds were spend.

