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
from pdb import set_trace as trace

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
        if time_array.sum() > .4:
            used_call_list.append(call)
            timed_calls.append(flop_per_s)
            timed_times.append(time_array)


    #
    # First plot: speed vs. time
    plt.figure(1)
    plt.subplot(1,2,2)
    n, bins, patches = plt.hist(timed_calls, num_bins,
                                weights=timed_times, range=(0, 5000),
                                histtype='barstacked', normed=0, alpha=0.5)
    plt.title(r'Time vs. Compute, total %2.2fs %2.2fGF average %2.2fGFLOP/S'
              % (total_time, total_gflop, total_gflop/total_time))
    plt.xlabel('GFLOP/s')
    plt.ylabel('op count / GFLOP')
    plt.xlim((0, 5000))
    plt.ylabel('time / s')
    plt.legend(used_call_list)

    # compute timing per parent call:
    bar_stash = dict()
    col_stash = dict()
    for call in used_call_list:
        unique_paren_list = set(backend.paren_dic[call])
        for paren in unique_paren_list:
            # add up times for "call" from "paren"
            time_stats = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.paren_dic[call])
                                   if x == paren]).sum()
            bar_stash[call + " from " + paren] =  time_stats

    col_stash =  ['b' if 'sub' in k else
                  'g' if 'mul' in k else
                  'r' if 'fprop_fc' in k else
                  'c' if 'add' in k else
                  'm' if 'te_fc' in k else
                  'k' for k in bar_stash.keys()]

    # Second plot: detailed breakdown of time
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.barh(range(len(bar_stash)), bar_stash.values(),
             color=col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(bar_stash)), bar_stash.keys())
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.title(r'Breakdown of MOP calls by parent')
    plt.xlabel('time/s')
    plt.show()

