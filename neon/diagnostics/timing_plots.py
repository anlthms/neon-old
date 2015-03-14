# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.
"""

# import numpy as np
# import pycuda.driver as drv
import matplotlib
matplotlib.use('Agg')  # not for plotting but write to file.
matplotlib.rcParams['pdf.fonttype'] = 42  # TTF to be editable
from matplotlib import pyplot as plt
#plt.interactive(1)
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
        if time_array.sum() > .01:
            used_call_list.append(call)
            timed_calls.append(flop_per_s)
            timed_times.append(time_array)

    # compute timing per parent call:
    paren_stash = dict()
    for call in used_call_list:
        unique_paren_list = set(backend.paren_dic[call])
        for paren in unique_paren_list:
            # add up times for "call" from "paren"
            time_stats = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.paren_dic[call])
                                   if x == paren]).sum()
            paren_stash[call + " from " + paren] =  time_stats

    # compute timing per layer call:
    layer_flops_stash = dict()
    layer_time_stash = dict()
    soumith_stash = dict()
    for call in used_call_list:
        unique_layer_list = set(backend.layer_dic[call])
        for layer in unique_layer_list:
            # add up times for "call" from "paren"
            time_stats = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).sum() # sum all iterations
            soumith_be = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).mean() # mean over iters
            import pdb; pdb.set_trace()
            flop_stats = np.array([backend.flop_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).sum()
            layer_flops_stash[call + " from " + layer] =  flop_stats / time_stats / 1e9
            layer_time_stash[call + " from " + layer] =  time_stats
            soumith_stash[call + " from " + layer] =  soumith_be

    # colors for the bars
    paren_col_stash =  ['b' if 'sub' in k else
                        'g' if 'mul' in k else
                        'r' if 'fprop_fc' in k else
                        'c' if 'add' in k else
                        'm' if 'te_fc' in k else
                        'k' for k in paren_stash.keys()]

    layer_col_stash =  ['b' if 'conv1' in k else
                        'g' if 'conv2' in k else
                        'r' if 'fc' in k else
                        'c' if 'anon' in k else
                        'm' if 'output' in k else
                        'k' for k in layer_flops_stash.keys()]

    # First plot: detailed breakdown of time
    plt.figure(1, figsize=(12, 6), dpi=120, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    plt.subplot(1,2,1)
    plt.barh(range(len(paren_stash)), paren_stash.values(),
             color=paren_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(paren_stash)), paren_stash.keys())
    plt.title(r'Breakdown of MOP calls by parent')
    plt.xlabel('time/s')

    # Second plot: speed vs. time
    plt.subplot(1,2,2)
    n, bins, patches = plt.hist(timed_calls, num_bins,
                                weights=timed_times, range=(0, 5000),
                                histtype='barstacked', normed=0, alpha=0.5)
    plt.title(r'Time vs. Compute, total %2.2fs %2.2fGF average %2.2fGFLOP/S'
              % (total_time, total_gflop, total_gflop/total_time))
    plt.xlabel('GFLOP/s')
    plt.ylabel('op count / GFLOP')
    plt.xlim((0, 5500))
    plt.ylabel('time / s')
    plt.legend(used_call_list)

    plt.savefig('figure1_i1k_max.pdf', dpi=500) # supposedly savefig overrides figure dpi value


    #
    # TODO: do a version of the second plots with layer instead of parent, and Flops instead of time.
    plt.figure(2, figsize=(12, 6), dpi=120, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.subplot(1,2,1)
    plt.barh(range(len(layer_flops_stash)), layer_flops_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), layer_flops_stash.keys())
    plt.title(r'Breakdown of MOP calls by layer')
    plt.xlim((0, 5500))
    plt.xlabel('GFLOPS')

    # second plot: time per call
    plt.subplot(1,2,2)
    plt.barh(range(len(layer_flops_stash)), layer_time_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), range(len(layer_flops_stash)))
    plt.title(r'Breakdown of MOP calls by layer')
    plt.xlim((0, 7))
    plt.xlabel('time / s')

    plt.savefig('figure2_i1k_max.pdf', dpi=500)

    # print out soumith benchmakr numers:
    #import pdb; pdb.set_trace()
    logger.info("Soumith Benchmarks")
    for i, key in enumerate(soumith_stash.keys()):
        logger.info("Performed %s in\t %2.2f ms ", key, 1000*soumith_stash[key])


    # def print_accuracy():
    fp32numbers = [31.53045, 32.56210, 32.11138, 32.63221, 32.12139]  # 32.19
    fp16normal  = [31.97115, 32.05128, 31.97115, 32.13141, 31.20994]  # 31.87
    fp16stochas = [32.51202, 32.33173, 31.67067, 31.89103, 32.11138]  # 32.10
    fp32a = [0.69311, 1.41627, 0.60897, 1.19992, 0.62901]
    fp16s = [1.02764, 0.87540, 0.66907, 0.67107, 0.97356]
    fp16n = [0.91947, 0.73317, 0.75521, 0.66907, 0.47676]

    plt.figure(3, figsize=(4, 4), dpi=120, facecolor='w', edgecolor='k')
    plt.hist([fp32numbers + [i+29 for i in fp32a],
              fp16normal + [i+29 for i in fp16n],
              fp16stochas + [i+29 for i in fp16s]
              ], 8, normed=0, alpha=0.5)
    plt.legend(['fp 32', 'fp16', 'fp16 sto.'])
    plt.savefig('figure3compare16vs32.pdf', dpi=500)


"""
# stochastic rounding

2015-03-13 17:23:49,755 INFO:mlp - train set misclass rate: 1.02764%
2015-03-13 17:23:49,808 INFO:mlp - test set misclass rate: 32.51202%

2015-03-13 17:25:06,032 INFO:mlp - train set misclass rate: 0.87540%
2015-03-13 17:25:06,085 INFO:mlp - test set misclass rate: 32.33173%

2015-03-13 17:26:23,200 INFO:mlp - train set misclass rate: 0.66907%
2015-03-13 17:26:23,253 INFO:mlp - test set misclass rate: 31.67067%

2015-03-13 17:27:40,618 INFO:mlp - train set misclass rate: 0.67107%
2015-03-13 17:27:40,671 INFO:mlp - test set misclass rate: 31.89103%

2015-03-13 17:28:58,087 INFO:mlp - train set misclass rate: 0.97356%
2015-03-13 17:28:58,140 INFO:mlp - test set misclass rate: 32.11138%


# pf 16 normal rounding
2015-03-13 17:12:18,698 INFO:mlp - train set misclass rate: 0.91947%
2015-03-13 17:12:18,748 INFO:mlp - test set misclass rate: 31.97115%

2015-03-13 17:13:31,959 INFO:mlp - train set misclass rate: 0.73317%
2015-03-13 17:13:32,009 INFO:mlp - test set misclass rate: 32.05128%

2015-03-13 17:14:45,751 INFO:mlp - train set misclass rate: 0.75521%
2015-03-13 17:14:45,801 INFO:mlp - test set misclass rate: 31.97115%

2015-03-13 17:16:00,422 INFO:mlp - train set misclass rate: 0.66907%
2015-03-13 17:16:00,472 INFO:mlp - test set misclass rate: 32.13141%

2015-03-13 17:17:14,994 INFO:mlp - train set misclass rate: 0.47676%
2015-03-13 17:17:15,045 INFO:mlp - test set misclass rate: 31.20994%


# FP32
2015-03-13 13:08:29,419 INFO:mlp - train set misclass rate: 0.69311%
2015-03-13 13:08:29,466 INFO:mlp - test set misclass rate: 31.53045%

2015-03-13 13:51:15,377 INFO:mlp - train set misclass rate: 1.41627%
2015-03-13 13:51:15,423 INFO:mlp - test set misclass rate: 32.56210%

2015-03-13 14:34:14,518 INFO:mlp - train set misclass rate: 0.60897%
2015-03-13 14:34:14,565 INFO:mlp - test set misclass rate: 32.11138%

2015-03-13 15:16:58,690 INFO:mlp - train set misclass rate: 1.19992%
2015-03-13 15:16:58,736 INFO:mlp - test set misclass rate: 32.63221%

2015-03-13 15:59:42,463 INFO:mlp - train set misclass rate: 0.62901%
2015-03-13 15:59:42,510 INFO:mlp - test set misclass rate: 32.12139%
"""