# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.


Soumith benchmarks:

ccn2
fprop_conv from conv1 in     54.62 ms per call with 10 calls totaling to 2675.86 GFLOPS, 10523.74GFLOP
update_conv from conv1 in    76.71 ms per call with 10 calls totaling to 1912.79 GFLOPS, 10564.85GFLOP


fprop_conv from conv2 in     115.07 ms per call with 10 calls totaling to 2267.50 GFLOPS, 18786.19GFLOP
bprop_conv from conv2 in     81.36 ms per call with 10 calls totaling to 3257.21 GFLOPS, 19079.72GFLOP
update_conv from conv2 in    154.68 ms per call with 10 calls totaling to 1693.39 GFLOPS, 18859.57GFLOP
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
    print "CALL LIST is", call_list
    used_call_list = []
    timed_calls = []
    timed_times = []
    total_time = 0
    total_tflop = 0
    num_bins = 30
    for call in call_list:
        logger.info("Performed %2.2f GFLOP \tin %2.2fs \t(%d %s calls)",
                    sum(backend.flop_dict[call])/ 1e9,
                    sum(backend.time_dict[call]),
                    len(backend.flop_dict[call]),
                    call)

        # Histogram of where the time is spent.
        tflop_array = np.array(backend.flop_dict[call]) / 1e12
        time_array = np.array(backend.time_dict[call])
        total_time += time_array.sum()
        total_tflop += tflop_array.sum()
        flop_per_s = tflop_array / time_array  # in GFLOP/s
        # plot only the biggest contributors
        if time_array.sum() > .001:
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
                                   if x == layer]).mean() # mean over iters! Mean must be the problem?!
            #fumith = np.array([backend.time_dict[call][i] for i, x in enumerate(backend.layer_dic[call]) if x == layer]) # 72 elements from the 72 minibatches in an epoch (72*128=9216, 3 macros)
            #import pdb; pdb.set_trace()
            flop_stats = np.array([backend.flop_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).sum()
            layer_flops_stash[call + " from " + layer] =  flop_stats / time_stats / 1e9
            layer_time_stash[call + " from " + layer] =  time_stats # sum of calls
            soumith_stash[call + " from " + layer] =  1000. * soumith_be # mean of calls. back to ms

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
             color=paren_col_stash, align='center', alpha=0.5) # color paren_col_stash
    plt.yticks(range(len(paren_stash)), paren_stash.keys())
    plt.title(r'Breakdown of MOP calls by parent')
    plt.xlabel('time/s')

    # Second plot: speed vs. time
    plt.subplot(1,2,2)
    n, bins, patches = plt.hist(timed_calls, num_bins,
                                weights=timed_times, range=(0, 5000),
                                #color=['g' for i in timed_calls],
                                histtype='barstacked', normed=0, alpha=0.5)
    plt.title(r'Total %2.1fs %2.0fTF average %2.0fTFLOP/s'
              % (total_time, total_tflop/1000., total_tflop/total_time))
    plt.xlabel('TFLOPS')
    plt.ylabel('op count / GFLOP')
    plt.xlim((0, 5500))
    plt.ylabel('Time (s)')
    plt.legend(used_call_list)
    sufx = 'inet_fp16'
    plt.savefig('figure1_'+sufx+'.pdf', dpi=500) # supposedly savefig overrides figure dpi value



    # TODO: do a version of the second plots with layer instead of parent, and Flops instead of time.
    plt.figure(2, figsize=(12, 6), dpi=120, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.subplot(1,2,1)
    plt.barh(range(len(layer_flops_stash)), layer_flops_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), layer_flops_stash.keys())
    plt.title(r'Breakdown of MOP calls by layer')
    plt.xlim((0, 5500))
    plt.xlabel('TFLOPS')

    # second plot: time per call
    plt.subplot(1,2,2)
    plt.barh(range(len(layer_flops_stash)), layer_time_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), range(len(layer_flops_stash)))
    plt.title(r'Breakdown of MOP calls by layer')
    #plt.xlim((0, 7))
    plt.xlabel('Time (s)')

    plt.savefig('figure2_'+sufx+'.pdf', dpi=500)


    # print out soumith benchmakr numers:
    #import pdb; pdb.set_trace()
    logger.info("Soumith Benchmarks")
    sum_of_all_calls = 0
    for i, key in enumerate(soumith_stash.keys()):
        logger.info("Performed %s in\t %2.2f ms per call with 10 calls totaling to %2.2f GFLOPS, %2.2fGFLOP", key, soumith_stash[key], layer_flops_stash[key], layer_flops_stash[key]*layer_time_stash[key])
        sum_of_all_calls += soumith_stash[key]
    logger.info("Total time in call %2.2f ms ", sum_of_all_calls)


