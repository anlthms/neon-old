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





def print_accuracy():
    fp16_sto_train=[0.77724, 0.77724, 0.61899, 0.79527, 0.86739, 0.50080, 1.10377,
                    1.20994, 0.63502, 0.85737, 0.64904, 1.13181, 0.55889, 0.89543,
                    0.69311,
                    0.37861, 0.92949, 0.86338, 0.49279, 1.20192, 0.67508, 1.02364,
                    0.79127, 0.79327, 0.59095, 0.98558, 0.44471, 0.47075, 0.61098,
                    0.59696, 1.55248, 1.07572, 0.56891, 1.03766, 1.22796, 1.89303,
                    1.20793]
    fp16_sto_test  =[31.97115, 31.93109, 31.63061, 32.03125, 32.17147, 30.68910,
                     32.81250, 32.31170, 32.39183, 31.97115, 31.51042, 32.39183,
                     32.01122, 32.21154, 30.86939,
                    31.43029, 32.69231, 31.71074, 31.65064, 32.09135, 31.69070,
                    31.97115, 31.16987, 32.71234, 31.00961, 32.67228, 31.81090,
                    31.51042, 31.91106, 32.51202, 32.87260, 31.63061, 32.03125,
                    31.35016, 32.63221, 32.89263, 31.95112]

    fp16_normal_train = [0.59095, 0.81530, 0.45072, 1.42027, 0.59696, 0.75721,
                         0.89343, 0.55088, 0.73518, 0.66907, 0.54888, 0.90745,
                         0.46875, 1.25200, 1.23598, 0.78926, 0.81330, 0.74119,
                         0.60897, 0.80729, 1.16186]
    fp16_normal_test = [31.93109, 32.05128, 31.41026, 33.33334, 31.00961,
                        31.93109, 31.93109, 31.45032, 31.20994, 31.00961,
                        31.25000, 31.73077, 32.41186, 32.85256, 31.97115,
                        32.69231, 33.05289, 30.80930, 31.35016, 31.69070,
                        32.73237]

    fp32_train = [0.65505, 0.60296, 0.70112, 0.85136, 0.70112, 0.63702, 0.52484,
                  0.59295, 1.87300, 0.58494, 0.67909, 0.46274, 0.53686, 0.88341,
                  0.66707, 0.80929, 0.75721, 0.55889, 0.78926, 0.64503, 0.66506,
                  1.31210, 0.73117, 0.83133, 0.81731, 1.24800, 1.10577, 0.57091]
    fp32_test = [31.43029, 32.13141, 32.52203, 31.22997, 31.39022, 31.06971,
                 31.59055, 31.09976, 33.96434, 30.93950, 32.06130, 30.53886,
                 32.70232, 32.17147, 31.27003, 31.56050, 30.80930, 32.10136,
                 32.82252, 32.40184, 30.55889, 32.58213, 30.70914, 31.76082,
                 32.06130, 33.17308, 32.52203, 31.51042]


    plt.figure(3, figsize=(4, 4), dpi=120, facecolor='w', edgecolor='k')
    plt.hist([fp32_test + [i+27 for i in fp32_train],
              fp16_normal_test + [i+27 for i in fp16_normal_train],
              fp16_sto_test + [i+27 for i in fp16_sto_train]
              ], 10, normed=0, alpha=0.5, histtype='stepfilled')
    plt.legend(['fp 32', 'fp16', 'fp16 sto.'])
    plt.savefig('figure3compare16vs32.pdf', dpi=500)


    # FANCY PLOT:
    for item in [fp16_sto_test, fp16_normal_test, fp32_test]:
        density=gaussian_kde(item)

        # set the covariance_factor, lower means more detail
        density.covariance_factor = lambda : .5
        density._compute_covariance()

        # generate a fake range of x values
        xs = np.arange(29,34,.1)

        # fill y values using density class
        ys = density(xs)
        plt.plot(xs,ys)
        plt.fill_between(xs,0,ys,alpha=0.5)
    plt.legend(['fp 32', 'fp16', 'fp16 sto.'])


    """
    fp32numbers = [31.53045, 32.56210, 32.11138, 32.63221, 32.12139]  # 32.19
    fp16normal  = [31.97115, 32.05128, 31.97115, 32.13141, 31.20994]  # 31.87
    fp16stochas = [32.51202, 32.33173, 31.67067, 31.89103, 32.11138]  # 32.10
    fp32a = [0.69311, 1.41627, 0.60897, 1.19992, 0.62901]
    fp16s = [1.02764, 0.87540, 0.66907, 0.67107, 0.97356]
    fp16n = [0.91947, 0.73317, 0.75521, 0.66907, 0.47676]
    """


def soumith_benchmark():
    soumith=dict()  #    L1  L2   L3  L4  L5   B1   B2   B3  B4  B5
    soumith['neon16'] = [38, 114, 40,  5,  9,  107, 250, 92, 10, 19]  ## too fast because of low entropy!
    soumith['neon32'] = [47, 172, 68, 9, 14,   144, 416, 143, 14, 28]  # padding, death!

    soumith['ccn2th'] = [57, 182, 68,  8, 14,  147, 438, 150, 15, 27]
    soumith['torch7'] = [132,212, 165,32, 48,  320, 584, 201, 37, 43]
    soumith['cu_dnn'] = [76, 000, 00, 13, 21,  194, 000, 000, 26, 45]
    # maxes = [i for key in soumith.keys() ]  # not a good idea since we want absolute times to be visible.
    plt.figure(4, figsize=(4, 6), dpi=120, facecolor='w', edgecolor='k')
    for i, key  in enumerate(soumith.keys()):
        plt.bar(arange(6+4)+.1*i, soumith[key], color=np.array((i,i,i))/6., width=0.08)
    plt.legend(soumith.keys())
    plt.ylabel('Time / ms')
    plt.xticks(range(6+4), ['L1', 'L2', 'L3', 'L4', 'L5', 'L1', 'L2', 'L3', 'L4', 'L5'])
    plt.xlabel('Forward                  Backward')
    plt.savefig('fig_soumith_bench.pdf', dpi=500)


    sumi = np.array(((38, 114, 40,  5,  9,  107, 250, 92, 10, 19),
                     (47, 172, 68, 9, 14,   144, 416, 143, 14, 28),
                     (57, 182, 68,  8, 14,  147, 438, 150, 15, 27),
                     (132,212, 165,32, 48,  320, 584, 201, 37, 43),
                     (76, 000, 00, 13, 21,  194, 000, 000, 26, 45)))

    plt.figure(4, figsize=(4, 6), dpi=120, facecolor='w', edgecolor='k')
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.bar(left=arange(10),
                height=sumi[i,:],
                color=['r', 'g', 'b', 'c', 'm', 'r', 'g', 'b', 'c', 'm'],
                width=0.8)
        plt.ylim((0,600))
    plt.legend(soumith.keys())
    plt.ylabel('Time / ms')
    plt.xticks(range(6+4), ['L1', 'L2', 'L3', 'L4', 'L5', 'L1', 'L2', 'L3', 'L4', 'L5'])
    plt.xlabel('Forward                  Backward')
    plt.savefig('fig_soumith_bench.pdf', dpi=500)


"""

CONFIG: input = 3x128x128 * ker = 3x96x11x11 (bs = 128, stride = 1)
nn.SpatialConvolutionMM                 :updateOutput():     133.10
nn.SpatialConvolutionMM              :updateGradInput():     132.02
nn.SpatialConvolutionMM            :accGradParameters():     186.72
nn.SpatialConvolutionMM                          :TOTAL:     451.84
ccn2.SpatialConvolution                 :updateOutput():      56.51
ccn2.SpatialConvolution              :updateGradInput():      79.35
ccn2.SpatialConvolution            :accGradParameters():      67.16
ccn2.SpatialConvolution                          :TOTAL:     203.02

CONFIG: input = 64x64x64 * ker = 64x128x9x9 (bs = 128, stride = 1)
nn.SpatialConvolutionMM                 :updateOutput():     211.80
nn.SpatialConvolutionMM              :updateGradInput():     231.67
nn.SpatialConvolutionMM            :accGradParameters():     352.08
nn.SpatialConvolutionMM                          :TOTAL:     795.55
ccn2.SpatialConvolution                 :updateOutput():     181.89
ccn2.SpatialConvolution              :updateGradInput():     178.54
ccn2.SpatialConvolution            :accGradParameters():     259.63
ccn2.SpatialConvolution                          :TOTAL:     620.06

CONFIG: input = 128x32x32 * ker = 128x128x9x9 (bs = 128, stride = 1)
nn.SpatialConvolutionMM                 :updateOutput():     164.65
nn.SpatialConvolutionMM              :updateGradInput():     117.85
nn.SpatialConvolutionMM            :accGradParameters():      83.39
nn.SpatialConvolutionMM                          :TOTAL:     365.88
ccn2.SpatialConvolution                 :updateOutput():      68.30
ccn2.SpatialConvolution              :updateGradInput():      65.97
ccn2.SpatialConvolution            :accGradParameters():      84.18
ccn2.SpatialConvolution                          :TOTAL:     218.45

CONFIG: input = 128x16x16 * ker = 128x128x7x7 (bs = 128, stride = 1)
nn.SpatialConvolutionMM                 :updateOutput():      32.11
nn.SpatialConvolutionMM              :updateGradInput():      21.33
nn.SpatialConvolutionMM            :accGradParameters():      15.28
nn.SpatialConvolutionMM                          :TOTAL:      68.72
ccn2.SpatialConvolution                 :updateOutput():       7.96
ccn2.SpatialConvolution              :updateGradInput():       6.38
ccn2.SpatialConvolution            :accGradParameters():       8.77
ccn2.SpatialConvolution                          :TOTAL:      23.11

CONFIG: input = 384x13x13 * ker = 384x384x3x3 (bs = 128, stride = 1)
nn.SpatialConvolutionMM                 :updateOutput():      47.87
nn.SpatialConvolutionMM              :updateGradInput():      21.93
nn.SpatialConvolutionMM            :accGradParameters():      21.49
nn.SpatialConvolutionMM                          :TOTAL:      91.29
ccn2.SpatialConvolution                 :updateOutput():      13.87
ccn2.SpatialConvolution              :updateGradInput():      12.79
ccn2.SpatialConvolution            :accGradParameters():      16.01
ccn2.SpatialConvolution                          :TOTAL:      42.66
"""
