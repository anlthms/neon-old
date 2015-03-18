# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Variance of model performance with fp32, fp16 and stochastic rounding from
CIFAR10 model, and a bar plot for the Toch7 benchmark nubmers.
"""

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
    for item, col in zip([fp16_sto_test, fp16_normal_test, fp32_test], ['r', 'g', 'b']):
        density=gaussian_kde(item)

        # set the covariance_factor, lower means more detail
        density.covariance_factor = lambda : .5
        density._compute_covariance()

        # generate a fake range of x values
        xs = np.arange(29, 35, .1)

        # fill y values using density class
        ys = density(xs)
        plt.plot(xs, ys, color=col)
        plt.fill_between(xs, 0, ys, facecolor=col, alpha=0.5)
    plt.legend(['fp 16 sto', 'fp16', 'fp32'])
    plt.savefig('figure3compare16vs32.pdf', dpi=500)

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
        plt.subplot(1, 5, i+1)
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


