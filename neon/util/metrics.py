# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

"""
Contains functions for computing error metrics.
"""

import hashlib
import os
import sys

import neon
from neon.util.compat import range
from neon.util.persist import ensure_dirs_exist


def misclass_sum(backend, targets, outputs, predlabels, labels,
                 misclass, retval):
    backend.argmax(targets, axis=0, out=labels)
    backend.argmax(outputs, axis=0, out=predlabels)
    backend.not_equal(predlabels, labels, misclass)
    backend.sum(misclass, axes=None, out=retval)


def auc(backend, targets, outputs):
    from sklearn import metrics
    return metrics.roc_auc_score(targets.asnumpyarray().ravel(),
                                 outputs.asnumpyarray().ravel())


def logloss(backend, targets, outputs, sums, temp, retval, eps=1e-15):
    backend.clip(outputs, eps, 1.0 - eps, out=outputs)
    backend.sum(outputs, axes=0, out=sums)
    # XXX: work around lack of broadcasting in gpu backend.
    for row in range(temp.shape[0]):
        temp[row] = sums
    backend.divide(outputs, temp, out=temp)
    backend.log(temp, out=temp)
    backend.multiply(targets, temp, out=temp)
    backend.sum(temp, axes=None, out=retval)


def dump_metrics(dump_file, experiment_file, start_time, elapsed_time, metrics,
                 field_sep="\t"):
    """
    Write or append collected metric values to the specified flat file.

    Arguments:
        dump_file (str): path to file to write. Will be created if doesn't
                         exist, or appended to (without header if it does)
        experiment_file (str): path to yaml file used to run this experiment
        start_time (str): date and time at which experiment was started.
        elapsed_time (float): time taken to run the experiment.
        metrics (dict): Collection of metric values, as returned from
                        FitPredictErrorExperiment.run() call.
        field_sep (str, optional): string used to separate each field in
                                   dump_file.  Defaults to tab character.
    """
    if dump_file is None or dump_file == '':
        df = sys.stdout()
    elif not os.path.exists(dump_file) or os.path.getsize(dump_file) == 0:
        ensure_dirs_exist(dump_file)
        df = file(dump_file, 'w')
        metric_names = []
        if isinstance(metrics, dict):
            metric_names = ["%s-%s" % (metric, dset) for metric in
                            sorted(metrics.keys()) for dset in
                            sorted(metrics[metric].keys())]
        df.write(field_sep.join(["host", "architecture", "os",
                                 "os_kernel_release", "neon_version",
                                 "yaml_name", "yaml_sha1", "start_time",
                                 "elapsed_time"] + metric_names) + "\n")
    else:
        df = file(dump_file, 'a')
    info = os.uname()
    trunc_exp_name = ("..." + os.path.sep +
                      os.path.dirname(experiment_file).split(os.path.sep)[-1] +
                      os.path.sep +
                      os.path.basename(experiment_file))
    # TODO: better handle situation where metrics recorded differ from those
    # already in file
    metric_vals = []
    if isinstance(metrics, dict):
        metric_vals = ["%.5f" % metrics[metric][dset] for metric in
                       sorted(metrics.keys()) for dset in
                       sorted(metrics[metric].keys())]
    df.write(field_sep.join([x.replace("\t", " ") for x in
                             [info[1], info[4], info[0], info[2],
                              neon.__version__, trunc_exp_name,
                              hashlib.sha1(open(experiment_file,
                                                'rb').read()).hexdigest(),
                              start_time, "%.3f" % elapsed_time] +
                             metric_vals]) + "\n")
    df.close()


def compare_metrics(dump_file, experiment_file, max_comps=10, field_sep="\t",
                    escape_colors=True):
    """
    Compares the most recent run of experiment_file with up to max_comps
    previous runs based on data collected in dump_file.  Results are displayed
    to the console.

    Arguments:
        dump_file (str): path to file to write. Will be created if doesn't
                         exist, or appended to (without header if it does)
        experiment_file (str): path to yaml file used to run this experiment
        max_comps (int, optional): collect and compare statistics against
                                   max_comps most recent prior runs of the
                                   same example.  Defaults to 10.
        field_sep (str, optional): string used to separate each field in
                                   dump_file.  Defaults to tab character.
        escape_colors (bool, optional): Should we dump diffs in a different
                                        color?  Default is true
    """
    def make_red(string):
        return "\033[31m%s\033[0m" % string

    def make_green(string):
        return "\033[32m%s\033[0m" % string

    def make_yellow(string):
        return "\033[93m%s\033[0m" % string

    data = file(dump_file).readlines()
    if len(data) < 1 or not data[0].startswith("host"):
        print("file: %s seems to have invalid format" % dump_file)
        return 1
    trunc_exp_name = ("..." + os.path.sep +
                      os.path.dirname(experiment_file).split(os.path.sep)[-1] +
                      os.path.sep +
                      os.path.basename(experiment_file))
    line_num = len(data) - 1
    header = data[0].rstrip('\r\n').split(field_sep)
    latest = None
    comps = []
    while line_num > 0 and len(comps) < max_comps:
        if trunc_exp_name in data[line_num]:
            if latest is None:
                latest = data[line_num].rstrip('\r\n').split(field_sep)
            else:
                comps.append(data[line_num].rstrip('\r\n').split(field_sep))
        line_num -= 1
    if latest is None:
        print("unable to find any lines containing %s" % trunc_exp_name)
        return 2
    for idx in range(8, len(header)):
        val = float(latest[idx])
        comp_sum = 0.0
        comp_count = 0
        for comp in comps:
            if comp[idx] != "nan":
                comp_sum += float(comp[idx])
                comp_count += 1
        if comp_count == 0:
            comp_mean = float("nan")
        else:
            comp_mean = comp_sum / comp_count
        if latest[idx] == "nan":
            val = make_yellow("nan")
        elif val < comp_mean:
            if header[idx] in ("auc"):
                val = make_red(latest[idx])
            else:
                val = make_green(latest[idx])
        elif val > comp_mean:
            if header[idx] in ("auc"):
                val = make_green(latest[idx])
            else:
                val = make_red(latest[idx])
        else:
            val = latest[idx]
        if comp_count == 0:
            comp_mean = make_yellow("nan")
        else:
            comp_mean = "%0.5f" % comp_mean
        print(field_sep + field_sep.join([header[idx] + ":", val,
                                          ", prior mean:", comp_mean]))
    return 0
