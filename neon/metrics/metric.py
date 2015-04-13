# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains generic performance metric base class and utility functions.
"""

import hashlib
import os
import sys

import neon
from neon.util.compat import range
from neon.util.persist import YAMLable, ensure_dirs_exist


class Metric(YAMLable):
    """
    A Metric quantitatively measures some aspect of model performance by
    contrasting the predictions generated by the model with actual expected
    outputs.

    Though metrics may examine device buffers, all computation is carried out
    on host, and the results are returned in host buffers.

    This abstract base class defines which operations each metric must support
    to be used within our framework.
    """

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a column vector,
                                             with each case in a separate
                                             column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def report(self):
        """
        Report this metric's current calculated value(s).

        Returns:
            float or array-like: computed metric value

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def clear(self):
        """
        Reset this metric's calculated value(s)

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()


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
                    escape_colors=True, color_threshold=.01):
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
        color_threshold (float, optional): How different does a value have to
                                           be from the comp mean to warrant
                                           being colored?  Specifiy as a
                                           percentage of the mean (as a value
                                           between 0 and 1).  Defaults to .01
                                           (i.e. 1%)
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
            val = make_yellow("nan") if escape_colors else "nan"
        elif escape_colors and (comp_mean - val) > color_threshold * comp_mean:
            # val has dropped substantially enough to warrant coloring
            if header[idx] in ("auc"):
                val = make_red(latest[idx])
            else:
                val = make_green(latest[idx])
        elif escape_colors and (val - comp_mean) > color_threshold * comp_mean:
            # val has increased substantially enough to warrant coloring
            if header[idx] in ("auc"):
                val = make_green(latest[idx])
            else:
                val = make_red(latest[idx])
        else:
            # no coloring needed
            val = latest[idx]
        if comp_count == 0:
            comp_mean = make_yellow("nan") if escape_colors else "nan"
        else:
            comp_mean = "%0.5f" % comp_mean
        print(field_sep + field_sep.join([header[idx] + ":", val,
                                          ", prior " + str(comp_count) +
                                          " item mean:", comp_mean]))
    return 0
