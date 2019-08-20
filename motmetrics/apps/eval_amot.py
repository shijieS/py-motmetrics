"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using AMOT ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

https://github.com/shijieS/AwesomeMOTDataset

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')   
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='amotd')
    parser.add_argument('--solver', type=str, help='LAP solver to use', default='lapsolver')
    return parser.parse_args()

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    tsfiles = [f for f in glob.glob(os.path.join(args.tests, '*.txt'))]
    sequence_list = [os.path.basename(f)[:-4] for f in tsfiles]
    gtfiles = []
    for f in tsfiles:
        gt_file_path = os.path.basename(f).replace('-', '/')
        gt_file_dir = os.path.dirname(gt_file_path)
        gt_file_name = os.path.basename(gt_file_path)[:-4]+".csv"
        gtfiles += [os.path.join(args.groundtruths, os.path.join(os.path.join(gt_file_dir, 'gt'), gt_file_name))]

    # sequence_list = [Path(g).parts[-3] for g in gtfiles]
    # tsfiles = [t for t in tsfiles if os.path.splitext(Path(t).parts[-1])[0] in sequence_list]
    # sequence_list = [os.path.splitext(Path(t).parts[-1])[0] for t in tsfiles]
    # gtfiles = [g for g in gtfiles if Path(g).parts[-3] in sequence_list]

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')

    logging.info("Loading ground truth files.")
    gt = OrderedDict([(sequence_list[i], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for i, f in zip(trange(len(gtfiles)), gtfiles)])

    logging.info("Loading testing files")
    ts = OrderedDict([(sequence_list[i], mm.io.loadtxt(f, fmt='amotd_test')) for i, f in zip(trange(len(tsfiles)), tsfiles)])

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logging.info('Running metrics')
    
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')