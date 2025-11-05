"""
inference.py

This module provides utilities for running inference using the trained model,

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: March 2025
"""

import utils
import pickle
import os
import torch
from metrics import ComputeSELDResults



def run_inference():

    params_file = os.path.join(model_dir, 'config.pkl')
    f = open(params_file, "rb")
    params = pickle.load(f)

    output_dir = 'test_outputs/fusion_model_conformer_model_conformer_aug'
    os.makedirs(params['output_dir'], exist_ok=True)


    seld_metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))


    with torch.no_grad():

        test_metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=False)
        test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
        utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    model_dir = "checkpoints/orig_mc_trained_ckpt_1"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_inference()
