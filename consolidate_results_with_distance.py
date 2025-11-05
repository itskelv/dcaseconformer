import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def consolidate_with_distance_override_fixed():
    """
    Consolidates outputs using a frame-level winner for class/azimuth,
    then overrides the distance with the best available prediction.
    (This version contains a bug fix for a pandas KeyError).
    """
    model_a_metrics = {
        0: [0.49, 18.84], 1: [0.53, 16.96], 2: [0.26, 22.87], 3: [0.18, 25.65],
        4: [0.19, 22.92], 5: [0.40, 23.03], 6: [0.17, 21.72], 7: [0.18, 17.64],
        8: [0.36, 23.91], 9: [0.31, 13.39], 10: [0.14, 19.50], 11: [0.00, 999],
        12: [0.00, 999],
    }
    model_b_metrics = {
        0: [0.52, 18.68], 1: [0.56, 15.44], 2: [0.47, 16.43], 3: [0.20, 22.23],
        4: [0.22, 22.00], 5: [0.41, 22.05], 6: [0.17, 21.09], 7: [0.21, 13.25],
        8: [0.43, 22.27], 9: [0.29, 11.78], 10: [0.16, 16.76], 11: [0.03, 53.00],
        12: [0.03, 9.00],
    }
    model_c_metrics = {
        0: [0.48, 19.60], 1: [0.51, 16.67], 2: [0.16, 40.58], 3: [0.17, 21.65],
        4: [0.19, 22.44], 5: [0.32, 24.34], 6: [0.13, 25.88], 7: [0.18, 13.98],
        8: [0.34, 25.22], 9: [0.30, 14.24], 10: [0.16, 28.70], 11: [0.09, 37.66],
        12: [0.11, 15.02],
    }

    base_dir = 'final_eval_outputs'
    model_a_dir = os.path.join(base_dir, 'model_conformer_aug', 'dev-test') # Distance Expert
    model_b_dir = os.path.join(base_dir, 'model_conformer_original', 'dev-test')
    model_c_dir = os.path.join(base_dir, 'sed_doa_model', 'dev-test')
    final_output_dir = os.path.join(base_dir, 'model_ensemble', 'dev-test')

    os.makedirs(final_output_dir, exist_ok=True)

    try:
        csv_files = [f for f in os.listdir(model_a_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} files. Applying DISTANCE OVERRIDE consolidation (FIXED).")
    except FileNotFoundError:
        print(f"Error: Directory not found '{model_a_dir}'.")
        return

    for file_name in tqdm(csv_files, desc="Consolidating with Distance Override"):
        path_a = os.path.join(model_a_dir, file_name)
        path_b = os.path.join(model_b_dir, file_name)
        path_c = os.path.join(model_c_dir, file_name)

        df_a = pd.read_csv(path_a) # This is our distance expert
        df_b = pd.read_csv(path_b)
        df_c = pd.read_csv(path_c)

        all_frames = sorted(list(set(df_a['frame']) | set(df_b['frame']) | set(df_c['frame'])))
        final_predictions_df = pd.DataFrame()

        for frame in all_frames:
            preds_a = df_a[df_a['frame'] == frame]
            preds_b = df_b[df_b['frame'] == frame]
            preds_c = df_c[df_c['frame'] == frame]

            avg_f_a = (sum(model_a_metrics.get(c, [0])[0] for c in preds_a['class']) / len(preds_a)) if not preds_a.empty else -1
            avg_f_b = (sum(model_b_metrics.get(c, [0])[0] for c in preds_b['class']) / len(preds_b)) if not preds_b.empty else -1
            avg_f_c = (sum(model_c_metrics.get(c, [0])[0] for c in preds_c['class']) / len(preds_c)) if not preds_c.empty else -1

            scores = [avg_f_a, avg_f_b, avg_f_c]
            predictions = [preds_a, preds_b, preds_c]
            winner_df = predictions[scores.index(max(scores))].copy()

            if winner_df.empty:
                continue

            for i, win_row in winner_df.iterrows():
                candidates = df_a[(df_a['frame'] == win_row['frame']) & (df_a['class'] == win_row['class'])]

                if not candidates.empty:
                    if len(candidates) == 1:
                        new_dist = candidates['distance'].iloc[0]
                    else:

                        winner_azimuth = win_row['azimuth']
                        best_candidate_idx = (candidates['azimuth'] - winner_azimuth).abs().idxmin()
                        new_dist = candidates.loc[best_candidate_idx, 'distance']

                    winner_df.loc[i, 'distance'] = new_dist

            final_predictions_df = pd.concat([final_predictions_df, winner_df])

        if not final_predictions_df.empty:
            final_predictions_df.sort_values(by=['frame', 'source'], inplace=True)
        final_predictions_df.to_csv(os.path.join(final_output_dir, file_name), index=False)

    print(f"\nConsolidation complete! Outputs are in: {final_output_dir}")

if __name__ == '__main__':
    consolidate_with_distance_override_fixed()