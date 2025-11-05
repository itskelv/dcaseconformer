"""
main_sed_doa.py

Train and evaluate a SELD model for the SED-DOA task.
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from parameters_sed_sde import params
from model_conformer_for_split_tasks import SELDConformerModel as SELDModel
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from extract_features import SELDFeatureExtractor
import utils

# --- Loss Functions ---
def loss_bce(pred_logits, target):
    return F.binary_cross_entropy(pred_logits, target)

def loss_mse(pred, target, mask=None):
    if mask is not None:
        return (mask * (pred - target) ** 2).sum() / mask.sum()
    else:
        return F.mse_loss(pred, target)

def loss_mspe(pred, target, mask=None, eps=1e-8):
    if mask is not None:
        return (mask * ((pred - target) / (target + eps)) ** 2).sum() / mask.sum()
    else:
        return (((pred - target) / (target + eps)) ** 2).mean()

# --- Training Epoch ---
# def train_epoch(model, train_loader, optimizer, device, params):
#     model.train()
#     total_loss = 0.0
#
#     for features, labels_from_loader in tqdm(train_loader, desc="Training", leave=False):
#         features, labels_from_loader = features.to(device), labels_from_loader.to(device)
#         optimizer.zero_grad()
#
#         preds = model(features, None)
#
#         B, T_labels, DC = labels_from_loader.shape
#         C = params['nb_classes']
#         D_actual = DC // C
#
#         labels_reshaped = labels_from_loader.view(B, T_labels, D_actual, C)
#
#         x_coord_component = labels_reshaped[:, :, 0, :]
#         y_coord_component = labels_reshaped[:, :, 1, :]
#         sed_gt = (torch.sqrt(x_coord_component ** 2 + y_coord_component ** 2) > 0.5).float()
#
#         sde_gt = labels_reshaped[:, :, 2:3, :]
#
#         print(f"Shape of preds['sed']: {preds['sed'].shape}, Shape of sed_gt: {sed_gt.shape}")
#         print(f"Shape of preds['sde']: {preds['sde'].shape}, Shape of sde_gt: {sde_gt.shape}")
#
#         # If prediction time dimension is different from label time dimension,
#         # interpolation would be needed here for sed_gt and sde_gt.
#         # Example:
#         # if preds['sed'].shape[1] != sed_gt.shape[1]:
#         #     target_pred_time = preds['sed'].shape[1]
#         #     sed_gt = sed_gt.permute(0, 2, 1) # B, C, T_labels
#         #     sed_gt = F.interpolate(sed_gt, size=target_pred_time, mode='nearest')
#         #     sed_gt = sed_gt.permute(0, 2, 1) # B, target_pred_time, C
#         #
#         # if preds['sde'].shape[1] != sde_gt.shape[1]:
#         #     target_pred_time = preds['sde'].shape[1]
#         #     sde_gt = sde_gt.squeeze(2).permute(0, 2, 1) # B, C, T_labels
#         #     sde_gt = F.interpolate(sde_gt, size=target_pred_time, mode='linear', align_corners=False)
#         #     sde_gt = sde_gt.permute(0, 2, 1).unsqueeze(2) # B, target_pred_time, 1, C
#
#         sed_loss = F.binary_cross_entropy(preds['sed'], sed_gt)
#         sde_loss = (((preds['sde'] - sde_gt) / (sde_gt + 1e-8)) ** 2).mean()  # loss_mspe inline
#
#         loss = 0.1 * sed_loss + 2.0 * sde_loss
#
#         if torch.isnan(loss) or torch.isinf(loss):
#             print("NaN or Inf loss detected in training. Skipping backward pass for this batch.")
#             print(f"sed_loss: {sed_loss.item()}, sde_loss: {sde_loss.item()}")
#             print("preds['sed'] sample:", preds['sed'][0, 0, :5])
#             print("sed_gt sample:", sed_gt[0, 0, :5])
#             print("preds['sde'] sample:", preds['sde'][0, 0, 0, :5])
#             print("sde_gt sample:", sde_gt[0, 0, 0, :5])
#             continue
#
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
#     return avg_loss
#
#
#
# # --- Validation Epoch ---
# def val_epoch(model, val_loader, device, params, metrics, output_dir):
#     model.eval()
#     val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.
#
#     with torch.no_grad():
#         for batch_idx, (features, labels) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
#             features, labels = features.to(device), labels.to(device)
#             preds = model(features, None)
#
#             # output_list = [preds['sed'], preds['sde']]
#             # logits = torch.cat(output_list, dim=2)
#             active_events = preds['sed'] > 0.5
#             x_component = active_events.float() * 0.6
#             y_component = torch.zeros_like(preds['sed'])
#             distance_component = preds['sde'].squeeze(2)
#             if params['modality'] == 'audio_visual':
#                 onscreen_placeholder = torch.zeros_like(preds['sed'])
#                 logits_for_eval = torch.cat(
#                     (x_component, y_component, distance_component, onscreen_placeholder),
#                     dim=2
#                 )
#             else:
#                 logits_for_eval = torch.cat(
#                     (x_component, y_component, distance_component),
#                     dim=2
#                 )
#
#             # print("Raw preds['sde'] from model:", preds['sde'])
#
#             # print("logits_eval", logits_for_eval)
#             B, T, DC = labels.shape
#             C = params['nb_classes']
#             D = DC // C
#             labels = labels.view(B, T, D, C)
#             sed_gt = labels[:, :, 0, :]  # SED at index 0
#             sde_gt = labels[:, :, 2:3, :]
#             sed_loss_val = loss_bce(preds['sed'], sed_gt)
#             sde_loss_val = loss_mspe(preds['sde'], sde_gt)
#
#             loss = 0.1 * sed_loss_val + 2.0 * sde_loss_val
#             val_loss_per_epoch += loss
#
#             print(f"Shape of preds['sed']: {preds['sed'].shape}, Shape of sed_gt: {sed_gt.shape}")
#             print(f"Shape of preds['sde']: {preds['sde'].shape}, Shape of sde_gt: {sde_gt.shape}")
#
#             utils.write_logits_to_dcase_format(
#                 logits_for_eval, params, output_dir,
#                 val_loader.dataset.label_files[batch_idx * params['batch_size']: (batch_idx + 1) * params['batch_size']]
#             )
#     avg_val_loss = val_loss_per_epoch / len(val_loader)
#
#     metric_scores = metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'))
#     return avg_val_loss, metric_scores


def train_epoch(model, train_loader, optimizer, device, params):
    model.train()
    total_loss = 0.0
    batches_processed = 0

    for features, labels_from_loader in tqdm(train_loader, desc="Training", leave=False):
        features, labels_from_loader = features.to(device), labels_from_loader.to(device)
        optimizer.zero_grad()

        preds = model(features, None)

        B, T_labels, DC = labels_from_loader.shape
        C = params['nb_classes']
        D_actual = DC // C

        labels_reshaped = labels_from_loader.view(B, T_labels, D_actual, C)

        x_coord_component = labels_reshaped[:, :, 0, :]
        y_coord_component = labels_reshaped[:, :, 1, :]
        # sed_gt is binary (0.0 or 1.0) and has shape (B, T_labels, C)
        sed_gt = (torch.sqrt(x_coord_component ** 2 + y_coord_component ** 2) > 0.5).float()

        # sde_gt is masked distance (Activity*Distance) and has shape (B, T_labels, 1, C)
        sde_gt = labels_reshaped[:, :, 2:3, :]

        # Ensure preds and gt have same time dimension (T_preds)
        # Your prints confirmed T_labels == preds time (50), so interpolation is currently not needed.
        # If they were different, interpolation would be applied here.
        T_preds = preds['sed'].shape[1]  # Should be same as T_labels based on your prints

        # SED Loss
        sed_loss = F.binary_cross_entropy(preds['sed'], sed_gt)

        # SDE Loss (Masked MSPE)
        # active_event_mask_for_sde has shape (B, T_preds, C)
        active_event_mask_for_sde = sed_gt
        # Expand mask to match sde_gt and preds['sde'] shape: (B, T_preds, 1, C)
        active_event_mask_for_sde_expanded = active_event_mask_for_sde.unsqueeze(2)

        # Calculate MSPE only for active events
        # Add epsilon to sde_gt in the denominator for stability where sed_gt is active
        sde_gt_stable = sde_gt + 1e-8

        diff_sde = preds['sde'] - sde_gt_stable  # Using stable gt for subtraction as well
        # Element-wise squared percentage error
        squared_percentage_error_sde = (diff_sde / sde_gt_stable) ** 2

        # Apply the mask
        masked_squared_error_sde = squared_percentage_error_sde * active_event_mask_for_sde_expanded

        num_active_elements_sde = active_event_mask_for_sde_expanded.sum()

        if num_active_elements_sde > 0:
            sde_loss = masked_squared_error_sde.sum() / num_active_elements_sde
        else:
            sde_loss = torch.tensor(0.0, device=device, dtype=features.dtype)

        loss = 0.1 * sed_loss + 2.0 * sde_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss in training: sed_loss={sed_loss.item()}, sde_loss={sde_loss.item()}. Skipping batch.")
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches_processed += 1

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss


def val_epoch(model, val_loader, device, params, metrics, output_dir):
    model.eval()
    total_val_loss = 0.0
    batches_processed = 0

    # Define dummy loss_bce and loss_mspe if not globally available for val_epoch scope
    # This is just for the loss calculation within val_epoch, not for backprop
    def bce_for_val(pred_logits, target):
        return F.binary_cross_entropy(pred_logits, target, reduction='mean')

    def mspe_for_val(pred, target, active_mask_expanded, eps=1e-8):
        target_stable = target + eps
        diff = pred - target_stable
        spe = (diff / target_stable) ** 2
        masked_spe = spe * active_mask_expanded
        num_active = active_mask_expanded.sum()
        if num_active > 0:
            return masked_spe.sum() / num_active
        return torch.tensor(0.0, device=device, dtype=pred.dtype)

    with torch.no_grad():
        for batch_idx, (features, labels_from_loader) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            features, labels_from_loader = features.to(device), labels_from_loader.to(device)

            preds = model(features, None)

            # Prepare logits_for_eval for DCASE format output
            # This uses model's direct SED probabilities and SDE predictions
            active_events_from_pred = preds['sed'] > 0.5
            x_component_for_eval = active_events_from_pred.float() * 0.6
            y_component_for_eval = torch.zeros_like(preds['sed'])
            distance_component_for_eval = preds['sde'].squeeze(2)

            if params['modality'] == 'audio_visual':
                onscreen_placeholder = torch.zeros_like(preds['sed'])
                logits_for_eval = torch.cat(
                    (x_component_for_eval, y_component_for_eval, distance_component_for_eval, onscreen_placeholder),
                    dim=2
                )
            else:
                logits_for_eval = torch.cat(
                    (x_component_for_eval, y_component_for_eval, distance_component_for_eval),
                    dim=2
                )

            # Unpack labels for validation loss calculation (consistent with train_epoch)
            B, T_labels, DC = labels_from_loader.shape
            C = params['nb_classes']
            D_actual = DC // C
            labels_reshaped = labels_from_loader.view(B, T_labels, D_actual, C)

            x_coord_gt = labels_reshaped[:, :, 0, :]
            y_coord_gt = labels_reshaped[:, :, 1, :]
            sed_gt_val = (torch.sqrt(x_coord_gt ** 2 + y_coord_gt ** 2) > 0.5).float()
            sde_gt_val = labels_reshaped[:, :, 2:3, :]

            # Ensure time dimensions match for val loss (should be same based on your prints)
            # If not, interpolation would be needed here too.

            # Calculate validation loss (consistent with training)
            sed_loss_val = bce_for_val(preds['sed'], sed_gt_val)

            active_mask_val_expanded = sed_gt_val.unsqueeze(2)
            sde_loss_val = mspe_for_val(preds['sde'], sde_gt_val, active_mask_val_expanded)

            current_loss = 0.1 * sed_loss_val + 2.0 * sde_loss_val

            if not (torch.isnan(current_loss) or torch.isinf(current_loss)):
                total_val_loss += current_loss.item()
                batches_processed += 1
            else:
                print(
                    f"NaN/Inf loss in validation: sed_loss={sed_loss_val.item()}, sde_loss={sde_loss_val.item()}. Skipping batch loss.")

            utils.write_logits_to_dcase_format(
                logits_for_eval, params, output_dir,
                val_loader.dataset.label_files[batch_idx * params['batch_size']: (batch_idx + 1) * params['batch_size']]
            )

    avg_val_loss = total_val_loss / batches_processed if batches_processed > 0 else float('nan')
    metric_scores = metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'))
    return avg_val_loss, metric_scores

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train SELD Model for sed-doa task")
    if restore_from_checkpoint:
        print('Loading params from the initial checkpoint')
        params_file = os.path.join(initial_checkpoint_path, 'config.pkl')
        f = open(params_file, "rb")
        loaded_params = pickle.load(f)
        params.clear()  # Clear the original params
        params.update(loaded_params)

    parser.add_argument('--save_suffix', type=str, default='sed_sde_model',
                        help='Suffix for saving checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    params['task'] = 'sed-sde'
    params['multiACCDOA'] = False
    reference = f"{params['net_type']}_{params['modality']}_sed-sde_{args.save_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    params['run_reference_name'] = reference

    checkpoints_folder, output_dir, _ = utils.setup(params)

    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    train_loader = DataLoader(
        dataset=DataGenerator(params=params, mode='dev_train'),
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        num_workers=params['nb_workers'],
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=DataGenerator(params=params, mode='dev_test'),
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['nb_workers'],
        drop_last=False
    )

    model = SELDModel(params=params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    best_f_score = float('-inf')

    if restore_from_checkpoint:
        print('Loading model weights and optimizer state dict from initial checkpoint...')
        model_ckpt = torch.load(os.path.join(initial_checkpoint_path, 'best_model.pth'), map_location=device, weights_only=False)
        model.load_state_dict(model_ckpt['seld_model'])
        optimizer.load_state_dict(model_ckpt['opt'])
        start_epoch = model_ckpt['epoch'] + 1
        best_f_score = model_ckpt['best_f_score']

    # for epoch in range(params['nb_epochs']):
    #     avg_train_loss = train_epoch(model, train_loader, optimizer, device, params)
    #     avg_val_loss, metric_scores = val_epoch(model, val_loader, device, params, metrics, output_dir)
    #
    #     val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
    #
    #     print(
    #         f"Epoch {epoch + 1}/{params['nb_epochs']} | "
    #         f"Train Loss: {avg_train_loss:.2f} | "
    #         f"Val Loss: {avg_val_loss:.2f} | "
    #         f"F-score: {val_f * 100:.2f} | "
    #         f"Ang Err: {val_ang_error:.2f} | "
    #         f"Dist Err: {val_dist_error:.2f} | "
    #         f"Rel Dist Err: {val_rel_dist_error:.2f}" +
    #         (f" | On-Screen Acc: {val_onscreen_acc:.2f}" if params['modality'] == 'audio_visual' else "")
    #     )
    #
    #     if val_f >= best_f_score:
    #         best_f_score = val_f
    #         save_path = os.path.join(checkpoints_folder, f"best_model_{args.save_suffix}.pth")
    #         torch.save({
    #             'seld_model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'params': params,
    #             'best_f_score': best_f_score
    #         }, save_path)
    #         print(f"Saved best model checkpoint to {save_path}")

    # Evaluate the best model on dev-test.
    # best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model_sed_doa_model.pth'), map_location=device,
    #                              weights_only=False)
    best_model_ckpt = torch.load(os.path.join('checkpoints/sed_sde_train1', 'best_model_sed_sde_model.pth'), map_location=device,
                                 weights_only=False)
    model.load_state_dict(best_model_ckpt['seld_model'])
    use_jackknife = params['use_jackknife']
    test_loss, test_metric_scores = val_epoch(model, val_loader, device, params, metrics, output_dir)
    test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    restore_from_checkpoint = False
    initial_checkpoint_path = 'checkpoints/SELDnet_audio_visual_multiACCDOA_20250331_173131'


    main()