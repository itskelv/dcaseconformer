"""
main_sed_doa_sde.py

Train and evaluate a SELD model for the SED-DOA-SDE task.
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from parameters import params
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

# --- Training Epoch ---
def train_epoch(model, train_loader, optimizer, device, params):
    model.train()
    total_loss = 0.0

    for features, labels in tqdm(train_loader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        preds = model(features, None)

        # Unpack labels (assuming [B, T, D*C])
        B, T, DC = labels.shape
        C = params['nb_classes']
        D = DC // C
        labels = labels.view(B, T, D, C)
        # print("labels_shape_after:", labels.shape)


        # sed_gt = labels[:, :, 0, :]      # SED at index 0
        x_coord_component = labels[:, :, 0, :]
        y_coord_component = labels[:, :, 1, :]
        # sed_gt is binary (0.0 or 1.0) and has shape (B, T_labels, C)
        sed_gt = (torch.sqrt(x_coord_component ** 2 + y_coord_component ** 2) > 0.5).float()
        doa_gt = torch.stack([labels[:, :, 0, :], labels[:, :, 1, :]], dim=2)  # (x,y) at index 1 and 2
        sde_gt = labels[:, :, 2:3, :]

        active_event_mask_for_sde = sed_gt
        active_event_mask_for_sde_expanded = active_event_mask_for_sde.unsqueeze(2)
        sde_gt_stable = sde_gt + 1e-8
        diff_sde = preds['sde'] - sde_gt_stable  # Using stable gt for subtraction as well
        squared_percentage_error_sde = (diff_sde / sde_gt_stable) ** 2
        masked_squared_error_sde = squared_percentage_error_sde * active_event_mask_for_sde_expanded
        num_active_elements_sde = active_event_mask_for_sde_expanded.sum()

        if num_active_elements_sde > 0:
            sde_loss = masked_squared_error_sde.sum() / num_active_elements_sde
        else:
            sde_loss = torch.tensor(0.0, device=device, dtype=features.dtype)

        # Compute Loss
        sed_loss = loss_bce(preds['sed'], sed_gt)
        doa_loss = loss_mse(preds['doa'], doa_gt)

        loss = 0.1 * sed_loss + 1.0 * doa_loss + 2.0 * sde_loss


        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss in training: sed_loss={sed_loss.item()}, sde_loss={sde_loss.item()}. Skipping batch.")
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# --- Validation Epoch ---
def val_epoch(model, val_loader, device, params, metrics, output_dir):
    model.eval()
    val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.


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
        for batch_idx, (features, labels) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            features, labels = features.to(device), labels.to(device)
            preds = model(features, None)

            output_list = [preds['doa'].reshape(preds['doa'].size(0), preds['doa'].size(1), -1), preds['sde'].squeeze(2)]
            logits = torch.cat(output_list, dim=2)
            B, T, DC = labels.shape
            C = params['nb_classes']
            D = DC // C
            labels = labels.view(B, T, D, C)
            # sed_gt = labels[:, :, 0, :]  # SED at index 0
            x_coord_component = labels[:, :, 0, :]
            y_coord_component = labels[:, :, 1, :]
            # sed_gt is binary (0.0 or 1.0) and has shape (B, T_labels, C)
            sed_gt = (torch.sqrt(x_coord_component ** 2 + y_coord_component ** 2) > 0.5).float()
            doa_gt = torch.stack([labels[:, :, 0, :], labels[:, :, 1, :]], dim=2)  # (x,y) at index 1 and 2
            sde_gt_val = labels[:, :, 2:3, :]

            sed_loss_val = bce_for_val(preds['sed'], sed_gt)
            doa_loss = loss_mse(preds['doa'], doa_gt)
            active_mask_val_expanded = sed_gt.unsqueeze(2)
            sde_loss_val = mspe_for_val(preds['sde'], sde_gt_val, active_mask_val_expanded)


            loss = 0.1 * sed_loss_val + 1.0 * doa_loss + 2.0 * sde_loss_val
            val_loss_per_epoch += loss
            utils.write_logits_to_dcase_format(
                logits, params, output_dir,
                val_loader.dataset.label_files[batch_idx * params['batch_size']: (batch_idx + 1) * params['batch_size']]
            )
    avg_val_loss = val_loss_per_epoch / len(val_loader)

    metric_scores = metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'))
    return avg_val_loss, metric_scores


# def loss_bce(pred_logits, target):
#     return F.binary_cross_entropy(pred_logits, target)
#
#
# def loss_mse(pred, target):  # Assuming no mask needed if applied via sed_gt for active events
#     return F.mse_loss(pred, target)
#
#
# def train_epoch(model, train_loader, optimizer, device, params):
#     model.train()
#     total_loss = 0.0
#     batches_processed = 0
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
#         labels_reshaped = labels_from_loader.view(B, T_labels, D_actual, C)
#
#         x_coord_gt_masked = labels_reshaped[:, :, 0, :]
#         y_coord_gt_masked = labels_reshaped[:, :, 1, :]
#         sed_gt = (torch.sqrt(x_coord_gt_masked ** 2 + y_coord_gt_masked ** 2) > 0.5).float()
#
#
#         doa_gt = torch.stack((x_coord_gt_masked, y_coord_gt_masked), dim=2)
#
#         sde_gt = labels_reshaped[:, :, 2:3, :]
#
#         sed_loss = loss_bce(preds['sed'], sed_gt)
#
#         active_event_mask_doa = sed_gt.unsqueeze(2).expand_as(preds['doa'])
#         # masked_doa_preds = preds['doa'] * active_event_mask_doa
#         masked_doa_preds = preds['doa']
#         # masked_doa_gt = doa_gt * active_event_mask_doa
#         masked_doa_gt = doa_gt
#         if active_event_mask_doa.sum() > 0:
#             doa_loss = loss_mse(masked_doa_preds, masked_doa_gt)
#         else:
#             doa_loss = torch.tensor(0.0, device=device, dtype=features.dtype)
#
#         active_event_mask_sde_expanded = sed_gt.unsqueeze(2)
#         sde_gt_stable = sde_gt + 1e-8
#         diff_sde = preds['sde'] - sde_gt_stable
#         squared_percentage_error_sde = (diff_sde / sde_gt_stable) ** 2
#         masked_squared_error_sde = squared_percentage_error_sde * active_event_mask_sde_expanded
#         num_active_elements_sde = active_event_mask_sde_expanded.sum()
#         if num_active_elements_sde > 0:
#             sde_loss = masked_squared_error_sde.sum() / num_active_elements_sde
#         else:
#             sde_loss = torch.tensor(0.0, device=device, dtype=features.dtype)
#
#         loss = 0.1 * sed_loss + 1.0 * doa_loss + 2.0 * sde_loss
#
#         if torch.isnan(loss) or torch.isinf(loss):
#             print(f"NaN/Inf loss: sed={sed_loss.item()}, doa={doa_loss.item()}, sde={sde_loss.item()}. Skip.")
#             continue
#
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         batches_processed += 1
#
#     avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
#     return avg_loss
#
#
# def val_epoch(model, val_loader, device, params, metrics, output_dir):
#     model.eval()
#     total_val_loss = 0.0
#     batches_processed = 0
#
#     def bce_for_val(pred_logits, target):
#         return F.binary_cross_entropy(pred_logits, target, reduction='mean')
#
#     def mse_for_val(pred, target, active_mask):
#         masked_pred = pred * active_mask
#         masked_target = target * active_mask
#         if active_mask.sum() > 0:
#             return F.mse_loss(masked_pred, masked_target, reduction='sum') / active_mask.sum()
#         return torch.tensor(0.0, device=device, dtype=pred.dtype)
#
#     def mspe_for_val(pred, target, active_mask_expanded, eps=1e-8):
#         target_stable = target + eps
#         diff = pred - target_stable
#         spe = (diff / target_stable) ** 2
#         masked_spe = spe * active_mask_expanded
#         num_active = active_mask_expanded.sum()
#         if num_active > 0:
#             return masked_spe.sum() / num_active
#         return torch.tensor(0.0, device=device, dtype=pred.dtype)
#
#     with torch.no_grad():
#         for batch_idx, (features, labels_from_loader) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
#             features, labels_from_loader = features.to(device), labels_from_loader.to(device)
#             preds = model(features, None)
#
#             pred_x_for_eval = preds['doa'][:, :, 0, :]
#             pred_y_for_eval = preds['doa'][:, :, 1, :]
#             pred_dist_for_eval = preds['sde'].squeeze(2)
#
#             output_list = [preds['doa'].reshape(preds['doa'].size(0), preds['doa'].size(1), -1), preds['sde'].squeeze(2)]
#             logits = torch.cat(output_list, dim=2)
#
#
#             if params['modality'] == 'audio_visual':
#                 onscreen_placeholder = torch.zeros_like(preds['sed'])
#                 logits_for_eval = torch.cat(
#                     (pred_x_for_eval, pred_y_for_eval, pred_dist_for_eval, onscreen_placeholder),
#                     dim=2
#                 )
#             else:
#                 logits_for_eval = torch.cat(
#                     (pred_x_for_eval, pred_y_for_eval, pred_dist_for_eval),
#                     dim=2
#                 )
#
#             # print("eval_logits", logits_for_eval)
#             B, T_labels, DC = labels_from_loader.shape
#             C = params['nb_classes']
#             D_actual = DC // C
#             labels_reshaped = labels_from_loader.view(B, T_labels, D_actual, C)
#
#             x_coord_gt_masked_val = labels_reshaped[:, :, 0, :]
#             y_coord_gt_masked_val = labels_reshaped[:, :, 1, :]
#             sed_gt_val = (torch.sqrt(x_coord_gt_masked_val ** 2 + y_coord_gt_masked_val ** 2) > 0.5).float()
#             doa_gt_val = torch.stack((x_coord_gt_masked_val, y_coord_gt_masked_val), dim=2)
#             sde_gt_val = labels_reshaped[:, :, 2:3, :]
#
#             sed_loss_val = bce_for_val(preds['sed'], sed_gt_val)
#
#             active_mask_doa_val = sed_gt_val.unsqueeze(2).expand_as(preds['doa'])
#             doa_loss_val = mse_for_val(preds['doa'], doa_gt_val, active_mask_doa_val)
#
#             active_mask_sde_val_expanded = sed_gt_val.unsqueeze(2)
#             sde_loss_val = mspe_for_val(preds['sde'], sde_gt_val, active_mask_sde_val_expanded)
#
#             current_loss = 0.1 * sed_loss_val + 1.0 * doa_loss_val + 2.0 * sde_loss_val
#
#             if not (torch.isnan(current_loss) or torch.isinf(current_loss)):
#                 total_val_loss += current_loss.item()
#                 batches_processed += 1
#             else:
#                 print(
#                     f"NaN/Inf val_loss: sed={sed_loss_val.item()}, doa={doa_loss_val.item()}, sde={sde_loss_val.item()}. Skip batch.")
#
#             utils.write_logits_to_dcase_format(
#                 logits, params, output_dir,
#                 val_loader.dataset.label_files[batch_idx * params['batch_size']: (batch_idx + 1) * params['batch_size']]
#             )
#
#     avg_val_loss = total_val_loss / batches_processed if batches_processed > 0 else float('nan')
#     metric_scores = metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'))
#     return avg_val_loss, metric_scores


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

    parser.add_argument('--save_suffix', type=str, default='sed_doa_sde_model',
                        help='Suffix for saving checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    params['task'] = 'sed-doa-sde'
    params['multiACCDOA'] = False
    reference = f"{params['net_type']}_{params['modality']}_sed-doa_sde_{args.save_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
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

    for epoch in range(params['nb_epochs']):
        avg_train_loss = train_epoch(model, train_loader, optimizer, device, params)
        avg_val_loss, metric_scores = val_epoch(model, val_loader, device, params, metrics, output_dir)

        val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores

        print(
            f"Epoch {epoch + 1}/{params['nb_epochs']} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"F-score: {val_f * 100:.2f} | "
            f"Ang Err: {val_ang_error:.2f} | "
            f"Dist Err: {val_dist_error:.2f} | "
            f"Rel Dist Err: {val_rel_dist_error:.2f}" +
            (f" | On-Screen Acc: {val_onscreen_acc:.2f}" if params['modality'] == 'audio_visual' else "")
        )

        if val_f >= best_f_score:
            best_f_score = val_f
            save_path = os.path.join(checkpoints_folder, f"best_model_{args.save_suffix}.pth")
            torch.save({
                'seld_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'params': params,
                'best_f_score': best_f_score
            }, save_path)
            print(f"Saved best model checkpoint to {save_path}")

    # Evaluate the best model on dev-test.
    best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model_sed_doa_sde_model.pth'), map_location=device,
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