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

from parameters_sed_doa import params
from model_conformer_for_split_tasks import SELDConformerModel as SELDModel
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from extract_features import SELDFeatureExtractor
import wandb

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

        # Compute Loss
        sed_loss = loss_bce(preds['sed'], sed_gt)
        doa_loss = loss_mse(preds['doa'], doa_gt)

        loss = 0.1 * sed_loss + 1.0 * doa_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# --- Validation Epoch ---
def val_epoch(model, val_loader, device, params, metrics, output_dir):
    model.eval()
    val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            features, labels = features.to(device), labels.to(device)
            preds = model(features, None)

            output_list = [preds['doa'].reshape(preds['doa'].size(0), preds['doa'].size(1), -1), torch.zeros_like(preds['sed'])]
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
            sed_loss = loss_bce(preds['sed'], sed_gt)
            doa_loss = loss_mse(preds['doa'], doa_gt)

            loss = 0.1 * sed_loss + 1.0 * doa_loss
            val_loss_per_epoch += loss
            utils.write_logits_to_dcase_format(
                logits, params, output_dir,
                val_loader.dataset.label_files[batch_idx * params['batch_size']: (batch_idx + 1) * params['batch_size']]
            )
    avg_val_loss = val_loss_per_epoch / len(val_loader)

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

    parser.add_argument('--save_suffix', type=str, default='sed_doa_model',
                        help='Suffix for saving checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    params['task'] = 'sed-doa'
    params['multiACCDOA'] = False
    reference = f"{params['net_type']}_{params['modality']}_sed-doa_{args.save_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    params['run_reference_name'] = reference

    checkpoints_folder, output_dir, summary_writer = utils.setup(params)

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

    wandb.init(
        project="seld_sed_doa",
        name = reference,
        config = {
            "learning_rate": params['learning_rate'],
            "architecture": "ConformerBlocks",
            "dataset": "SynthDataset + Dev-train-tau",
            "eval_dataset": "dev-test-sony",
            "epochs": 200,
        },
    )
    wandb.config.update(params)
    wandb.watch(model, log="all", log_freq=100)

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

        wandb.log({
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "val_f1": val_f * 100,
            "val_ang_error": val_ang_error
        })

        summary_writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        summary_writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        summary_writer.add_scalar('Metric/F1', val_f * 100, epoch)
        summary_writer.add_scalar('Metric/DOA_Error', val_ang_error, epoch)
        print(f"TensorBoard writing to: {summary_writer.log_dir}")

        try:
            summary_writer.flush()
        except Exception as e:
            print(f"Flush error: {e}")

        print(
            f"Epoch {epoch + 1}/{params['nb_epochs']} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"F-score: {val_f * 100:.2f} | "
            f"Ang Err: {val_ang_error:.2f} | " +
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
    best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model_sed_doa_model.pth'), map_location=device,
                                 weights_only=False)
    model.load_state_dict(best_model_ckpt['seld_model'])
    use_jackknife = params['use_jackknife']
    test_loss, test_metric_scores = val_epoch(model, val_loader, device, params, metrics, output_dir)
    test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    summary_writer.close()
    wandb.finish()

    utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    restore_from_checkpoint = False
    initial_checkpoint_path = 'checkpoints/SELDnet_audio_visual_multiACCDOA_20250331_173131'


    main()