import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pickle
import wandb
import time
import numpy as np

from cst_conformer.parameters_optimised import params
from cst_conformer.model_seld_optimized import SeldOptimizedConformer
from loss import SELDLossADPIT, SELDLossSingleACCDOA, SELDLossWeightedADPIT # Add import
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from extract_features import SELDFeatureExtractor
import utils # Make sure this utils.py is the one from your working pipeline

# Global device variable, to be set in main
device = None

def train_epoch(model, train_iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for i, (input_features, labels) in enumerate(train_iterator):
        optimizer.zero_grad()
        audio_features = input_features.to(device)
        target_labels = labels.to(device)

        logits = model(audio_features)
        loss = criterion(logits, target_labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_iterator)

def validate_epoch(model, val_iterator, criterion, seld_metrics_calculator, epoch_output_dir, is_eval_final=False):
    model.eval()
    epoch_loss = 0.0

    # Create a split-specific directory within the epoch_output_dir for predictions
    # Assuming 'dev-test' is the split name used by DataGenerator for validation.
    # utils.write_logits_to_dcase_format will use this split name to create the subdir.
    # The split name should ideally come from params or DataGenerator if it varies.
    # For now, hardcoding 'dev-test' as it's typical for validation splits in DCASE.
    current_split_pred_dir = os.path.join(epoch_output_dir, params.get('val_split_name', 'dev-test'))
    os.makedirs(current_split_pred_dir, exist_ok=True)

    # Clean previous predictions for this specific split if not final evaluation
    if os.path.exists(current_split_pred_dir) and not is_eval_final:
        for f_name in os.listdir(current_split_pred_dir):
            if f_name.endswith('.csv'):
                try:
                    os.remove(os.path.join(current_split_pred_dir, f_name))
                except OSError as e:
                    print(f"Error removing file {f_name} from {current_split_pred_dir}: {e}")

    with torch.no_grad():
        for i, (input_features, labels) in enumerate(val_iterator):
            audio_features = input_features.to(device)
            target_labels = labels.to(device)

            logits = model(audio_features)
            loss = criterion(logits, target_labels)
            epoch_loss += loss.item()

            start_idx = i * params['batch_size']
            end_idx = start_idx + audio_features.shape[0]
            current_batch_label_files = val_iterator.dataset.label_files[start_idx:end_idx]

            utils.write_logits_to_dcase_format(
                logits.cpu(),
                params,
                epoch_output_dir, # Base directory for predictions for this epoch
                current_batch_label_files,
                split=params.get('val_split_name', 'dev-test') # Pass the split name
            )

    avg_epoch_loss = epoch_loss / len(val_iterator)

    # Metrics calculator now points to the directory containing CSVs for the current split
    if not os.path.exists(current_split_pred_dir) or not os.listdir(current_split_pred_dir):
        print(f"Warning: Prediction CSV path {current_split_pred_dir} is empty or does not exist. Returning NaN metrics.")
        num_classes = params.get('nb_classes', 13)
        empty_classwise = [np.full(num_classes, np.nan) for _ in range(4)] # F, AngE, DistE, RelDistE
        if params.get('modality') == 'audio_visual':
             empty_classwise.append(np.full(num_classes, np.nan)) # OnscreenAcc
        metric_scores_tuple = (np.nan, np.nan, np.nan, np.nan,
                               np.nan if params.get('modality') == 'audio_visual' else 0.0,
                               np.array(empty_classwise))
    else:
        metric_scores_tuple = seld_metrics_calculator.get_SELD_Results(
            pred_files_path=current_split_pred_dir, # Path to CSVs for this split
            is_jackknife=params['use_jackknife'] if is_eval_final else False
        )
    return avg_epoch_loss, metric_scores_tuple


def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    restore_from_checkpoint = False # Set to True and specify path if needed
    # initial_checkpoint_path = 'checkpoints_optimized/your_checkpoint_run_name' # Example

    if restore_from_checkpoint:
        print('Loading params from the initial checkpoint')
        # params_file = os.path.join(initial_checkpoint_path, 'config.pkl')
        # This part needs careful handling if params are truly restored.
        # For now, we assume params from parameters_optimised.py are used.
        pass # Simplified for now

    # utils.setup now returns: checkpoints_folder, output_dir_for_this_run, summary_writer
    checkpoints_folder, output_dir_for_this_run, summary_writer = utils.setup(params)

    # val_output_dir_temp is where epoch-wise predictions will be stored (inside output_dir_for_this_run)
    # This is effectively the same as output_dir_for_this_run from utils.setup
    val_output_dir_temp = output_dir_for_this_run

    wandb.init(
        project=params.get("wandb_project", "dcase2025-seld-optimized"),
        name=params.get("wandb_run_name", f"{params['net_type']}_{time.strftime('%Y%m%d_%H%M%S')}"),
        config=params
    )

    print("Starting feature extraction...")
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')
    print("Feature extraction completed.")

    print("Setting up data loaders...")
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    train_loader = DataLoader(
        dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'],
        shuffle=params['shuffle'], drop_last=True, pin_memory=(device.type == 'cuda')
    )
    dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    val_loader = DataLoader(
        dataset=dev_test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'],
        shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda')
    )
    print("Data loaders ready.")

    print("Initializing model, criterion, and optimizer...")
    seld_model = SeldOptimizedConformer(params=params).to(device)
    wandb.watch(seld_model, log="all", log_freq=100)

    if params['multiACCDOA']:
        criterion = SELDLossWeightedADPIT(params=params).to(device)
    else:
        criterion = SELDLossSingleACCDOA(params=params).to(device)

    optimizer = torch.optim.Adam(
        params=seld_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay']
    )

    # ref_gt_folder_val is usually the 'metadata_dev' inside your root DCASE dataset directory
    # metrics.py's ComputeSELDResults iterates through subfolders (splits) within this.
    ref_gt_folder_val = os.path.join(params['root_dir'], 'metadata_dev')
    if not os.path.exists(ref_gt_folder_val):
        print(f"Warning: Reference GT folder for validation not found at {ref_gt_folder_val}. Metrics might fail.")

    seld_metrics_calc = ComputeSELDResults(params=params, ref_files_folder=ref_gt_folder_val)
    print("Model setup complete.")

    start_epoch = 0
    best_f_score = float('-inf')

    if restore_from_checkpoint and os.path.exists(os.path.join(initial_checkpoint_path, 'best_model.pth')):
        print('Loading model weights and optimizer state dict from initial checkpoint...')
        # model_ckpt = torch.load(os.path.join(initial_checkpoint_path, 'best_model.pth'), map_location=device)
        # seld_model.load_state_dict(model_ckpt['model_state_dict']) # Use 'seld_model' if that's the key
        # optimizer.load_state_dict(model_ckpt['opt'])
        # start_epoch = model_ckpt['epoch'] + 1
        # best_f_score = model_ckpt['best_f_score']
        pass # Simplified: Implement full checkpoint loading if needed, matching keys from original main.py

    print("Starting training loop...")
    for epoch in range(start_epoch, params['nb_epochs']):
        avg_train_loss = train_epoch(seld_model, train_loader, optimizer, criterion)

        avg_val_loss, metric_scores = validate_epoch(
            seld_model, val_loader, criterion, seld_metrics_calc, val_output_dir_temp
        )

        val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, _ = metric_scores
        best_f_score = float('-inf')

        summary_writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        summary_writer.add_scalar('Loss/Validation_Epoch', avg_val_loss, epoch)
        summary_writer.add_scalar('Metric/F1_Score', val_f * 100 if not np.isnan(val_f) else 0, epoch)
        summary_writer.add_scalar('Metric/DOA_Error', val_ang_error if not np.isnan(val_ang_error) else 180, epoch)
        summary_writer.add_scalar('Metric/Distance_Error', val_dist_error if not np.isnan(val_dist_error) else 100, epoch)
        summary_writer.add_scalar('Metric/Relative_Distance_Error', val_rel_dist_error if not np.isnan(val_rel_dist_error) else 1.0, epoch)
        if params['modality'] == 'audio_visual':
             summary_writer.add_scalar('Metric/Onscreen_Accuracy', val_onscreen_acc * 100 if not np.isnan(val_onscreen_acc) else 0, epoch)

        wandb_log_dict = {
            "epoch": epoch + 1, "train_loss_epoch": avg_train_loss, "val_loss_epoch": avg_val_loss,
            "val_f1_score": val_f * 100 if not np.isnan(val_f) else 0,
            "val_doa_error": val_ang_error if not np.isnan(val_ang_error) else 180,
            "val_distance_error": val_dist_error if not np.isnan(val_dist_error) else 100,
            "val_relative_distance_error": val_rel_dist_error if not np.isnan(val_rel_dist_error) else 1.0
        }
        if params['modality'] == 'audio_visual':
            wandb_log_dict["val_onscreen_accuracy"] = val_onscreen_acc * 100 if not np.isnan(val_onscreen_acc) else 0
        wandb.log(wandb_log_dict)

        print(time.strftime('%Y-%m-%d %H:%M:%S'))
        print(
            f"Epoch {epoch + 1}/{params['nb_epochs']} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | "
            f"F-score: {val_f * 100:.2f} | Ang Err: {val_ang_error:.2f} | Dist Err: {val_dist_error:.2f} | "
            f"Rel Dist Err: {val_rel_dist_error:.2f}" +
            (f" | On-Screen Acc: {val_onscreen_acc * 100:.2f}" if params['modality'] == 'audio_visual' and not np.isnan(val_onscreen_acc) else "")
        )
        print(f"TensorBoard writing to: {summary_writer.log_dir}")
        try:
            summary_writer.flush()
        except Exception as e:
            print(f"TensorBoard flush error: {e}")

        if not np.isnan(val_f) and val_f >= best_f_score:
            best_f_score = val_f
            checkpoint_path = os.path.join(checkpoints_folder, "best_model.pth")
            net_save_dict = {
                'epoch': epoch, 'model_state_dict': seld_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_f_score': best_f_score,
                'best_val_angular_error': val_ang_error, 'best_val_distance_error': val_dist_error,
                'best_val_relative_distance_error': val_rel_dist_error, 'params': params
            }
            if params['modality'] == 'audio_visual':
                net_save_dict['best_val_onscreen_accuracy'] = val_onscreen_acc
            torch.save(net_save_dict, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} (Best F1: {best_f_score*100:.2f}%)")
            try:
                wandb.save(checkpoint_path)
            except Exception as e:
                print(f"Wandb save error: {e}")

    print("Training finished.")

    print("Loading best model for final evaluation on test set...")
    best_model_path = os.path.join(checkpoints_folder, "best_model.pth")
    if os.path.exists(best_model_path):
        # FIX 1: Added weights_only=False to torch.load
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

        seld_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} located at {best_model_path}")

        final_eval_output_dir = os.path.join(output_dir_for_this_run, "final_eval_preds")
        os.makedirs(final_eval_output_dir, exist_ok=True)

        # FIX 2: Removed the extra 'device' argument from the function call
        _, final_metrics_tuple = validate_epoch(
            seld_model, val_loader, criterion, seld_metrics_calc, final_eval_output_dir, is_eval_final=True
        )
        print("\n--- Final Evaluation Results on Dev-Test Set ---")
        utils.print_results(*final_metrics_tuple, params=params)

        wandb_final_log = {
            "final_test_f1_score": final_metrics_tuple[0] * 100 if not np.isnan(final_metrics_tuple[0]) else 0,
            "final_test_doa_error": final_metrics_tuple[1] if not np.isnan(final_metrics_tuple[1]) else 180,
            "final_test_distance_error": final_metrics_tuple[2] if not np.isnan(final_metrics_tuple[2]) else 100,
            "final_test_relative_distance_error": final_metrics_tuple[3] if not np.isnan(
                final_metrics_tuple[3]) else 1.0,
        }
        if params['modality'] == 'audio_visual' and len(final_metrics_tuple) > 4:
            wandb_final_log["final_test_onscreen_accuracy"] = final_metrics_tuple[4] * 100 if not np.isnan(
                final_metrics_tuple[4]) else 0
        wandb.log(wandb_final_log)
    else:
        print(f"Best model checkpoint not found at {best_model_path} for final evaluation.")

    wandb.finish()
    summary_writer.close()
    print("Process completed.")


if __name__ == '__main__':
    main()