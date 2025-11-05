import torch
from parameters import params
from model_conformer_for_split_tasks import SELDConformerModel as SELDModel
import os
from parameters import params
import utils
from main_sed_doa import  val_epoch
from torch.utils.data import DataLoader
from data_generator import DataGenerator


# Set task explicitly
params['task'] = 'sed-doa'   # We are inspecting only for sed-doa now
params['multiACCDOA'] = False  # Make sure this matches
params['modality'] = 'audio'  # Audio only

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SELDModel(params=params).to(device)
model.eval()

# Create dummy input
batch_size = 4
time_steps = 50   # (could be any number depending on your hop_size and sequence)
input_tensor = torch.randn(batch_size, time_steps, 2, 64).to(device)
input_tensor = input_tensor.permute(0, 2, 1, 3)  # → (B, 2, T, 64)

# Forward pass
with torch.no_grad():
    preds = model(input_tensor, None)  # video_input is None for 'audio' modality

# Inspect output
print("=== Model Output Shapes ===")
print(f"Input Shape: {input_tensor.shape}")
print(f"SED logits shape: {preds['sed'].shape}")  # Expecting [B, T', 13]
print(f"DOA predictions shape: {preds['doa'].shape}")  # Expecting [B, T', 2, 13]
print(f"SDE (should be None): {preds['sde']}")  # For sed-doa task, sde is None

# val_loader = DataLoader(
#         dataset=DataGenerator(params=params, mode='dev_test'),
#         batch_size=params['batch_size'],
#         shuffle=False,
#         num_workers=params['nb_workers'],
#         drop_last=False
#     )
#
# checkpoints_folder, output_dir, _ = utils.setup(params)
#
# best_model_ckpt = torch.load(os.path.join('checkpoints/SELDnet_audio_sed-doa_sed_doa_model_20250428_155200', 'best_model_sed_doa_model.pth'), map_location=device,
#                                  weights_only=False)
# seld_model = SELDModel(params=params).to(device)
#
#
# seld_model.load_state_dict(best_model_ckpt['seld_model'])
# use_jackknife = params['use_jackknife']
# test_loss, test_metric_scores = val_epoch(seld_model, val_loader, seld_loss, seld_metrics, output_dir,
#                                           is_jackknife=use_jackknife)
# test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
# utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)
