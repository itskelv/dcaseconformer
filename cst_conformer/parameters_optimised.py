"""
parameters_optimized.py

This module stores all the configurable parameters and hyperparameters for the
optimized SELD model, incorporating CST-former inspired components.
Adapted for DCASE2025 stereo input.
"""

import numpy as np

params = {
    # Task and Model Type
    'task': 'sed-doa-sde',  # Sound Event Detection, Direction of Arrival, Sound Distance Estimation
    'modality': 'audio',  # Audio-only
    'net_type': 'SELD_Optimized_Conformer',  # Name for the new model architecture

    # Data directories
    'root_dir': '../DCASE2025_SELD_dataset',  # Main directory for DCASE2025 dataset
    'feat_dir': '../DCASE2025_SELD_dataset/features',  # Directory to store extracted features

    # Output directories
    'log_dir': 'logs_optimized',  # Directory for TensorBoard logs, etc.
    'checkpoints_dir': 'checkpoints_optimized',  # Directory to save model checkpoints
    'output_dir': 'outputs_optimized',  # Directory to save prediction files

    # Audio feature extraction parameters
    'sampling_rate': 24000,
    'hop_length_s': 0.02,  # Corresponds to 100 frames per second if frame_len is also 0.02s
    'nb_mels': 64,  # Number of Mel frequency bins
    'audio_input_channels': 2,  # Stereo input

    # Label and sequence parameters
    'max_polyphony': 3,  # Maximum number of overlapping sound events per class to detect for Multi-ACCDOA
    'nb_classes': 13,  # Number of unique sound event classes (adjust if DCASE2025 has different)
    # Check DCASE2025 for actual number of classes
    'label_sequence_length': 50,
    # Number of frames in one label sequence (e.g., 50 frames * 0.02s/frame_hop = 1s effective label window)
    # This might need adjustment based on how features/data are chunked.
    # The DCASE baseline often uses 50 frames for a 1s chunk with 100ms frame hop (meaning 5s audio).
    # If hop_length_s is 0.02s, then 50 frames = 1s. Let's assume features are processed in 1s chunks for now.

    # --- CNN Encoder (Frontend) Parameters (Inspired by CST-former's Encoder) ---
    'encoder_type': 'conv',  # Type of CNN encoder: 'conv', 'ResNet', 'SENet'
    # 'conv' uses basic ConvBlocks, 'ResNet' uses ResidualBlocks, 'SENet' uses SEBasicBlocks

    'nb_cnn2d_filt': 64,  # Number of filters in the CNN layers (used by CST-former's Encoder)
    # This replaces 'nb_conv_filters' from your original params for consistency with CST structure

    # PATCH: Adjusted t_pool_size to match label sequence length after pooling,
    # assuming input features from DataGenerator are longer (e.g., ~250 frames for 5s audio at 0.02s hop)
    # and label_sequence_length (50) is the target output frames.
    # A total pooling of 5 (e.g., 250_input_frames / 5_pool_factor = 50_output_frames) is needed.
    # Applying it in the first CNN block, similar to original SELDnet.
    't_pool_size': [5, 1, 1],  # Temporal pooling size for each CNN block layer.

    'f_pool_size': [4, 4, 2],  # Frequency pooling size for each CNN block layer (CST-former default: [4,4,2])
    'cnn_dropout_rate': 0.05,  # Dropout rate within CNN blocks (replaces 'dropout' for CNN part)
    't_pooling_loc': 'front',  # Temporal pooling location in CNN: 'front' or 'end'. CST-former often uses 'front'.

    # --- CST/CMT Block Parameters (Sequence Model) ---
    'CMT_block': True,  # True to use CMT_block as the main sequence processor
    'CMT_split': False,
    # False for standard CMT block layer (spectral then temporal within one attention call if FreqAtten=True)
    # True would mean separate Spectral Conformer then Temporal Conformer logic.
    'nb_cmt_layers': 3,  # Number of CMT_Layers within the CMT_block (replaces 'num_conformer_blocks')
    # (CST-former uses 'nb_self_attn_layers' for this, e.g., 2 or 4)
    'nb_heads': 8,  # Number of attention heads in MHSA (replaces 'nb_attn_heads')
    'cmt_dropout_rate': 0.1,
    # Dropout rate for attention and FFNs within CMT block (replaces 'dropout' for conformer part)
    'ffn_ratio_cmt': 4.0,  # Expansion ratio for IRFFN in CMT layers
    'conv_kernel_size_cmt': 31,  # Kernel size for 1D depthwise conv in Conformer-like parts (if we adapt it to CMT)
    # The IRFFN in CST's CMT_Layers uses 2D convs. Standard Conformer uses 1D.
    # The provided CMT_Block.py uses 2D LPU and 2D IRFFN.

    # Attention mechanism flags for CST_attention within CMT_Layers
    'ChAtten_DCA': False,  # Divided Channel Attention (primarily for >2 channel mic arrays)
    'ChAtten_ULE': False,  # Unfolded Local Embedding for channel attention (primarily for >2 channel mic arrays)
    'FreqAtten': True,  # Enable spectral attention processing
    'LinearLayer': True,  # Whether to use an extra linear layer after MHSA in CST attention parts

    # --- Fully Connected Network (FNN) Parameters (Backend) ---
    'nb_fnn_layers': 1,
    'fnn_size': 128,
    'fnn_dropout_rate': 0.05,  # Dropout for the final FNN layers

    # Loss function parameters
    'multiACCDOA': True,  # True for Multi-ACCDOA output format, False for single ACCDOA per class
    'thresh_unify': 15,  # Threshold for unifying predicted DOAs in ADPIT loss (degrees) - specific to some loss impl.

    # Training parameters
    'nb_epochs': 2,
    'batch_size': 256,  # Adjusted from 256 due to potentially larger model
    'nb_workers': 4,  # Number of workers for DataLoader
    'shuffle': True,  # Shuffle training data

    # Optimizer parameters
    'learning_rate': 3e-4,  # Might need adjustment for the new architecture
    'weight_decay': 1e-5,  # Weight decay for optimizer

    # Data folds (adjust based on DCASE2025 dataset structure)
    # These are placeholders, update them according to the actual DCASE2025 dev set splits
    'dev_train_folds': ['fold1', 'fold3'],  # Example
    'dev_test_folds': ['fold4'],  # Example

    # Metric parameters (from your original params, seem standard for SELD)
    'average': 'macro',
    'lad_doa_thresh': 20,
    'lad_dist_thresh': float('inf'),  # Or a specific value if distance accuracy is critical
    'lad_reldist_thresh': 1.0,
    'lad_req_onscreen': False,  # Not relevant for audio-only
    'use_jackknife': False,  # Set to True for final robust evaluation if needed (slow)
    'segment_based_metrics': False,  # Event-based is more common now

    'loss_weights': {'doa': 1.0, 'dist': 10.0},

    # Parameters from CST-former that might be useful or need mapping:
    # 'baseline': False, # Not using GRU baseline from CST-former
    # 'nb_mel_bins': 64, # Already have 'nb_mels'
    # 'unique_classes': 13, # Already have 'nb_classes'
    # 'input_nb_ch': 2, # Defined as 'audio_input_channels'
}


