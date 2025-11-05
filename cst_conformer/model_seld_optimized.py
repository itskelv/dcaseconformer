"""
model_seld_optimized.py

Optimized SELD model for DCASE2025 stereo input, incorporating
CST-former inspired components: a CNN frontend and a CMT_Block for sequence modeling.
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Assuming these custom modules are in the same directory or accessible via Python path
from cst_conformer.layers_cst import FCLayerCST
from cst_conformer.layers_cst import EncoderCNN
from cst_conformer.layers_cst import CMTBlockCST


class SeldOptimizedConformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.audio_input_channels = params['audio_input_channels']  # Should be 2 for stereo

        # 1. CNN Frontend (EncoderCNN from encoder_cnn_cst.py)
        # This will take raw feature input [B, audio_input_channels, T_raw_feat, F_mels]
        # and output [B, nb_cnn2d_filt, T_pooled, F_pooled]
        self.cnn_encoder = EncoderCNN(
            in_channels=self.audio_input_channels,
            params=params
        )

        # Calculate dimensions after CNN encoder
        # T_raw_feat and F_mels depend on feature extraction (e.g., from a 1s audio chunk)
        # For F_pooled:
        self.pooled_freq_dim = params['nb_mels']
        for pool_f in params['f_pool_size']:
            self.pooled_freq_dim = np.floor(self.pooled_freq_dim / pool_f)
        self.pooled_freq_dim = int(self.pooled_freq_dim)

        # For T_pooled: This depends on the input sequence length to the CNN
        # and params['t_pool_size']. If 'label_sequence_length' (e.g., 50 frames)
        # is the number of frames *after* all processing, we need to be careful.
        # Let's assume the input T to CNN corresponds to params['label_sequence_length'] *before* cnn t_pooling
        # This is a bit circular. The DataGenerator should provide fixed size T, F to the model.
        # Let's assume T_feat_in_cnn and F_feat_in_cnn are known/fixed by data pipeline.
        # The EncoderCNN itself handles the pooling.

        # 2. CMT Block (Sequence Modeling from cmt_block_cst.py)
        # Takes output of cnn_encoder: [B, nb_cnn2d_filt, T_pooled_cnn, F_pooled_cnn]
        # Outputs: [B, T_pooled_cnn, D_model_cmt]
        # where D_model_cmt = nb_cnn2d_filt * F_pooled_cnn
        self.cmt_sequence_model = CMTBlockCST(
            cnn_output_channels=params['nb_cnn2d_filt'],
            pooled_freq_dim=self.pooled_freq_dim,  # F_pooled_cnn
            params=params
        )

        # Dimension of the output from CMTBlockCST
        self.d_model_cmt = params['nb_cnn2d_filt'] * self.pooled_freq_dim

        # 3. Fully Connected Network (Backend from layers_cst.py)
        self.output_dim = self._compute_output_dim(params)
        self.fnn_output_stage = FCLayerCST(
            in_dim=self.d_model_cmt,
            out_dim=self.output_dim,
            params=params
        )

        # Output Activations (as in original SELDConformerModel)
        self.doa_act = nn.Tanh()  # For cartesian x, y, (z)
        self.dist_act = nn.ReLU()  # For distance (must be non-negative)
        # self.onscreen_act = nn.Sigmoid() # Not used for audio-only

        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        # Most submodules (CNN, CMT, FC) handle their own init.
        # Can add specific initializations here if needed for layers directly in this class.
        pass

    def _compute_output_dim(self, params):
        # Based on Multi-ACCDOA format
        # Each of max_polyphony tracks has (x, y, distance) for audio-only
        # Number of elements per track for audio-only: 3 (x, y, dist)
        # Total elements = max_polyphony * num_elements_per_track * nb_classes
        if params['multiACCDOA']:
            # For audio-only: 3 elements (x, y, dist) per polyphony instance
            return params['max_polyphony'] * 3 * params['nb_classes']
        else:
            # For single ACCDOA (1 instance per class): 3 elements * nb_classes
            return 3 * params['nb_classes']

    def forward(self, audio_feat):
        # Input audio_feat: [B, audio_input_channels, T_feat_raw, F_mels]
        # (e.g., from DataGenerator: B, 2, 50, 64 if T_feat_raw is 50 for 1s with 0.02s hop before CNN pooling)

        # 1. Pass through CNN Encoder
        # Output: [B, nb_cnn2d_filt, T_pooled_cnn, F_pooled_cnn]
        cnn_out = self.cnn_encoder(audio_feat)

        # 2. Pass through CMT Sequence Model
        # Input: [B, nb_cnn2d_filt, T_pooled_cnn, F_pooled_cnn]
        # Output: [B, T_pooled_cnn, D_model_cmt] where D_model_cmt = nb_cnn2d_filt * F_pooled_cnn
        sequence_out = self.cmt_sequence_model(cnn_out)

        # 3. Pass through FNN Output Stage
        # Input: [B, T_pooled_cnn, D_model_cmt]
        # Output: [B, T_pooled_cnn, self.output_dim]
        predictions = self.fnn_output_stage(sequence_out)

        # 4. Process output (apply activations and reshape if necessary)
        processed_predictions = self._process_output_activations(predictions)

        return processed_predictions

    def _process_output_activations(self, pred):
        # pred shape: [B, T, total_output_dim]
        # total_output_dim depends on multiACCDOA, nb_classes
        # For audio-only, multiACCDOA:
        # Reshape to [B, T, max_polyphony, nb_classes, 3_elements] for applying activations easily

        B, T, _ = pred.shape
        nb_classes = self.params['nb_classes']

        if self.params['multiACCDOA']:
            max_polyphony = self.params['max_polyphony']
            # Expected pred: [B, T, max_polyphony * nb_classes * 3]
            pred_reshaped = pred.reshape(B, T, max_polyphony, nb_classes, 3)  # x, y, dist

            doa_pred = self.doa_act(pred_reshaped[..., 0:2])  # X, Y coordinates
            dist_pred = self.dist_act(pred_reshaped[..., 2:3])  # Distance

            # Concatenate back and reshape to original pred output format [B, T, output_dim]
            final_pred = torch.cat((doa_pred, dist_pred), dim=-1)
            final_pred = final_pred.reshape(B, T, -1)  # Flatten last three dims
        else:  # Single ACCDOA per class
            # Expected pred: [B, T, nb_classes * 3]
            pred_reshaped = pred.reshape(B, T, nb_classes, 3)  # x, y, dist

            doa_pred = self.doa_act(pred_reshaped[..., 0:2])
            dist_pred = self.dist_act(pred_reshaped[..., 2:3])

            final_pred = torch.cat((doa_pred, dist_pred), dim=-1)
            final_pred = final_pred.reshape(B, T, -1)

        return final_pred

