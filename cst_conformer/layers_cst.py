# layers_cst.py
"""
Contains fundamental layer building blocks, adapted from the CST-former
repository files provided by the user (layers.py).
Modifications for clarity, stereo input context, and integration.
CORRECTED: CMTLayer and CMTBlockCST interaction for shape consistency.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Helper functions ---
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def make_pairs(x):
    """Converts an int to a tuple (x, x) if it's not already a tuple."""
    return x if isinstance(x, tuple) else (x, x)


# --- Basic Convolutional Blocks (Used by Encoder_CNN_CST) ---
class ConvBlockCST(nn.Module):
    """
    A basic convolutional block: Conv2D -> BatchNorm -> ReLU.
    From user-provided layers.py, renamed for clarity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)  # Bias False if BN follows
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self, m=None):  # Allow calling directly
        if m is None:
            self.apply(self._init_weights)  # Apply to all submodules if m is None
            return

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBlockTwoCST(nn.Module):
    """
    Two sequential ConvBlockCSTs.
    From user-provided layers.py.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        # First ConvBlock: in_channels -> out_channels
        self.conv_block1 = ConvBlockCST(in_channels, out_channels, kernel_size, stride, padding)
        # Second ConvBlock: out_channels -> out_channels
        self.conv_block2 = ConvBlockCST(out_channels, out_channels, kernel_size, stride, padding)
        # No separate init needed as ConvBlockCST handles its own initialization

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class ResidualBlockCST(nn.Module):
    """
    A basic residual block. Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU.
    From user-provided layers.py.
    Assumes in_channels == out_channels for the identity mapping.
    If stride > 1 or in_channels != out_channels, a projection shortcut is needed (not implemented here).
    """

    def __init__(self, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        # Ensuring out_channels of conv1 is 'channels' for residual connection
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               bias=False)  # Stride should typically be 1 for conv2 in basic block
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self, m=None):  # Allow calling directly
        if m is None:
            self.apply(self._init_weights)  # Apply to all submodules if m is None
            return

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_init):
        identity = x_init

        out = self.conv1(x_init)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Add residual
        out = self.relu2(out)
        return out


class SEBasicBlockCST(nn.Module):
    """
    Basic ResNet block with Squeeze-and-Excitation.
    From user-provided layers.py (SEBasicBlock).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)  # stride is 1 for the second conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayerCST(out_channels, reduction)
        self.downsample = downsample
        # self.stride = stride # Not used directly

        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class SELayerCST(nn.Module):
    """
    Squeeze-and-Excitation Layer.
    From user-provided layers.py.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()  # Initialize weights of Linear layers

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:  # Check if bias exists
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- CMT Components (from user-provided layers.py, for CMT_Block_CST) ---
class LocalPerceptionUnitCST(nn.Module):
    """
    Local Perception Unit (LPU) using a 3x3 depthwise convolution.
    From user-provided layers.py (LocalPerceptionUint).
    """

    def __init__(self, dim, act=False):  # 'dim' is num_channels
        super().__init__()
        self.act_after_conv = act  # Whether to apply GELU + BN after conv
        # Depthwise convolution
        self.conv_3x3_dw = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3,
            padding=1, groups=dim, bias=False  # Bias False for DW conv if BN follows
        )
        if self.act_after_conv:
            # CST-former applies GELU then BN. Standard is BN then GELU.
            # Let's follow CST-former for now if act=True
            self.activation = nn.GELU()
            self.bn = nn.BatchNorm2d(dim)

        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # No bias to init for conv_3x3_dw as bias=False
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_3x3_dw(x)
        if self.act_after_conv:
            out = self.activation(out)  # GELU first
            out = self.bn(out)  # Then BatchNorm
        return out


class InvertedResidualFeedForwardCST(nn.Module):
    """
    Inverted Residual Feed-Forward Network (IRFFN).
    From user-provided layers.py. Uses 2D Convolutions.
    Structure (as per user's original layers.py):
    1. x_processed = BN(GELU(Conv1x1_expand(x)))
    2. dw_features = Conv3x3_dw(x_processed)
    3. activated_dw_features = BN(GELU(dw_features))
    4. res_added = x_processed + activated_dw_features
    5. projected = BN(Conv1x1_project(res_added))
    """

    def __init__(self, dim, dim_ratio=4.0):  # 'dim' is num_channels
        super().__init__()
        expanded_dim = int(dim_ratio * dim)

        # Expansion: 1x1 Conv, GELU, BN
        self.conv1x1_expand = nn.Conv2d(dim, expanded_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(expanded_dim)

        # Depthwise Convolution: 3x3 DWConv
        self.conv3x3_dw = nn.Conv2d(expanded_dim, expanded_dim, kernel_size=3, stride=1, padding=1, groups=expanded_dim,
                                    bias=False)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(expanded_dim)

        # Projection: 1x1 Conv, BN
        self.conv1x1_project = nn.Conv2d(expanded_dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Pointwise Conv -> GELU -> BN
        x_processed = self.bn1(self.act1(self.conv1x1_expand(x)))

        # 2. Depthwise Conv on x_processed
        dw_features = self.conv3x3_dw(x_processed)
        # 3. Activation (GELU, BN) on dw_features
        activated_dw_features = self.bn2(self.act2(dw_features))

        # 4. Residual connection (as per user's original layers.py for IRFFN)
        res_added = x_processed + activated_dw_features

        # 5. Pointwise Conv (Projection) -> BN
        projected = self.bn3(self.conv1x1_project(res_added))
        return projected


# --- Attention Components (for CST_Attention_CST, from user-provided layers.py) ---
class SpectralAttentionCST(nn.Module):
    """
    Spectral Attention module.
    From user-provided layers.py (Spec_attention).
    Input x: [B, T, F*C]
    Output: [B, T, F*C] (after MHSA, optional linear, residual, LayerNorm)
    """

    def __init__(self, embed_dim_channel, num_heads, dropout_rate, use_linear_layer, temp_embed_dim_overall):
        super().__init__()
        self.use_linear_layer = use_linear_layer
        self.sp_mhsa = nn.MultiheadAttention(embed_dim=embed_dim_channel, num_heads=num_heads,
                                             dropout=dropout_rate, batch_first=True)
        self.sp_layer_norm = nn.LayerNorm(temp_embed_dim_overall)
        if self.use_linear_layer:
            self.sp_linear = nn.Linear(temp_embed_dim_overall, temp_embed_dim_overall)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, C_dim, T_dim, F_dim):  # x is [B, T, F*C]
        x_init = x
        x_attn_in = rearrange(x_init, 'b t (f c) -> (b t) f c', c=C_dim, f=F_dim).contiguous()
        xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
        xs = rearrange(xs, '(b t) f c -> b t (f c)', t=T_dim).contiguous()

        if self.use_linear_layer:
            xs = self.activation(self.sp_linear(xs))

        xs = self.dropout(xs)
        xs = xs + x_init
        x_out = self.sp_layer_norm(xs)
        return x_out


class TemporalAttentionCST(nn.Module):
    """
    Temporal Attention module.
    From user-provided layers.py (Temp_attention).
    Input x: [B, T, F*C]
    Output: [B, T, F*C] (after MHSA, optional linear, residual, LayerNorm)
    """

    def __init__(self, embed_dim_freq_slot, num_heads, dropout_rate, use_linear_layer, temp_embed_dim_overall):
        super().__init__()
        self.use_linear_layer = use_linear_layer
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=embed_dim_freq_slot, num_heads=num_heads,
                                               dropout=dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(temp_embed_dim_overall)
        if self.use_linear_layer:
            self.temp_linear = nn.Linear(temp_embed_dim_overall, temp_embed_dim_overall)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, C_dim, T_dim, F_dim):  # x is [B, T, F*C]
        x_init = x
        x_attn_in = rearrange(x_init, 'b t (f c) -> (b f) t c', c=C_dim, f=F_dim).contiguous()
        xt, _ = self.temp_mhsa(x_attn_in, x_attn_in, x_attn_in)
        xt = rearrange(xt, '(b f) t c -> b t (f c)', f=F_dim).contiguous()

        if self.use_linear_layer:
            xt = self.activation(self.temp_linear(xt))

        xt = self.dropout(xt)
        xt = xt + x_init
        x_out = self.temp_layer_norm(xt)
        return x_out


# --- Final FC Layer (from user-provided layers.py, for final output) ---
class FCLayerCST(nn.Module):
    """
    Fully Connected output layer.
    From user-provided layers.py (FC_layer).
    """

    def __init__(self, in_dim, out_dim, params):  # out_dim is total output dimension for SELD
        super().__init__()
        self.fnn_list = nn.ModuleList()

        current_dim = in_dim
        if params['nb_fnn_layers'] > 0:
            for _ in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(current_dim, params['fnn_size'], bias=True))
                self.fnn_list.append(nn.ReLU(inplace=True))  # Added ReLU as typical for hidden FCs
                if params['fnn_dropout_rate'] > 0:
                    self.fnn_list.append(nn.Dropout(params['fnn_dropout_rate']))
                current_dim = params['fnn_size']

        self.fnn_list.append(nn.Linear(current_dim, out_dim, bias=True))
        self._init_weights()

    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        for layer in self.fnn_list:
            x = layer(x)
        return x


# --- CNN Encoder (Moved from encoder_cnn_cst.py to be self-contained with its dependencies) ---
class ConvEncoderCST(nn.Module):
    """ Basic Convolutional Encoder using ConvBlockCST. """

    def __init__(self, in_channels, params):
        super().__init__()
        self.t_pooling_loc = params["t_pooling_loc"]
        self.conv_block_list = nn.ModuleList()
        current_channels = in_channels
        for i in range(len(params['f_pool_size'])):
            out_channels_current_block = params['nb_cnn2d_filt']
            self.conv_block_list.append(
                ConvBlockCST(in_channels=current_channels, out_channels=out_channels_current_block)
            )
            temporal_pool = params['t_pool_size'][i] if self.t_pooling_loc == 'front' else 1
            freq_pool = params['f_pool_size'][i]
            self.conv_block_list.append(nn.MaxPool2d((temporal_pool, freq_pool)))
            if params['cnn_dropout_rate'] > 0:
                self.conv_block_list.append(nn.Dropout2d(p=params['cnn_dropout_rate']))
            current_channels = out_channels_current_block

    def forward(self, x):
        for layer in self.conv_block_list:
            x = layer(x)
        return x


class ResNetEncoderCST(nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()
        self.t_pooling_loc = params["t_pooling_loc"]
        self.first_conv = ConvBlockCST(in_channels=in_channels, out_channels=params['nb_cnn2d_filt'])
        self.res_block_stages = nn.ModuleList()
        current_channels = params['nb_cnn2d_filt']
        for i in range(len(params['f_pool_size'])):
            stage = nn.Sequential(
                ResidualBlockCST(channels=current_channels),
                ResidualBlockCST(channels=current_channels),
                nn.MaxPool2d(
                    (params['t_pool_size'][i] if self.t_pooling_loc == 'front' else 1, params['f_pool_size'][i])),
                nn.Dropout2d(p=params['cnn_dropout_rate']) if params['cnn_dropout_rate'] > 0 else nn.Identity()
            )
            self.res_block_stages.append(stage)

    def forward(self, x):
        x = self.first_conv(x)
        for stage in self.res_block_stages:
            x = stage(x)
        return x


class SENetEncoderCST(nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()
        self.t_pooling_loc = params["t_pooling_loc"]
        self.first_conv = ConvBlockCST(in_channels=in_channels, out_channels=params['nb_cnn2d_filt'])
        self.se_block_stages = nn.ModuleList()
        current_channels = params['nb_cnn2d_filt']
        for i in range(len(params['f_pool_size'])):
            downsample_layer = None
            stage_blocks = []
            stage_blocks.append(
                SEBasicBlockCST(current_channels, current_channels, stride=1, downsample=downsample_layer))
            stage_blocks.append(SEBasicBlockCST(current_channels, current_channels, stride=1, downsample=None))
            temporal_pool = params['t_pool_size'][i] if self.t_pooling_loc == 'front' else 1
            freq_pool = params['f_pool_size'][i]
            stage_blocks.append(nn.MaxPool2d((temporal_pool, freq_pool)))
            if params['cnn_dropout_rate'] > 0:
                stage_blocks.append(nn.Dropout2d(p=params['cnn_dropout_rate']))
            self.se_block_stages.append(nn.Sequential(*stage_blocks))

    def forward(self, x):
        x = self.first_conv(x)
        for stage in self.se_block_stages:
            x = stage(x)
        return x


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()
        self.encoder_type = params['encoder_type']
        if self.encoder_type == 'ResNet':
            self.encoder = ResNetEncoderCST(in_channels, params)
        elif self.encoder_type == 'conv':
            self.encoder = ConvEncoderCST(in_channels, params)
        elif self.encoder_type == 'SENet':
            self.encoder = SENetEncoderCST(in_channels, params)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def forward(self, x):
        return self.encoder(x)


# --- CST Attention Layer (Moved from cst_attention_cst.py) ---
class CSTAttentionLayer(nn.Module):
    def __init__(self, temp_embed_dim, cnn_output_channels, params):
        super().__init__()
        self.params = params
        self.cnn_output_channels = cnn_output_channels
        self.temp_embed_dim = temp_embed_dim
        self.use_ch_atten_dca = params.get('ChAtten_DCA', False)
        self.use_ch_atten_ule = params.get('ChAtten_ULE', False)
        self.use_freq_atten = params.get('FreqAtten', True)
        self.use_linear_layer_after_mhsa = params.get('LinearLayer', True)
        self.dropout_rate = params.get('cmt_dropout_rate', 0.1)
        self.num_heads = params.get('nb_heads', 8)

        if self.use_ch_atten_dca or self.use_ch_atten_ule:
            print(
                "Warning: ChAtten_DCA or ChAtten_ULE are enabled but not fully adapted for stereo in this CSTAttentionLayer.")

        if self.use_freq_atten:
            self.spectral_attention = SpectralAttentionCST(
                embed_dim_channel=self.cnn_output_channels,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                use_linear_layer=self.use_linear_layer_after_mhsa,
                temp_embed_dim_overall=self.temp_embed_dim
            )
        self.temporal_attention = TemporalAttentionCST(
            embed_dim_freq_slot=self.cnn_output_channels,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_linear_layer=self.use_linear_layer_after_mhsa,
            temp_embed_dim_overall=self.temp_embed_dim
        )
        self._init_weights()

    def _init_weights(self, m=None):
        if m is None: self.apply(self._init_weights); return
        if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight);
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x, C_dim, T_dim, F_dim):  # x is [B, T_dim, F_dim * C_dim]
        current_x = x
        if self.use_freq_atten:
            current_x = self.spectral_attention(current_x, C_dim, T_dim, F_dim)
        current_x = self.temporal_attention(current_x, C_dim, T_dim, F_dim)
        return current_x


# --- CMT Block (Moved from cmt_block_cst.py and CORRECTED) ---
class CMTLayer(nn.Module):
    def __init__(self, cnn_output_channels, pooled_freq_dim, params):
        super().__init__()
        self.params = params
        self.cnn_output_channels = cnn_output_channels
        self.pooled_freq_dim = pooled_freq_dim
        self.temp_embed_dim = self.pooled_freq_dim * self.cnn_output_channels
        self.cmt_split = params.get('CMT_split', False)
        self.ffn_ratio = params.get('ffn_ratio_cmt', 4.0)
        self.dropout_rate = params.get('cmt_dropout_rate', 0.1)

        if not self.cmt_split:
            self.lpu = LocalPerceptionUnitCST(dim=self.cnn_output_channels, act=False)
            self.cst_attention = CSTAttentionLayer(
                temp_embed_dim=self.temp_embed_dim,
                cnn_output_channels=self.cnn_output_channels,
                params=params
            )
            self.norm1_for_irffn_input = nn.LayerNorm(self.cnn_output_channels)
            self.irffn = InvertedResidualFeedForwardCST(dim=self.cnn_output_channels, dim_ratio=self.ffn_ratio)
        else:
            self.lpu_s = LocalPerceptionUnitCST(dim=self.cnn_output_channels, act=False)
            self.spectral_attention = SpectralAttentionCST(
                embed_dim_channel=self.cnn_output_channels, num_heads=params['nb_heads'],
                dropout_rate=self.dropout_rate, use_linear_layer=params['LinearLayer'],
                temp_embed_dim_overall=self.temp_embed_dim
            )
            self.norm_s1 = nn.LayerNorm(self.cnn_output_channels)
            self.irffn_s = InvertedResidualFeedForwardCST(dim=self.cnn_output_channels, dim_ratio=self.ffn_ratio)

            self.lpu_t = LocalPerceptionUnitCST(dim=self.cnn_output_channels, act=False)
            self.temporal_attention = TemporalAttentionCST(
                embed_dim_freq_slot=self.cnn_output_channels, num_heads=params['nb_heads'],
                dropout_rate=self.dropout_rate, use_linear_layer=params['LinearLayer'],
                temp_embed_dim_overall=self.temp_embed_dim
            )
            self.norm_t1 = nn.LayerNorm(self.cnn_output_channels)
            self.irffn_t = InvertedResidualFeedForwardCST(dim=self.cnn_output_channels, dim_ratio=self.ffn_ratio)

        self.drop_path = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

    def forward(self, x_cnn_output_4d):  # Expects [B, C_cnn, T_p, F_p]
        B, C_cnn, T_p, F_p = x_cnn_output_4d.shape

        if not self.cmt_split:
            x_lpu_processed = self.lpu(x_cnn_output_4d)
            x_after_lpu = x_cnn_output_4d + self.drop_path(x_lpu_processed)

            x_att_in_3d = rearrange(x_after_lpu, 'b c t f -> b t (f c)', c=C_cnn, f=F_p).contiguous()
            x_after_attention_3d = self.cst_attention(x_att_in_3d, C_dim=C_cnn, T_dim=T_p, F_dim=F_p)

            x_for_irffn_norm_4d = rearrange(x_after_attention_3d, 'b t (f c) -> b c t f', c=C_cnn, f=F_p).contiguous()

            x_permuted_for_norm = x_for_irffn_norm_4d.permute(0, 2, 3, 1).contiguous()  # B, T, F, C
            normed_for_irffn = self.norm1_for_irffn_input(x_permuted_for_norm)
            input_to_irffn_4d = normed_for_irffn.permute(0, 3, 1, 2).contiguous()  # B, C, T, F

            x_irffn_processed = self.irffn(input_to_irffn_4d)
            x_after_irffn_4d = input_to_irffn_4d + self.drop_path(x_irffn_processed)
            return x_after_irffn_4d  # Return 4D tensor for iteration

        else:  # CMT_split = True
            # Spectral Part
            x_lpu_s = self.lpu_s(x_cnn_output_4d)
            x_after_lpu_s = x_cnn_output_4d + self.drop_path(x_lpu_s)
            x_spec_att_in_3d = rearrange(x_after_lpu_s, 'b c t f -> b t (f c)', c=C_cnn, f=F_p).contiguous()
            x_after_spec_att_3d = self.spectral_attention(x_spec_att_in_3d, C_dim=C_cnn, T_dim=T_p, F_dim=F_p)

            x_for_irffn_s_norm_4d = rearrange(x_after_spec_att_3d, 'b t (f c) -> b c t f', c=C_cnn, f=F_p).contiguous()
            x_permuted_s = x_for_irffn_s_norm_4d.permute(0, 2, 3, 1).contiguous()
            normed_for_irffn_s = self.norm_s1(x_permuted_s)
            input_to_irffn_s_4d = normed_for_irffn_s.permute(0, 3, 1, 2).contiguous()

            x_irffn_s = self.irffn_s(input_to_irffn_s_4d)
            x_spectral_out_4d = input_to_irffn_s_4d + self.drop_path(x_irffn_s)  # This is [B,C,T,F]


            x_lpu_t = self.lpu_t(x_spectral_out_4d)  # Input is 4D
            x_after_lpu_t = x_spectral_out_4d + self.drop_path(x_lpu_t)
            x_temp_att_in_3d = rearrange(x_after_lpu_t, 'b c t f -> b t (f c)', c=C_cnn, f=F_p).contiguous()
            x_after_temp_att_3d = self.temporal_attention(x_temp_att_in_3d, C_dim=C_cnn, T_dim=T_p, F_dim=F_p)

            x_for_irffn_t_norm_4d = rearrange(x_after_temp_att_3d, 'b t (f c) -> b c t f', c=C_cnn, f=F_p).contiguous()
            x_permuted_t = x_for_irffn_t_norm_4d.permute(0, 2, 3, 1).contiguous()
            normed_for_irffn_t = self.norm_t1(x_permuted_t)
            input_to_irffn_t_4d = normed_for_irffn_t.permute(0, 3, 1, 2).contiguous()

            x_irffn_t = self.irffn_t(input_to_irffn_t_4d)
            x_temporal_out_4d = input_to_irffn_t_4d + self.drop_path(x_irffn_t)
            return x_temporal_out_4d


class CMTBlockCST(nn.Module):
    def __init__(self, cnn_output_channels, pooled_freq_dim, params):
        super().__init__()
        self.params = params
        self.cnn_output_channels = cnn_output_channels
        self.pooled_freq_dim_local = pooled_freq_dim  # Store for rearranging
        self.num_cmt_layers = params.get('nb_cmt_layers', 4)

        self.cmt_layers = nn.ModuleList([
            CMTLayer(
                cnn_output_channels=cnn_output_channels,
                pooled_freq_dim=pooled_freq_dim,
                params=params
            ) for _ in range(self.num_cmt_layers)
        ])

        self.d_model_cmt = self.cnn_output_channels * self.pooled_freq_dim_local
        self.final_norm = nn.LayerNorm(self.d_model_cmt)

    def forward(self, x_cnn_output_4d):
        current_features_4d = x_cnn_output_4d
        for cmt_layer in self.cmt_layers:
            current_features_4d = cmt_layer(current_features_4d)

        final_sequence_unnormed_3d = rearrange(current_features_4d, 'b c t f -> b t (f c)',
                                               c=self.cnn_output_channels,
                                               f=self.pooled_freq_dim_local).contiguous()

        final_sequence_normed_3d = self.final_norm(final_sequence_unnormed_3d)
        return final_sequence_normed_3d
