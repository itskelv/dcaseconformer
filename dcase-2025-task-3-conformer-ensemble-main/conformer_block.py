# conformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, conv_kernel_size=31):
        super(ConformerBlock, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x is of shape (batch, time_steps, d_model)

        residual = x
        x, _ = self.mhsa(x, x, x)  # self-attention
        x = self.layer_norm1(x + residual)

        residual = x
        x_conv = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm2(x_conv + residual)

        residual = x
        x_ff = self.feed_forward(x)
        x = self.layer_norm3(x_ff + residual)
        return x
