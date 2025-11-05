import torch
import torch.nn as nn

from model import ConvBlock
import numpy as np

from conformer_block import ConformerBlock


class SELDConformerModel(nn.Module):
    def __init__(self, params):
        super(SELDConformerModel, self).__init__()
        self.params = params
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            in_channels = params['nb_conv_filters'] if conv_cnt else 2  # stereo input
            self.conv_blocks.append(ConvBlock(
                in_channels=in_channels,
                out_channels=params['nb_conv_filters'],
                pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                dropout=params['dropout']
            ))

        d_model = params['nb_conv_filters'] * int(np.floor(params['nb_mels'] / np.prod(params['f_pool_size'])))

        self.num_conformer_blocks = 3
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, nhead=params['nb_attn_heads'], dropout=params['dropout'])
            for _ in range(self.num_conformer_blocks)
        ])

        self.fnn_list = nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(nn.Linear(d_model if fc_cnt == 0 else params['fnn_size'], params['fnn_size']))
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else d_model, self._compute_output_dim(params)))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        if params['modality'] == 'audio_visual':
            self.onscreen_act = nn.Sigmoid()

    def _compute_output_dim(self, params):
        if params['multiACCDOA']:
            return params['max_polyphony'] * (3 if params['modality'] == 'audio' else 4) * params['nb_classes']
        else:
            return (3 if params['modality'] == 'audio' else 4) * params['nb_classes']

    def forward(self, audio_feat, vid_feat=None):
        for conv_block in self.conv_blocks:
            audio_feat = conv_block(audio_feat)
        audio_feat = audio_feat.transpose(1, 2).contiguous()  # shape: [B, T, C, F]
        audio_feat = audio_feat.view(audio_feat.size(0), audio_feat.size(1), -1)  # [B, T, D]

        for conformer in self.conformer_blocks:
            audio_feat = conformer(audio_feat)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)
            fused_feat = self.fuse_audio_video(audio_feat, vid_feat)
        else:
            fused_feat = audio_feat

        for fnn in self.fnn_list[:-1]:
            fused_feat = fnn(fused_feat)
        pred = self.fnn_list[-1](fused_feat)

        pred = self._process_output(pred)
        return pred

    def fuse_audio_video(self, audio_feat, vid_feat):
        vid_proj = nn.Linear(vid_feat.size(-1), audio_feat.size(-1)).to(audio_feat.device)
        vid_feat = vid_proj(vid_feat)
        decoder_layer = nn.TransformerDecoderLayer(d_model=audio_feat.size(-1),
                                                   nhead=self.params['nb_attn_heads'],
                                                   batch_first=True).to(audio_feat.device)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.params['nb_transformer_layers']).to(
            audio_feat.device)
        fused_feat = transformer_decoder(audio_feat, vid_feat)
        return fused_feat

    def _process_output(self, pred):

        if self.params['modality'] == 'audio':
            if self.params['multiACCDOA']:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, self.params['nb_classes'])
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=3).reshape(pred.size(0), pred.size(1), -1)
            else:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, self.params['nb_classes'])
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=2).reshape(pred.size(0), pred.size(1), -1)
        else:
            if self.params['multiACCDOA']:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 4, self.params['nb_classes'])
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=3).reshape(pred.size(0), pred.size(1), -1)
            else:
                pred = pred.reshape(pred.size(0), pred.size(1), 4, self.params['nb_classes'])
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=2).reshape(pred.size(0), pred.size(1), -1)
        return pred

