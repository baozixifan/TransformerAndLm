# File   : conformer.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from otrans.module import *
from otrans.utils import get_enc_padding_mask
from otrans.attention import MultiHeadedAttention, MultiHeadedAttentionPdrophead
from otrans.module import LayerNorm, PositionwiseFeedForward
from otrans.ConformerConvolution import ConformerConvolutionModule
from otrans.module import PositionalEncoding, LDPEPositionalEncoding, LRPEPositionalEncoding
# from otrans.module.pos import MixedPositionalEncoding, RelPositionalEncoding


logger = logging.getLogger(__name__)


class ConformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout=0.0, ffn_dropout=0.0,
                 residual_dropout=0.1, conv_dropout=0.0, macaron_style=True, ffn_scale=0.5, conv_bias=True,
                 relative_positional=True, activation='glu', drop_head_rate=0):
        super(ConformerEncoderBlock, self).__init__()

        self.macaron_style = macaron_style
        self.ffn_scale = ffn_scale
        self.relative_positional = relative_positional
        self.residual_dropout = residual_dropout

        if self.macaron_style:
            self.pre_ffn = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation=activation)
            self.macaron_ffn_norm = nn.LayerNorm(d_model)

        # if self.relative_positional:
        #     self.mha = MultiHeadedSelfAttentionWithRelPos(n_heads, d_model, slf_attn_dropout)
        # else:
        self.mha = MultiHeadedAttentionPdrophead(n_heads, d_model, slf_attn_dropout, p=drop_head_rate)

        self.mha_norm = nn.LayerNorm(d_model)

        self.conv = ConformerConvolutionModule(d_model, cov_kernel_size, conv_bias, conv_dropout)
        self.conv_norm = nn.LayerNorm(d_model)

        self.post_ffn = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation=activation)
        self.post_ffn_norm = nn.LayerNorm(d_model)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, pos=None):

        if self.macaron_style:
            residual = x
            x = self.macaron_ffn_norm(x)
            x = residual + self.ffn_scale * F.dropout(self.pre_ffn(x), p=self.residual_dropout)

        residual = x
        x = self.mha_norm(x)

        # if self.relative_positional:
        #     slf_attn_out = self.mha(x, mask.unsqueeze(1), pos)
        # else:
        #     slf_attn_out = self.mha(x, mask.unsqueeze(1))

        slf_attn_out = self.mha(x, x, x, mask)

        x = residual + F.dropout(slf_attn_out, p=self.residual_dropout)

        residual = x
        x = self.conv_norm(x)
        x = residual + F.dropout(self.conv(x, mask), p=self.residual_dropout)

        residual = x
        x = self.post_ffn_norm(x)
        x = residual + self.ffn_scale * F.dropout(self.post_ffn(x), p=self.residual_dropout)

        x = self.final_norm(x)

        return x, mask


class ConformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, d_ff, cov_kernel_size, n_heads, nblocks=12, pos_dropout_rate=0.0,
                 slf_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1, conv_dropout=0.0, macaron_style=True,
                 ffn_scale=0.5, conv_bias=True, relative_positional=True, activation='glu', drop_head_rate=0):
        super(ConformerEncoder, self).__init__()

        # self.relative_positional = relative_positional

        # if self.relative_positional:
        #     self.posemb = RelPositionalEncoding(d_model, pos_dropout)
        # else:
        #     self.posemb = MixedPositionalEncoding(d_model, pos_dropout)

        self.embed = Conv2dSubsampling(input_size, d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList(
            [
                ConformerEncoderBlock(
                    d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout, ffn_dropout, residual_dropout,
                    conv_dropout, macaron_style, ffn_scale, conv_bias, relative_positional, activation, drop_head_rate
                ) for _ in range(nblocks)
            ]
        )

        self.output_size = d_model

    def forward(self, inputs, input_length):

        enc_mask = get_enc_padding_mask(inputs, input_length)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)
            enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        return enc_output, enc_mask

