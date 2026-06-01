from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from transformer.model import DecoderOnlyTransformer
from .quantized_modules import (
    QuantizedFeedForward,
    QuantizedLayerNorm,
    QuantizedMultiHeadSelfAttention,
    QuantizedPositionalEmbedding,
    QuantizedTokenEmbedding,
)


def create_fp16_model(model: DecoderOnlyTransformer) -> DecoderOnlyTransformer:
    quantized = deepcopy(model)
    quantized.half()
    quantized.eval()
    return quantized


def create_int8_weight_only_model(model: DecoderOnlyTransformer) -> DecoderOnlyTransformer:
    quantized = deepcopy(model)
    quantized.token_embedding = QuantizedTokenEmbedding(model.token_embedding.weight.data)
    quantized.position_embedding = QuantizedPositionalEmbedding(model.position_embedding.weight.data)
    quantized.final_norm = QuantizedLayerNorm(
        model.final_norm.gamma.data,
        model.final_norm.beta.data,
        model.final_norm.eps,
    )

    for block, source_block in zip(quantized.blocks, model.blocks):
        block.ln1 = QuantizedLayerNorm(
            source_block.ln1.gamma.data,
            source_block.ln1.beta.data,
            source_block.ln1.eps,
        )
        block.attn = QuantizedMultiHeadSelfAttention(source_block.attn)
        block.ln2 = QuantizedLayerNorm(
            source_block.ln2.gamma.data,
            source_block.ln2.beta.data,
            source_block.ln2.eps,
        )
        block.ffn = QuantizedFeedForward(source_block.ffn)

    quantized.eval()
    return quantized


def estimate_model_bytes(model: nn.Module) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total
