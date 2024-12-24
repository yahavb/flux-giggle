import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Union, Tuple

DTYPE=torch.bfloat16

class TracingTransformerEmbedderWrapper(nn.Module):
    def __init__(
            self,
            x_embedder,
            context_embedder,
            time_text_embed,
            pos_embed):
        super().__init__()
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.time_text_embed = time_text_embed
        self.pos_embed = pos_embed

    def forward(
            self,
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            txt_ids,
            img_ids):

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)
        return hidden_states, temb, image_rotary_emb

class TracingTransformerBlockChunk(nn.Module):
    def __init__(self, blocks_subset):
        super().__init__()
        self.blocks_subset = nn.ModuleList(blocks_subset)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        for block in self.blocks_subset:
            encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
            )
        print("[DEBUG] TracingTransformerBlockChunk, encoder_hidden_states.shape =", encoder_hidden_states.shape)
        print("[DEBUG] TracingTransformerBlockChunk, hidden_states.shape =", hidden_states.shape)
        return encoder_hidden_states, hidden_states

class TracingTransformerBlockWrapper(nn.Module):
    def __init__(self, transformer, transformerblock):
        super().__init__()
        self.transformerblock = transformerblock
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb):
        for block in self.transformerblock:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )
        return encoder_hidden_states, hidden_states

class TracingSingleTransformerBlockChunk(nn.Module):
    def __init__(self, single_blocks_subset):
        super().__init__()
        self.blocks_subset = nn.ModuleList(single_blocks_subset)

    def forward(self, hidden_states, temb, image_rotary_emb):
        for block in self.blocks_subset:
            hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb
            )
        print("[DEBUG] TracingSingleTransformerBlockChunk, hidden_states.shape =", hidden_states.shape)
        return hidden_states

class TracingSingleTransformerBlockWrapper(nn.Module):
    def __init__(self, transformer, transformerblock):
        super().__init__()
        self.transformerblock = transformerblock
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(self, hidden_states, temb, image_rotary_emb):
        for block in self.transformerblock:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )
        return hidden_states


class TracingTransformerOutLayerWrapper(nn.Module):
    def __init__(self, norm_out, proj_out):
        super().__init__()
        self.norm_out = norm_out
        self.proj_out = proj_out

    def forward(self, hidden_states, encoder_hidden_states, temb):
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        return (self.proj_out(hidden_states),)

class TracingSingleTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_hidden_dim = None

        self.norm = None
        self.proj_mlp = None
        self.act_mlp = None
        self.proj_out = None
        self.proj_out_2 = None

        self.attn = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        gate = gate.unsqueeze(1)
        hidden_states = gate * (self.proj_out(attn_output)
                                + self.proj_out_2(mlp_hidden_states))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == DTYPE:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states
