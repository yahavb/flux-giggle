import argparse
import torch
import torch.nn as nn
import torch_neuronx
import neuronx_distributed
import os

from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Any, Dict, Optional, Union

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)

TEXT_ENCODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_1/compiled_model/model.pt')
TEXT_ENCODER_2_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_2/compiled_model/model.pt')
VAE_DECODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'decoder/compiled_model/model.pt')
EMBEDDERS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/embedders/model.pt')
OUT_LAYERS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/transformer_out_layers/model.pt')

SINGLE_TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/single_transformer_block')
TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/transformer_blocks')

def load_multiple_models(model_dir):
    models = []
    for filename in sorted(os.listdir(model_dir)):
        if filename.endswith(".pt"):
            models.append(torch.jit.load(os.path.join(model_dir, filename)))
    return models

class NeuronFluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        config,
        x_embedder,
        context_embedder
    ):
        super().__init__()
        self.transformer_blocks_model = load_multiple_models(TRANSFORMER_BLOCKS_DIR)
        self.single_transformer_blocks_model = load_multiple_models(SINGLE_TRANSFORMER_BLOCKS_DIR)
        self.out_layers_model = torch.jit.load(OUT_LAYERS_PATH)
        self.embedders_model = torch.jit.load(EMBEDDERS_PATH)
        self.config = config
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.device = torch.device("cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        print("[DEBUG] forward() called with:")
        print("  hidden_states.shape =", None if hidden_states is None else hidden_states.shape)
        print("  encoder_hidden_states.shape =", None if encoder_hidden_states is None else encoder_hidden_states.shape)

        hidden_states = self.x_embedder(hidden_states)
        print("[DEBUG] After x_embedder:")
        print("  hidden_states.shape =", hidden_states.shape)


        hidden_states, temb, image_rotary_emb = self.embedders_model(
            hidden_states.to(torch.bfloat16),
            timestep.to(torch.bfloat16) if timestep is not None else None,
            guidance.to(torch.bfloat16) if guidance is not None else None,
            pooled_projections.to(torch.bfloat16) if pooled_projections is not None else None,
            txt_ids.to(torch.bfloat16) if txt_ids is not None else None,
            img_ids.to(torch.bfloat16) if img_ids is not None else None
        )
        print("[DEBUG] After embedders_model:")
        print("  hidden_states.shape =", hidden_states.shape)
        print("  temb.shape =", None if temb is None else temb.shape)
        print("  image_rotary_emb.shape =", None if image_rotary_emb is None else image_rotary_emb.shape)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        print("[DEBUG] After context_embedder:")
        print("  encoder_hidden_states.shape =", encoder_hidden_states.shape)

        image_rotary_emb = image_rotary_emb.type(torch.bfloat16)
        print("[DEBUG] Before transformer_blocks_model call:")
        print("  hidden_states.shape =", hidden_states.shape)
        print("  encoder_hidden_states.shape =", encoder_hidden_states.shape)
        
        encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
        for block in self.transformer_blocks_model:
             encoder_hidden_state,hidden_states = block(
                     hidden_states.to(torch.bfloat16),
                     encoder_hidden_states,
                     temb.to(torch.bfloat16),
                     image_rotary_emb.to(torch.bfloat16)
             )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states],dim=1)
        print("[DEBUG] After torch.cat with encoder_hidden_states, hidden_states.shape =", hidden_states.shape)
        print("[DEBUG] Before single_transformer_blocks_model call:")
        for single_block_chunk in self.single_transformer_blocks_model:
             hidden_states = single_block_chunk(
                     hidden_states.to(torch.bfloat16),
                     temb.to(torch.bfloat16),
                     image_rotary_emb.to(torch.bfloat16)
             )

        hidden_states = hidden_states.to(torch.bfloat16)
        print("[DEBUG] After single_transformer_blocks_model call:")
        print("  hidden_states.shape =", hidden_states.shape)

        return self.out_layers_model(
            hidden_states,
            encoder_hidden_states,
            temb
        )


class NeuronFluxCLIPTextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        output = self.encoder(emb)
        output = CLIPEncoderOutput(output)
        return output


class CLIPEncoderOutput():
    def __init__(self, dictionary):
        self.pooler_output = dictionary["pooler_output"]


class NeuronFluxT5TextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        return torch.unsqueeze(self.encoder(emb)["last_hidden_state"], 1)


def run_inference(
        prompt,
        height,
        width,
        max_sequence_length,
        num_inference_steps):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    pipe.text_encoder = NeuronFluxCLIPTextEncoderModel(
        pipe.text_encoder.dtype,
        torch.jit.load(TEXT_ENCODER_PATH))
    pipe.text_encoder_2 = NeuronFluxT5TextEncoderModel(
        pipe.text_encoder_2.dtype,
        torch.jit.load(TEXT_ENCODER_2_PATH))
    pipe.transformer = NeuronFluxTransformer2DModel(
        pipe.transformer.config,
        pipe.transformer.x_embedder,
        pipe.transformer.context_embedder)

    pipe.vae.decoder = torch.jit.load(VAE_DECODER_PATH)

    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length
    ).images[0]
    image.save(os.path.join(COMPILER_WORKDIR_ROOT, "flux-dev.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="prompt for image to be generated; generates cat by default"
    )
    parser.add_argument(
        "-hh",
        "--height",
        type=int,
        default=256,
        help="height of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=256,
        help="width of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=32,
        help="maximum sequence length for the text embeddings"
    )
    parser.add_argument(
        "-n",
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps to run in generating image"
    )
    args = parser.parse_args()
    run_inference(
        args.prompt,
        args.height,
        args.width,
        args.max_sequence_length,
        args.num_inference_steps)
