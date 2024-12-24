import argparse
import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux \
    import FluxTransformer2DModel
from model import (TracingTransformerEmbedderWrapper,
                   TracingSingleTransformerBlockChunk,
                   TracingTransformerBlockWrapper,
                   TracingTransformerBlockChunk,
                   TracingSingleTransformerBlockWrapper,
                   TracingTransformerOutLayerWrapper)

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
DTYPE=torch.bfloat16
PIPELINE="black-forest-labs/FLUX.1-dev"

def trace_flux(height, width, max_sequence_length):
    pipe = FluxPipeline.from_pretrained(PIPELINE,torch_dtype=DTYPE)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe

    mod_pipe_transformer_f = TracingTransformerEmbedderWrapper(
        transformer.x_embedder, transformer.context_embedder,
        transformer.time_text_embed, transformer.pos_embed)

    hidden_states = torch.rand([1, height * width // 256, 3072],dtype=DTYPE)
    timestep = torch.rand([1], dtype=DTYPE)
    guidance = torch.rand([1], dtype=DTYPE)
    pooled_projections = torch.rand([1, 768], dtype=DTYPE)
    txt_ids = torch.rand([1, max_sequence_length, 3], dtype=DTYPE)
    img_ids = torch.rand([1, height * width // 256, 3], dtype=DTYPE)
    sample_inputs = hidden_states, timestep, guidance, pooled_projections,txt_ids, img_ids

    transformer_embedders_neuron=torch_neuronx.trace(
            mod_pipe_transformer_f,
            sample_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,'compiler_workdir'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
            )
    torch_neuronx.async_load(transformer_embedders_neuron)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    embedders_path = os.path.join(compiled_model_path, 'embedders')
    os.makedirs(embedders_path, exist_ok=True)
    model_filename = os.path.join(embedders_path,'model.pt')
    torch.jit.save(transformer_embedders_neuron, model_filename)

    del transformer_embedders_neuron
    del transformer
    hidden_states = torch.rand([1, height * width // 256, 3072],dtype=DTYPE)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],dtype=DTYPE)
    temb = torch.rand([1, 3072], dtype=DTYPE)
    image_rotary_emb = torch.rand([1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],dtype=DTYPE)
    sample_inputs = hidden_states, encoder_hidden_states,temb, image_rotary_emb
    pipe = FluxPipeline.from_pretrained(PIPELINE,torch_dtype=DTYPE)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe
    all_blocks = list(transformer.transformer_blocks)
    num_blocks = len(all_blocks)
    chunk_size = 4

    for chunk_start in range(0, num_blocks, chunk_size):
        chunk_end = chunk_start + chunk_size
        blocks_subset = all_blocks[chunk_start:chunk_end]
        mod_pipe_transformer_f = TracingTransformerBlockChunk(blocks_subset)

        transformer_blocks_neuron=torch_neuronx.trace(
            mod_pipe_transformer_f,
            sample_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,'compiler_workdir'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )
        torch_neuronx.async_load(transformer_blocks_neuron)
        model_chunk_dir = os.path.join(compiled_model_path, f"transformer_blocks")
        os.makedirs(model_chunk_dir, exist_ok=True)
        model_filename = os.path.join(model_chunk_dir,f'chunk_{chunk_start}.pt')
        torch.jit.save(transformer_blocks_neuron, model_filename)

        del transformer_blocks_neuron
    del transformer

    hidden_states = torch.rand([1, height * width // 256 + max_sequence_length, 3072],dtype=DTYPE)
    temb = torch.rand([1, 3072], dtype=DTYPE)
    image_rotary_emb = torch.rand([1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],dtype=DTYPE)
    sample_inputs = hidden_states, temb, image_rotary_emb

    pipe = FluxPipeline.from_pretrained(PIPELINE,torch_dtype=DTYPE)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe

    all_blocks = list(transformer.single_transformer_blocks)
    num_blocks = len(all_blocks)
    chunk_size = 4
    for chunk_start in range(0, num_blocks, chunk_size):
        chunk_end = chunk_start + chunk_size
        blocks_subset = all_blocks[chunk_start:chunk_end]
        mod_pipe_transformer_f = TracingSingleTransformerBlockChunk(blocks_subset)

        single_transformer_block_neuron=torch_neuronx.trace(
            mod_pipe_transformer_f,
            sample_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,'compiler_workdir'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

        model_chunk_dir = os.path.join(compiled_model_path, "single_transformer_block")
        os.makedirs(model_chunk_dir, exist_ok=True)
        model_filename = os.path.join(model_chunk_dir,f'chunk_{chunk_start}.pt')
        torch.jit.save(single_transformer_block_neuron, model_filename)

        del single_transformer_block_neuron
    del transformer

    hidden_states = torch.rand([1, height * width // 256 + max_sequence_length, 3072],dtype=DTYPE)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],dtype=DTYPE)
    temb = torch.rand([1, 3072], dtype=DTYPE)
    sample_inputs = hidden_states, encoder_hidden_states, temb

    pipe = FluxPipeline.from_pretrained(PIPELINE,torch_dtype=DTYPE)
    transformer: FluxTransformer2DModel = copy.deepcopy(pipe.transformer)
    del pipe

    mod_pipe_transformer_f = TracingTransformerOutLayerWrapper(
        transformer.norm_out, transformer.proj_out)

    transformer_out_layers_neuron=torch_neuronx.trace(
            mod_pipe_transformer_f,
            sample_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,'compiler_workdir'),
            compiler_args=["--enable-fast-loading-neuron-binaries","--optlevel=1"]
            )
    torch_neuronx.async_load(transformer_out_layers_neuron)

    transformer_out_layers_path = os.path.join(compiled_model_path, 'transformer_out_layers')
    os.makedirs(transformer_out_layers_path, exist_ok=True)
    model_filename = os.path.join(transformer_out_layers_path,'model.pt')
    torch.jit.save(transformer_out_layers_neuron,model_filename)

    del transformer_out_layers_neuron
    del transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    trace_flux(
        args.height,
        args.width,
        args.max_sequence_length)
