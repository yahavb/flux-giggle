import argparse
import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from model import TracingT5TextEncoderWrapper

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)


def trace_text_encoder_2(max_sequence_length):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    text_encoder_2 = copy.deepcopy(pipe.text_encoder_2)
    del pipe

    text_encoder_2 = TracingT5TextEncoderWrapper(text_encoder_2)

    emb = torch.zeros((1, max_sequence_length), dtype=torch.int64)

    text_encoder_2_neuron = torch_neuronx.trace(
        text_encoder_2.neuron_text_encoder,
        emb,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

    torch_neuronx.async_load(text_encoder_2_neuron)

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                           'compiled_model/model.pt')
    torch.jit.save(text_encoder_2_neuron, text_encoder_2_filename)

    del text_encoder_2
    del text_encoder_2_neuron


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=32,
        help="maximum sequence length for the text embeddings"
    )
    args = parser.parse_args()
    trace_text_encoder_2(args.max_sequence_length)
