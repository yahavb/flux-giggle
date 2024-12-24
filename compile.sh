#!/bin/bash -x

myloc="$(dirname "$0")"

python "$myloc/text_encoder_1/compile.py"
python "$myloc/text_encoder_2/compile.py" 
python "$myloc/transformer/compile.py" 
python "$myloc/decoder/compile.py" 
