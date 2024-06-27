#!/bin/bash
poetry install --with dev,nvidia-gpu  # ADDITIONAL: Need to install GPU dependencies

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.bashrc

# ADDITIONAL: Set up CUDNN path
CUDNN_PATH=$(dirname $(poetry run python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib" >> ~/.bashrc
