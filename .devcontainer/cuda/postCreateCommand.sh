#!/bin/bash
poetry install --with dev,nvidia-gpu  # ADDITIONAL: Need to install GPU dependencies

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.bashrc

# Tell user which devcontainer they are on
echo "" >> ~/.bashrc  # Newline
echo "# Notify that user is on CUDA devcontainer" >> ~/.bashrc
echo "echo '+----------------------------------+'" >> ~/.bashrc
echo "echo '|                                  |'" >> ~/.bashrc
echo "echo -e '| \x1b[34;3mYou are on the \x1b[34;1;4mCUDA\x1b[0m\x1b[34;3m devcontainer\x1b[0m |'" >> ~/.bashrc
echo "echo '|                                  |'" >> ~/.bashrc
echo "echo '+----------------------------------+'" >> ~/.bashrc

# ADDITIONAL: Set up CUDNN path
CUDNN_PATH=$(dirname $(poetry run python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib" >> ~/.bashrc
