#!/bin/bash
source /venv/bin/activate
poetry install --with dev

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.bashrc
