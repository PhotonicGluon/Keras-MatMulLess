#!/bin/bash
poetry install --with dev

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.bashrc

# Tell user which devcontainer they are on
echo "" >> ~/.bashrc  # Newline
echo "# Notify that user is on CPU devcontainer" >> ~/.bashrc
echo "echo '+---------------------------------+'" >> ~/.bashrc
echo "echo '|                                 |'" >> ~/.bashrc
echo "echo -e '| \x1b[34;3mYou are on the \x1b[34;1;4mCPU\x1b[0m\x1b[34;3m devcontainer\x1b[0m |'" >> ~/.bashrc
echo "echo '|                                 |'" >> ~/.bashrc
echo "echo '+---------------------------------+'" >> ~/.bashrc
