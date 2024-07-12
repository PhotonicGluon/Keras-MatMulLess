# Dev Containers

This directory contains the configuration for dev containers, which is used to initialize the development environment in Codespaces, Visual Studio Code, and JetBrains IDEs. The environment is installed with all the necessary dependencies for development and is ready for linting, formatting, and running tests.

## Visual Studio Code

[![CPU Dev Container](https://img.shields.io/static/v1?label=CPU%20Dev%20Container&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PhotonicGluon/Keras-MatMulLess)

If you already have Visual Studio Code and Docker installed, you can click the badge above or [here](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PhotonicGluon/Keras-MatMulLess) to get started. Clicking these links will cause VS Code to automatically install the Dev Containers extension if needed, clone the source code into a container volume, and spin up a dev container for use. Follow [this guide](https://code.visualstudio.com/docs/devcontainers/tutorial) for more details.

### CUDA

Keras-MML offers a `cuda` dev container for working with CUDA.

> [!IMPORTANT]  
> Edit the [`Dockerfile`](.devcontainer/cuda/Dockerfile) file to set up the architecture properly.
> By default it is using `amd64`. So, if you are on a `arm64` system, **uncomment the appropriate lines in the file**!

## JetBrains IDEs

Open either `.devcontainer/base/devcontainer.json` or `.devcontainer/cuda/devcontainer.json` in your JetBrains IDE. Click the docker icon to create a dev container. Follow [this guide](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html) for more details.

## GitHub Codespaces

Create a codespace for the repo by clicking the "Code" button on the main page of the repo, selecting the "Codespaces" tab, and clicking the "+". The configurations will automatically be used. Follow [this guide](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository) for more details.
