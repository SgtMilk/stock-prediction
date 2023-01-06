# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Setup working dorectory
WORKDIR /app

# We are copying this before everything else because it's faster
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /app/requirements.txt

# Set environment variables

# Runtime operations (CMD)
