# Use the official Python image as a parent image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    swig \
    portaudio19-dev \
    libpulse-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --upgrade pip \
    && pip install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
