# Thanks to PierrunoYT for making the original version that I took 99% of this from.
# I only added the audio book side
# Original repository found here https://github.com/PierrunoYT/Kokoro-TTS-Local

# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading, automatic dependency management, and a web interface.

## Features

- Local text-to-speech synthesis using the Kokoro-82M model
- Multiple voice support with easy voice selection (31 voices available)
- Automatic model and voice downloading from Hugging Face
- Phoneme output support and visualization
- Interactive CLI and web interface
- Automated ebook dubbing
- Voice listing functionality
- Cross-platform support (Windows, Linux, macOS)
- Real-time generation progress display
- Multiple output formats (WAV, MP3, AAC)

## Prerequisites

- Python 3.8 or higher
- FFmpeg (optional, for MP3/AAC conversion)
- CUDA-compatible GPU (optional, for faster generation)
- Git (for version control and package management)


## Installation

INSTRUCTIONS FOR WINDOWS ONLY

## Clone repository and make virtual environment
	git clone https://github.com/MicrowavedBred/KokoroAudioBook
	python -m venv (desired name for venv)
	(name you chose)\Scripts\activate

## Install dependencies
	pip install -r requirements.txt	

## Install FFMPEG
Download FFMPEG

	https://ffmpeg.org/
	After downloading, extract it somewhere and place this location in system path variables.
	(Location you saved it)\ffmpeg-master-latest-win64-gpl-shared\bin

## Enable CUDA
To enable CUDA you need to install cuda toolkit 12.4. (Latest supported version for pytorch)

	https://developer.nvidia.com/cuda-12-4-0-download-archive

Add these CUDA 12.4 locations to system path variables.

	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp

If desired install the CUDA version of pytorch. (Much faster processing if you have an Nvidia GPU)

	# While in your virtual environment
	pip uninstall torch
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Test CUDA functionality

	# With the virtual environment enabled, you can test if CUDA works with these commands.
	import torch
	print("CUDA Available:", torch.cuda.is_available())
	print("CUDA Version:", torch.version.cuda)
	print("PyTorch Version:", torch.__version__)


## Usage
Once everything is installed run either of these scripts from the virtual environment:

	# Model and voice will install automatically when either script is ran
	python tts_demo.py
	or
	python AIAudioBook.py

tts_demo.py is great for testing different voices with shorter text.
AIAudioBook.py is only for dubbing audio books.

If you have any issues, I've only tested this on python 3.9.
