#!/bin/bash


conda create -n modscan python=3.10 -y
conda activate modscan


pip install diffusers transformers accelerate scipy safetensors
pip install Pillow --upgrade