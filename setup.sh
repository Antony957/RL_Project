#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

conda create -n spr python=3.8 -y


conda run -n spr pip install pip==24.0
conda run -n spr pip install wheel==0.38.4 setuptools==65.5.0
conda run -n spr pip install -r requirements.txt --no-build-isolation
