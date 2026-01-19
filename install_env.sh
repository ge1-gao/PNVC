conda create -n video-coding
conda activate video-coding
conda install python==3.13.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install compressai accelerate pytorch-msssim timm deepspeed omegaconf