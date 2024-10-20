pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --progress-bar off
pip install -qqq --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes --progress-bar off