# borrowed from VILA install_env.sh
conda create -n vtimellm python=3.10 -y # make sure you install python 3.10
conda activate vtimellm 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# change torch, torchvision in pyproject.toml to the correct version

pip install --upgrade pip  # enable PEP 660 support
# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda-toolkit -y
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install git+https://github.com/huggingface/transformers@v4.36.2

