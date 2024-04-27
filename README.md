# library install
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas matplotlib
