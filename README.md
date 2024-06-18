# Multispectral Snapshot Image Registration (MSIR)

The paper is currently under review in IEEE Transactions on Image Processing and available on arXiv:

https://arxiv.org/abs/2406.11284

If you use any of the code or data, please cite
```
@article{sippel2024,
   title={Multispectral Snapshot Image Registration Using Learned Cross Spectral Disparity Estimation and a Deep Guided Occlusion Reconstruction Network},
   author={Sippel, Frank and Seiler, Jürgen and Kaup, André},
   journal={arXiv preprint arXiv:2406.11284},
   year={2024}
}
```

# Anaconda environment

```
conda create -n msir python=3.11
conda activate msir
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install imageio
conda install matplotlib
conda install scipy
pip install opencv-python-headless
pip install timm==0.5.4
```

An environment file is provided as well.

# Run multispectral image registration

To run the code, simply acivate the anaconda environment and execute
```
python run.py
```
Then, the three real-world scenes of the paper are registered.
