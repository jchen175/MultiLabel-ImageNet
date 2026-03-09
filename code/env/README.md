# Environment Setup

We provide a Conda-based environment for reproducing our experiments.

1. **Create Environment**

   ```bash
   conda env create -f environment.yml
   conda activate multilabel_imagenet
   ```

   This installs Python 3.11 and all required packages from `requirements.txt`.

2. **Install Apex (Required for DeiT v3)**

   DeiT v3 requires NVIDIA Apex for training.

   Please install it manually:

   ```bash
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir \
     --no-build-isolation \
     --config-settings "--build-option=--cpp_ext" \
     --config-settings "--build-option=--cuda_ext" \
     .
   ```

3. **Environment Notes**

   Tested with:

   - Python 3.11
   - PyTorch 2.6 (CUDA 12.4)
   - timm 1.0.21
