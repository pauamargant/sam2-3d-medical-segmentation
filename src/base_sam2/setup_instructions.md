# Native SAM2 Setup Instructions
This documents provides instructions for setting up and installing the Native SAM2 implementation.
Based on meta's SAM2 repository, keeping only the implementation itself.

To install the module use:
```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```
To download the pre-trained weights, use:
```bash
mkdir checkpoints && \
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```