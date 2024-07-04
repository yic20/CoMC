[ICML2024] Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition
===

## Get started
The code is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), please follow the instructions in [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) to install.

```bash
# Create a conda environment
conda create -y -n comc python=3.8

# Activate the environment
conda activate comc

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Clone this repo
git clone https://github.com/yic20/CoMC.git
cd CoMC

# Install Dassl
cd Dassl.pytorch-master/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```
## Dataset
Download the test images of MSCOCO-2014 from the official [cite](https://cocodataset.org/#download).

The text training data can be download from [here](https://pan.quark.cn/s/dfca05fd6e96).

## Testing
```bash
cd scripts
bash main_eval.sh coco2014_distill rn50_coco2014 coco

```

## Training

```bash
cd scripts  
bash main.sh coco2014_distill rn50_coco2014 experiment_1

```

## Generate your own dataset
We use the openAI [API](https://platform.openai.com/docs/overview) to generate text dataset. If you need to generate your own dataset, you need to fill in your openAI API key in generate_descriptions.py
```bash
python generate_descriptions.py

```

If you have any questions about the code, you can contact me through (yichengliu_e@163.com).
