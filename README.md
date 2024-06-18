[ICML2024] Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition
===

## Get started
The code is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), please follow the instructions in [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) to install.

You can download the text data [here](https://pan.quark.cn/s/dfca05fd6e96).
## Testing
```
cd scripts
bash main_eval.sh coco2014_distill rn50_coco2014 coco
```

## Training
```
cd scripts  
bash main.sh coco2014_distill rn50_coco2014 experiment_1
```

## Generate your own dataset
```
python generate_descriptions.py
```

If you have any questions about the code, you can contact me through (yichengliu_e@163.com).
