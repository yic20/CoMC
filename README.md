[ICML2024] Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition
===
## Testing
`
cd scripts
bash main_eval.sh coco2014_distill rn50_coco2014 experiment_1
`

## Training
`
cd scripts
bash main.sh coco2014_distill rn50_coco2014 experiment_1
`

## Generate your own dataset
`
python generate_descriptions.py
`

## Acknowledgement
We use code from TaI-DPT, CoOp, and Dassl. Thanks for their work!
