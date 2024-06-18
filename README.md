[ICML2024] Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition
===
## Training
`
cd scripts
bash main.sh coco2014_distill rn50_coco2014 experiment_1
`
## Testing
`
cd scripts
bash main_eval.sh coco2014_distill rn50_coco2014 experiment_1
`

## Acknowledgement
We borrow code from TaI-DPT, CoOp, and Dassl. Thanks for their work!
