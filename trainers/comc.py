import os.path as osp
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter
import pickle
from tqdm import tqdm
import pickle5 as pickle
import numpy as np
from thop import profile

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pdb
from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class MultiLabelClassifier_MLP(nn.Module):
    def __init__(self, nums_class):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, nums_class)
        )
    def forward(self, feature):
        logits = self.layers(feature)
        return logits

class MultiLabelClassifier(nn.Module):
    def __init__(self, nums_class):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(1024, nums_class, bias=False))
    
    def forward(self, feature):
        logit = self.adapter(feature)
        return logit
        
        

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True, if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_templates, return_interm_layers=False):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)
        self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        self.classifier = MultiLabelClassifier(nums_class=self.num_classes)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.templates = text_templates
    
    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    def forward(self, image=None, captions=None, if_test=False):
        if if_test:        
            image_feat = self.encode_image(image)
            text_features = self.templates
            b, c, h, w = image_feat.shape
            x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)
            # g_x = x.mean(0, keepdim=True)
            # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW)xBxC        
            
            x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
            x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
            image_features = x

            image_feature_, _ = self.model.visual.attnpool(image_feat, if_pos=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_feature_ /= image_feature_.norm(dim=-1, keepdim=True)
            # print(image_features.shape)
            
            
            sim = image_feature_@text_features.T.float()
            sim = (sim*100).softmax(dim=-1)
            mapped_embedding = sim@text_features.float()
            mapped_embedding /= mapped_embedding.norm(dim=-1,keepdim=True)
            fused_embedding = 0.7*image_feature_+0.3*mapped_embedding   
            logits_ = self.classifier(fused_embedding)
            
            sim_local = image_features@text_features.T.float()
            sim_local = (sim_local*100).softmax(dim=-1)
            mapped_embedding_l = sim_local@text_features.float()
            mapped_embedding_l /= mapped_embedding_l.norm(dim=-1, keepdim=True)
            fused_embedding_l = 0.7*image_features+0.3*mapped_embedding_l
            logits_local = self.classifier(fused_embedding_l)
            
            logits_local = torch.max(logits_local, dim = 0).values
            return logits_, logits_local
            
        else:
            image_feat = self.text_encoder(captions, None, if_embedding=False, if_sequence=True) 
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
        
            image_feature_ /= image_feature_.norm(dim=-1, keepdim=True)
        
            
            
            noise_std = 0.1
            noise = torch.randn(image_feature_.shape) * noise_std
            logits_ = self.classifier(image_feature_ + noise.cuda())
            
            # logits_local = torch.max(logits_local, dim = 0).values
            return logits_
            

            

@TRAINER_REGISTRY.register()
class comc(TrainerX):
    def model_inference(self, input):
        return self.model(input, if_test=True)
        # return self.model(None, input)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        
 

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, output_pos = self.model_inference(input)
            self.evaluator.process(output, label, output_pos)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self.text_templates = torch.load('/disk1/lyc/CoMC/suppl/coco_templates.pt', map_location='cuda:0')
        self.model = CustomCLIP(cfg, classnames, clip_model, text_templates=self.text_templates)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)
                                                        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.classifier, cfg.MODEL.INIT_WEIGHTS)
        

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.classifier, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("comc", self.model.classifier, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(None, image)
            if   self.cfg.TRAIN.LOSSFUNC == 'sigmoid':
                loss = norm_logits_BCEloss(output, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'focal':
                loss = sigmoid_focal_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'asl':
                loss = ASL_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'ranking':
                loss = ranking_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                loss = ranking_loss(output, label, scale_ = 1.0, margin_ = 1)
            else:
                loss = soft_cross_entropy(output, label)

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            print(state_dict)
