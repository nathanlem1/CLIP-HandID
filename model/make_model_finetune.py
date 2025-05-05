"""
This code adapts CLIP image encoder to fine-tune on hands dataset for person identification based on hand images. The
text encoder is not used i.e. it is frozen (see in 'train_finetune.py').
"""
import torch
import torch.nn as nn
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.vocab_size = clip_model.vocab_size
        self.context_length = clip_model.context_length

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        #  The activations of the highest layer of the transformer at the [EOS] token are used as the feature
        #  representation of the text, which is layer normalized and then linearly projected into the multi-modal
        # embedding space. # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # Joint multimodal embedding

        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, args):
        super(build_transformer, self).__init__()
        self.model_name = args.backbone_name
        # self.model_name = cfg.MODEL.NAME
        # self.cos_layer = cfg.MODEL.COS_LAYER
        # self.neck = cfg.MODEL.NECK
        # self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B/16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((args.input_size[0]-16)//args.stride_size[0] + 1)
        self.w_resolution = int((args.input_size[1]-16)//args.stride_size[1] + 1)
        self.vision_stride_size = args.stride_size[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x):
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]  # Take classifier "token" (global embedding), leaving out
            # local features (embeddings) corresponding to 14x14 patches (196).

        elif self.model_name == 'ViT-B/16':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # B,512  B,128,512
            # Take classifier "token" (global embedding), leaving out local features (embeddings) corresponding to
            # 14x14 patches (196).
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        else:
            raise ValueError('')

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)
        return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], [
            torch.cat([img_feature, img_feature_proj], dim=1), torch.cat([feat, feat_proj], dim=1)]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(args, num_class):
    model = build_transformer(num_class, args)
    return model


from .clip import clip
# from clip import clip  # For testing this code
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HandID Baseline Training: Fine-tuning CLIP image encoder")
    parser.add_argument('--input_size', type=tuple, default=(224, 224), help='')
    parser.add_argument('--stride_size', type=tuple, default=(16, 16), help='')
    parser.add_argument('--backbone_name', default='ViT-B/16', type=str,
                        help='Used backbone model name - RN50 for ResNet50 or ViT-B/16 for Vision Transformer.')
    args = parser.parse_args()

    num_classes = 72  # Just to test
    model = make_model(args, num_class=num_classes)
    print(model)

    print('ok')
