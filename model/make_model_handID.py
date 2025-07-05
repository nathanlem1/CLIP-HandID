"""
This code adapts CLIP image encoder to handID to train on hands dataset for person identification based on hand images.
The text encoder is not used i.e. it is frozen (see in 'train_handID.py').
"""
import math
import os
import torch
import torch.nn as nn
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class TextEncoderCustom(nn.Module):
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

    def forward(self, pseudo_token_embeddings, tokenized_prompts):
        x = pseudo_token_embeddings + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        #  The activations of the highest layer of the transformer at the [EOS] token are used as the feature
        #  representation of the text, which is layer normalized and then linearly projected into the multi-modal
        # embedding space. # x.shape = [batch_size, n_ctx, transformer.width]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # Joint multimodal
        # embedding
        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, args):
        super(build_transformer, self).__init__()
        self.model_name = args.backbone_name
        self.is_learn_tokens = args.is_learn_tokens
        self.is_interaction_network = args.is_interaction_network
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
        # For text-image
        self.classifier_proj_txt_img = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj_txt_img.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        # For text-image
        self.bottleneck_proj_txt_img = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj_txt_img.bias.requires_grad_(False)
        self.bottleneck_proj_txt_img.apply(weights_init_kaiming)

        self.h_resolution = int((args.input_size[0]-16)//args.stride_size[0] + 1)
        self.w_resolution = int((args.input_size[1]-16)//args.stride_size[1] + 1)
        self.vision_stride_size = args.stride_size[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if self.is_learn_tokens:
            self.text_encoder = TextEncoderCustom(clip_model)
        else:
            self.text_encoder = TextEncoder(clip_model)

        if self.is_learn_tokens:
            # self.inversion_network_transformer = InversionNetworkTransformer(clip_model, self.in_planes_proj)
            self.inversion_network_mlp = InversionNetworkMLP(self.in_planes_proj, self.in_planes_proj * 4)
            if self.is_interaction_network:
                self.interaction_network = InteractionNetwork(self.in_planes_proj)

    def forward(self, x):
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)  # B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]  # Take classifier "token" (global embedding), leaving out
            # local features (embeddings) corresponding to 14x14 patches (196).#
            image_features_proj = image_features_proj.permute(1, 0, 2)  # For inversion_network input

        elif self.model_name == 'ViT-B/16':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            # Take classifier "token" (global embedding), leaving out local features (embeddings) corresponding to
            # 14x14 patches (196).
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        if self.is_learn_tokens:
            # pseudo_token_embeddings = self.inversion_network_transformer(image_features_proj)  # Uses Transformer
            # pseudo_token_embeddings = self.inversion_network_transformer(img_feature_proj)  # Uses Transformer
            pseudo_token_embeddings = self.inversion_network_mlp(img_feature_proj)  # Uses MLP

            text_features = encode_text_with_placeholders(self.text_encoder, pseudo_token_embeddings)

            if self.is_interaction_network:
                # # --- Multi-modal interaction -----------------
                # # Method 1
                # # text_features_new = text_features.unsqueeze(1).expand(image_features_proj.shape)
                # # Concatenate text_features and image patches
                # text_features_new = torch.concat((text_features.unsqueeze(1), image_features_proj[:, 1:]), dim=1)
                # image_text_features_proj = self.interaction_network(text_features_new, image_features_proj,
                #                                                      image_features_proj)
                # img_txt_feature_proj = image_text_features_proj[:, 0, :] #Take first row of 197 (B,197,in_planes_proj]

                # Method2, NB: Both methods give similar result but method 2 is computationally cheap!
                img_txt_feature_proj = self.interaction_network(text_features, img_feature_proj, img_feature_proj)
                # ---------------------------------------------

            # Apply Batch Normalization
            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)
            if self.is_interaction_network:
                feat_proj_txt_img = self.bottleneck_proj_txt_img(img_txt_feature_proj)  # For text-image interaction

            # Apply classifier
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            if self.is_interaction_network:
                cls_score_proj_txt_img = self.classifier_proj_txt_img(feat_proj_txt_img)  # For text-image interaction

            if self.is_interaction_network:
                return [cls_score, cls_score_proj, cls_score_proj_txt_img], [text_features, img_feature_proj,
                        img_txt_feature_proj], [torch.cat([img_feature, img_feature_proj, img_txt_feature_proj], dim=1),
                        torch.cat([feat, feat_proj, feat_proj_txt_img], dim=1)]
            else:
                # If interaction_network is NOT used but only inversion_network is used.
                return [cls_score, cls_score_proj], [text_features, img_feature_proj], [
                        torch.cat([img_feature, img_feature_proj], dim=1), torch.cat([feat, feat_proj], dim=1)]
        else:
            # Apply Batch Normalization
            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)

            # Apply classifier
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)

            return [cls_score, cls_score_proj], [img_feature, img_feature_proj], [
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


# Inversion Network
class InversionNetworkMLP(nn.Module):
    def __init__(self, image_embedding_dim: int, hidden_dim: int, num_tokens=1, token_dim=512, dropout=0.5):
        """
        Inversion Network using Multi-layer Perceptron (MLP)
        Takes as input the visual features of an image and outputs the pseudo-word embedding.

        Args:
            image_embedding_dim (int): CLIP image encoder feature dimension
            hidden_dim (int): MLP hidden dimension (eg. 512), we set hidden_dim = image_embedding_dim * 4
            num_tokens (int): Number of pseudo-word tokens (can be >=1)
            token_dim (int): Dimension of each token (output dimension)
            dropout (float): Dropout probability (e.g. 0.1, 0.5)
        """
        super(InversionNetworkMLP, self).__init__()

        # A 3-layer MLP with LayerNorm and Dropout for mapping visual features to a sequence of pseudo-word tokens.
        self.visual_to_token = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, num_tokens * token_dim),
            nn.LayerNorm(num_tokens * token_dim)
        )

        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, x):
        """
        Input: (batch_size, image_embedding_dim) -> Output: (batch_size, num_tokens, token_dim)
        x - is visual features
        """

        # Learnable MLP transformation to map visual features to pseudo-word tokens
        tokens = self.visual_to_token(x)  # Shape: (batch_size, num_tokens * token_dim)

        if self.num_tokens > 1:  # Reshape to (batch_size, num_tokens, token_dim)
            tokens = tokens.view(tokens.size(0), self.num_tokens, self.token_dim)

        return tokens


class InversionNetworkTransformer(nn.Module):
    """
    Inversion Network using Transformer
        Takes as input the visual features of an image and outputs the pseudo-word embedding.
    """
    def __init__(self, clip_model, image_embedding_dim: int, hidden_dim=512, num_pseudo_tokens=4, token_dim=512):
        super(InversionNetworkTransformer, self).__init__()

        self.clip_model = clip_model
        self.num_pseudo_tokens = num_pseudo_tokens  # Can be >=1
        self.token_dim = token_dim  # token_dim = self.clip_model.token_embedding.weight.shape[-1]  # 512 for ViT-B/16
        self.tokenizer = clip.tokenize

        # Patch projection
        self.patch_proj = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(num_pseudo_tokens, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            # norm_first=True,  # layer norm is done prior to attention and feedforward operations.
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)  # Adjust num_layers: 2, 3, etc

        # Output projection
        self.to_clip_space = nn.Linear(hidden_dim, self.token_dim)

        # Initialize with CLIP's [EOS] embedding
        device = self.clip_model.token_embedding.weight.device
        with torch.no_grad():
            eos_id = self.tokenizer("")[0][-1]  # EOS token
            eos_embed = self.clip_model.token_embedding(torch.tensor([eos_id]).to(device))
            self.to_clip_space.weight.data.normal_(mean=eos_embed.mean(), std=0.02)

    def forward(self, img_embeddings):
        """
        Args:
            img_embeddings: [batch, num_patches, vit_embed_dim] OR [batch, embed_dim]
                           (e.g., [B, 197, 1024] for ViT-B/16  OR [B, 1024])
        Returns:
            pseudo_tokens: [batch, num_pseudo_tokens, clip_token_dim]
        """
        batch_size = img_embeddings.shape[0]

        # 1. Project patches
        patch_feats = self.patch_proj(img_embeddings)  # [B, 197, hidden_dim]

        # 2. Prepare transformer inputs
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_tokens, hidden_dim]
        if img_embeddings.dim() == 2:
            patch_feats = patch_feats.unsqueeze(1)  # img_embeddings is of size [B, hidden_dim]
        transformer_input = torch.cat([queries, patch_feats], dim=1)  # [B, num_tokens+197, hidden_dim]

        # 3. Process through transformer
        outputs = self.transformer(transformer_input)
        pseudo_tokens = outputs[:, :self.num_pseudo_tokens]  # [B, num_tokens, hidden_dim]

        # 4. Project to CLIP token space
        clip_tokens = self.to_clip_space(pseudo_tokens)  # [B, num_tokens, token_dim]

        return clip_tokens


@torch.no_grad()
def encode_text_with_placeholders(text_encoder, pseudo_token_embeddings):
    """
    Use the CLIP model to encode a text with pseudo tokens.
    Process batch of prompts with placeholder tokens

    The original CLIP implementation from OpenAI doesn't officially support adding new tokens to its vocabulary.
    Our approach: Token Substitution + Embedding Replacement
    Since we can't modify CLIP's tokenizer, we'll:
        1. Use an existing rare token as our placeholder (e.g., "sks", "xxx", "ztoken", "qqq", "abc")
        2. Replace its embedding with our learned pseudo-token during inference

    Args:
        text_encoder: Customised text encoder from the CLIP model
        pseudo_token_embeddings: Pseudo token embeddings obtained from visual features to pseudo token inversion.
    Returns:
        text_features: [batch_size, embed_dim]
    """
    batch_size = pseudo_token_embeddings.shape[0]

    # Choose a rare token as our placeholder.
    # Recommended Placeholder Tokens, ordered by effectiveness: ["sks", "xxx", "ztoken", "qqq", "abc"]
    placeholder_token = "sks"  # Rarely used in natural language, best performing in practice.
    # placeholder_token = "$"  # This is also possible, placeholder_token = 259 for "$".
    placeholder_id = clip.tokenize(placeholder_token)[0][1]  # Get token ID

    # Process batch of prompts. Here "sks" (rare token) is used as a placeholder token for a "s*" token used in some
    # literatures as 'A photo of a s*', in our case the composed prompt is 'A photo of a s* hand'
    prompts = ["A photo of a sks hand"] * batch_size  # Can be different prompts of length batch_size
    # prompts = ["A photo of a $ hand"] * batch_size  # In case placeholder_token = "$" is used!

    # Tokenize all prompts i.e. it produces tokens
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(pseudo_token_embeddings.device)  # 77 is context
    # length which refers to the maximum number of tokens (words, sub-words, or characters) that the model can process
    # in a single input sequence. In this case, we use word-level tokenization.

    # Get base embeddings
    text_embeddings = text_encoder.token_embedding(text_inputs)  # [batch_size, n_ctx, d_model], n_ctx = 77

    # Find placeholder positions
    placeholder_positions = (text_inputs == placeholder_id).nonzero(as_tuple=True)

    # Replace placeholders with corresponding embeddings. Average all pseudo-tokens (or use first token).
    # This replaces the word embedding of the placeholders with the pseudo tokens for each element in the batch.
    for batch_idx, token_idx in zip(*placeholder_positions):
        if batch_idx < batch_size:  # Safety check
            text_embeddings[batch_idx, token_idx] = pseudo_token_embeddings[
                batch_idx % len(pseudo_token_embeddings)].mean(dim=0, keepdim=True)

    # Encode through CLIP text_encoder
    with torch.no_grad():
        text_features = text_encoder(text_embeddings, text_inputs)
    return text_features


# Interaction Network
class InteractionNetwork(nn.Module):
    """
    Multi-modal interaction network (text and image)
    """
    def __init__(self, embedding_dim: int):
        super(InteractionNetwork, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads=8)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        # # This enforced image_encoder to produce nan values (starting from epoch =30).
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Adjust num_layers: 1, 2, 3, etc.

        self.feedforward1 = nn.Linear(embedding_dim, embedding_dim)
        self.feedforward2 = nn.Linear(embedding_dim, embedding_dim)
        self.multihead_attn1 = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.multihead_attn2 = nn.MultiheadAttention(embedding_dim, num_heads=8)

    def forward(self, Q, K, V):
        """
        Q: query from text encoder
        K: key from image encoder
        V: value from image encoder
        """
        cross_attention, _ = self.multihead_attn(Q, K, V)  # Cross-attention
        # output = cross_attention  # For using multi-head attention only
        # output = self.transformer(cross_attention)  # This enforced image_encoder to produce nan values (starting from
        # # epoch =30). Because of this, we use the following code.

        self_attention_out1, _ = self.multihead_attn1(cross_attention, cross_attention, cross_attention)
        feedforward_out1 = self.feedforward1(self_attention_out1)
        self_attention_out2, _ = self.multihead_attn2(feedforward_out1, feedforward_out1, feedforward_out1)
        output = self.feedforward2(self_attention_out2)

        return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HandID Baseline Training: Proposed end-to-end prompt learning and "
                                                 "CLIP image encoder fine-tuning on hands dataset for hand-based "
                                                 "person identification.")
    parser.add_argument('--input_size', type=tuple, default=(224, 224), help='')
    parser.add_argument('--stride_size', type=tuple, default=(16, 16), help='')
    parser.add_argument('--backbone_name', default='ViT-B/16', type=str,
                        help='Used backbone model name - RN50 for ResNet50 or ViT-B/16 for Vision Transformer.')
    args = parser.parse_args()

    num_classes = 72
    model = make_model(args, num_class=num_classes)
    print(model)

    print('ok')
