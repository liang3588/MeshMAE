# calculate the center of the patch, then calcualte the positional embedding
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import copy
import numpy as np
import math
from functools import partial

import torch
import timm.models.vision_transformer

from timm.models.vision_transformer import PatchEmbed, Block


class Head(nn.Module):
    def __init__(self, dim=1024):
        super(Head, self).__init__()
        self.head = nn.Linear(dim, 40)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        out = self.head(x)
        return out


class Linear_probe(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm):
        super(Linear_probe, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.dim = embed_dim
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.max_pooling = nn.MaxPool2d((64, 1))
        self.max_pooling2 = nn.MaxPool2d((256, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )
        self.head = nn.Linear(embed_dim, 40)
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):

        feats_patches = feats
        centers_patches = centers

        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        batch, channel, num_patches, *_ = feats_patches.shape
        tokens = self.to_patch_embedding(feats_patches)
        pos_emb = self.pos_embedding(center_of_patches)
        tokens = tokens + pos_emb

        for blk in self.blocks:
            tokens = blk(tokens)

        x = self.norm(tokens)
        # outcome = x[:, -1]
        zero_tokens = torch.zeros((batch, 256 - num_patches, self.dim), dtype=torch.float32).cuda()
        tokens = torch.cat((x, zero_tokens), dim=1)
        tokens = self.max_pooling2(tokens).squeeze(1)
        return tokens


class Mesh_baseline(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512, drop_path=0.1,
                 patch_size=64, norm_layer=nn.LayerNorm):
        super(Mesh_baseline, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.dim = embed_dim
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer,drop_path=drop_path)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        self.max_pooling = nn.MaxPool2d((64, 1))
        self.max_pooling2 = nn.MaxPool2d((256, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )
        self.head = nn.Linear(embed_dim, 40)
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):

        feats_patches = feats
        centers_patches = centers

        center_of_patches = torch.sum(centers_patches, dim=2) / 64

        pos_emb = self.pos_embedding(center_of_patches)

        batch, channel, num_patches, *_ = feats_patches.shape

        tokens = self.to_patch_embedding(feats_patches)

        tokens = tokens + pos_emb

        for blk in self.blocks:
            tokens = blk(tokens)

        x = self.norm(tokens)
        zero_tokens = torch.zeros((batch, 256 - num_patches, self.dim), dtype=torch.float32).cuda()
        tokens = torch.cat((x, zero_tokens), dim=1)
        tokens = self.max_pooling2(tokens).squeeze(1)
        x = self.head(tokens)

        return x


class Mesh_baseline_seg(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, seg_part=4, drop_path =0.2, fpn=False, face_pos=False):
        super(Mesh_baseline_seg, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.face_pos = face_pos
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear((patch_dim) * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.to_face_embedding = nn.Sequential(
            Rearrange('b c h p -> b h p c', p=patch_size),
            nn.Linear(patch_dim if not self.face_pos else patch_dim+3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.fpn = fpn
        if self.fpn:
            self.linears = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)), 
                                        nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)),
                                        nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))])
            
        self.max_pooling = nn.MaxPool2d((64, 1))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, seg_part)
        )
        self.head = nn.Linear(embed_dim, seg_part)
        self.head1 = nn.Linear(embed_dim*2, seg_part)
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):

        feats_patches = feats
        
        
        centers_patches = centers

        batch, channel, num_patches, *_ = feats_patches.shape
        cordinates_patches = cordinates
        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        pos_emb = self.pos_embedding(center_of_patches)
        center_of_patches = center_of_patches.unsqueeze(2).repeat(1,1,64,1)
        feats_patches = feats.permute(0,3,1,2)

        tokens = self.to_patch_embedding(feats_patches)
        if not self.face_pos:
            tokens_seg = self.to_face_embedding(feats_patches)
        else:
            face_pos = (centers_patches - center_of_patches).permute(0,3,1,2)
            tokens_seg = self.to_face_embedding(torch.cat([feats_patches, face_pos], dim=1))

        cls_tokens = self.cls_token.expand(feats_patches.shape[0], -1, -1)

        tokens = tokens + pos_emb
        tokens = torch.cat((tokens, cls_tokens), dim=1)
        # patch to encoder tokens

        tokens_s = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i % 4 == 3:
                tokens_s.append(tokens)

        if self.fpn:
            tokens = 0
            for l, t in zip(self.linears, tokens_s):
                tokens = tokens + l(t)

        x = self.norm(tokens)
        outcome = x[:, 0:-1]
        outcome = outcome.unsqueeze(2).repeat(1, 1, 64, 1)
        x = self.head(outcome)
        tokens_seg = torch.cat((tokens_seg, outcome), dim=3)
        x_seg = self.head1(tokens_seg)
        return x, x_seg


class Mesh_mae(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, weight=0.2):
        super(Mesh_mae, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.weight = weight
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.embed_dim = embed_dim
        self.decoer_pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # --------------------------------------------------------------------------

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.to_points = nn.Linear(decoder_embed_dim, 64 * 9)
        self.to_pointsnew = nn.Linear(decoder_embed_dim, 45 * 3)
        self.to_points_seg = nn.Linear(decoder_embed_dim, 9)
        self.to_features = nn.Linear(decoder_embed_dim, 64 * (channels))
        self.to_features_seg = nn.Linear(decoder_embed_dim, channels)
        self.build_loss_func()
        self.initialize_weights()
        self.decoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.max_pooling = nn.MaxPool2d((256, 1))

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()

    def initialize_weights(self):

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):

        minFs = min(Fs)
        min_patch_number = minFs / 64

        min_patch_number = int(min_patch_number.detach().cpu().numpy())
        feats_patches = feats
        centers_patches = centers
        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        batch, channel, num_patches, *_ = feats_patches.shape

        cordinates_patches = cordinates
        pos_emb = self.pos_embedding(center_of_patches)

        encoder_cls_token_pos = self.encoder_cls_token_pos.repeat(batch, 1, 1)

        tokens = self.to_patch_embedding(feats_patches)

        num_masked = int(self.masking_ratio * min_patch_number)

        rand_indices = torch.rand(batch, min_patch_number).argsort(dim=-1).cuda()

        left_indices = torch.rand(batch, num_patches - min_patch_number).argsort(dim=-1).cuda() + min_patch_number

        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        unmasked_indices = torch.cat((unmasked_indices, left_indices), dim=1)

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]
        cls_tokens = self.cls_token.expand(feats_patches.shape[0], -1, -1)
        tokens_unmasked = torch.cat((tokens_unmasked, cls_tokens), dim=1)
        pos_emb_a = torch.cat((pos_emb[batch_range, unmasked_indices], encoder_cls_token_pos), dim=1)
        tokens_unmasked = tokens_unmasked + pos_emb_a
        # print(tokens_unmasked.shape)
        # encoded_tokens = self.blocks(tokens_unmasked)
        for blk in self.blocks:
            tokens_unmasked = blk(tokens_unmasked)
        tokens_unmasked = self.norm(tokens_unmasked)
        encoded_tokens = tokens_unmasked

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.decoder_embed(encoded_tokens)
        mask_tokens = self.mask_token.repeat(batch, num_masked, 1)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)

        decoder_pos_emb = self.decoer_pos_embedding(center_of_patches)

        decoder_cls_token_pos = self.decoder_cls_token_pos.repeat(batch, 1, 1)
        decoder_pos_emb = torch.cat((decoder_pos_emb[batch_range, masked_indices],
                                     decoder_pos_emb[batch_range, unmasked_indices], decoder_cls_token_pos), dim=1)
        decoder_tokens = decoder_tokens + decoder_pos_emb
        # decoded_tokens = self.decoder_blocks(decoder_tokens)
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        decoded_tokens = decoder_tokens
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # splice out the mask tokens and project to pixel values
        recovered_tokens = decoded_tokens[:, :num_masked]
        pred_vertices_coordinates = self.to_pointsnew(recovered_tokens)
        faces_values_per_patch = feats_patches.shape[-1]
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates,
                                                  (batch, num_masked, 45, 3)).contiguous()

        # get the patches to be masked for the final reconstruction loss
        # print(pred_vertices_coordinates.shape, torch.sum(centers_patches[batch_range,masked_indices],dim=2).shape)
        center = torch.sum(centers_patches[batch_range, masked_indices], dim=2) / 64
        pred_vertices_coordinates = pred_vertices_coordinates + center.unsqueeze(2).repeat(1, 1, 45, 1)
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates, (batch * num_masked, 45, 3)).contiguous()
        cordinates_patches = cordinates_patches[batch_range, masked_indices]

        cordinates_patches = torch.reshape(cordinates_patches, (batch, num_masked, -1, 3)).contiguous()
        cordinates_unique = torch.unique(cordinates_patches, dim=2)
        cordinates_unique = torch.reshape(cordinates_unique, (batch * num_masked, -1, 3)).contiguous()
        masked_feats_patches = feats_patches[batch_range, :, masked_indices]

        pred_faces_features = self.to_features(recovered_tokens)
        pred_faces_features = torch.reshape(pred_faces_features, (batch, num_masked, channel, faces_values_per_patch))

        # calculate reconstruction loss
        # print(pred_vertices_coordinates.shape, cordinates_unique.shape)

        shape_con_loss = self.loss_func_cdl1(pred_vertices_coordinates, cordinates_unique)

        feats_con_loss = F.mse_loss(pred_faces_features, masked_feats_patches)
        # print(shape_con_loss, feats_con_loss)
        loss = feats_con_loss + self.weight * shape_con_loss
        #######################################################################
        # if you are going to show the reconstruct shape, please using the following codes
        # pred_vertices_coordinates = pred_vertices_coordinates.reshape(batch, num_masked, -1, 3)
        #return loss, masked_indices, unmasked_indices, pred_vertices_coordinates, cordinates
        #######################################################################
        return loss
