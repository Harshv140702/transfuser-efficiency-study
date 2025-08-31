import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
import re

class TransfuserBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, m1, m2, use_velocity=True):
        super().__init__()
        self.config = config
        self.use_velocity = use_velocity

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        # Use timm to create a pretrained efficientnet_b0
        # We create a temporary model with features_only=True to get channel info
        temp_encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        feature_info = temp_encoder.feature_info

        # Create the actual encoders for use in the model
        self.image_encoder = timm.create_model('efficientnet_b0', pretrained=True)
        self.lidar_encoder = timm.create_model('efficientnet_b0', pretrained=False) # Not pretrained for LiDAR modality

        # Modify the LiDAR encoder's first convolution to accept the specified number of input channels
        if config.use_point_pillars:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
        if config.use_target_point_image:
            in_channels += 1
            
        # Get the original weights and parameters from the stem convolution
        out_channels = self.lidar_encoder.conv_stem.out_channels
        kernel_size = self.lidar_encoder.conv_stem.kernel_size
        stride = self.lidar_encoder.conv_stem.stride
        padding = self.lidar_encoder.conv_stem.padding
        bias = self.lidar_encoder.conv_stem.bias is not None
        
        # Create a new conv layer with the correct number of input channels
        self.lidar_encoder.conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, bias=bias)

        # Initialize the Transformers with the correct embedding dimensions from EfficientNet blocks
        self.transformer1 = GPT(n_embd=feature_info[1]['num_chs'], n_head=config.n_head, block_exp=config.block_exp,
                                n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors,
                                img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors,
                                lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop, config=config, use_velocity=self.use_velocity)

        self.transformer2 = GPT(n_embd=feature_info[2]['num_chs'], n_head=config.n_head, block_exp=config.block_exp,
                                n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors,
                                img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors,
                                lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop, config=config, use_velocity=self.use_velocity)

        self.transformer3 = GPT(n_embd=feature_info[3]['num_chs'], n_head=config.n_head, block_exp=config.block_exp,
                                n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors,
                                img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors,
                                lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop, config=config, use_velocity=self.use_velocity)

        self.transformer4 = GPT(n_embd=feature_info[4]['num_chs'], n_head=config.n_head, block_exp=config.block_exp,
                                n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors,
                                img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors,
                                lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop, config=config, use_velocity=self.use_velocity)
        
        # Channel reduction convs
        final_conv_channels = self.image_encoder.conv_head.out_channels
        if final_conv_channels != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(final_conv_channels, self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(final_conv_channels, self.config.perception_output_features, (1, 1))
        else:
            self.change_channel_conv_image = nn.Identity()
            self.change_channel_conv_lidar = nn.Identity()

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image (tensor): input image tensor
            lidar (tensor): input LiDAR BEV tensor
            velocity (tensor): input velocity from speedometer
        '''
        image_tensor = normalize_imagenet(image)
        lidar_tensor = lidar

        # Initial feature extraction (stem)
        image_features = self.image_encoder.act1(self.image_encoder.bn1(self.image_encoder.conv_stem(image_tensor)))
        lidar_features = self.lidar_encoder.act1(self.lidar_encoder.bn1(self.lidar_encoder.conv_stem(lidar_tensor)))
        
        # Stage 1 fusion
        image_features = self.image_encoder.blocks[0:2](image_features)
        lidar_features = self.lidar_encoder.blocks[0:2](lidar_features)
        image_embd1 = self.avgpool_img(image_features)
        lidar_embd1 = self.avgpool_lidar(lidar_features)
        image_fused1, lidar_fused1 = self.transformer1(image_embd1, lidar_embd1, velocity)
        image_fused1 = F.interpolate(image_fused1, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused1 = F.interpolate(lidar_fused1, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + image_fused1
        lidar_features = lidar_features + lidar_fused1

        # Stage 2 fusion
        image_features = self.image_encoder.blocks[2:3](image_features)
        lidar_features = self.lidar_encoder.blocks[2:3](lidar_features)
        image_embd2 = self.avgpool_img(image_features)
        lidar_embd2 = self.avgpool_lidar(lidar_features)
        image_fused2, lidar_fused2 = self.transformer2(image_embd2, lidar_embd2, velocity)
        image_fused2 = F.interpolate(image_fused2, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused2 = F.interpolate(lidar_fused2, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + image_fused2
        lidar_features = lidar_features + lidar_fused2
        
        # Stage 3 fusion
        image_features = self.image_encoder.blocks[3:5](image_features)
        lidar_features = self.lidar_encoder.blocks[3:5](lidar_features)
        image_embd3 = self.avgpool_img(image_features)
        lidar_embd3 = self.avgpool_lidar(lidar_features)
        image_fused3, lidar_fused3 = self.transformer3(image_embd3, lidar_embd3, velocity)
        image_fused3 = F.interpolate(image_fused3, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused3 = F.interpolate(lidar_fused3, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + image_fused3
        lidar_features = lidar_features + lidar_fused3

        # Stage 4 fusion
        image_features = self.image_encoder.blocks[5:7](image_features)
        lidar_features = self.lidar_encoder.blocks[5:7](lidar_features)
        image_embd4 = self.avgpool_img(image_features)
        lidar_embd4 = self.avgpool_lidar(lidar_features)
        image_fused4, lidar_fused4 = self.transformer4(image_embd4, lidar_embd4, velocity)
        image_fused4 = F.interpolate(image_fused4, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused4 = F.interpolate(lidar_fused4, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + image_fused4
        lidar_features = lidar_features + lidar_fused4

        # Pass through head
        image_features = self.image_encoder.act2(self.image_encoder.bn2(self.image_encoder.conv_head(image_features)))
        lidar_features = self.lidar_encoder.act2(self.lidar_encoder.bn2(self.lidar_encoder.conv_head(lidar_features)))

        # Downsample channels to perception_output_features
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)
        
        x4 = lidar_features
        image_features_grid = image_features

        # Global pooling and feature flattening
        image_features_pooled = self.image_encoder.global_pool(image_features)
        image_features_pooled = torch.flatten(image_features_pooled, 1)
        lidar_features_pooled = self.lidar_encoder.global_pool(lidar_features)
        lidar_features_pooled = torch.flatten(lidar_features_pooled, 1)

        fused_features = image_features_pooled + lidar_features_pooled

        features = self.top_down(x4)
        return features, image_features_grid, fused_features


class SegDecoder(nn.Module):
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_class = config.num_class

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.num_class, 3, 1, 1),
                    )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)

        return x


class DepthDecoder(nn.Module):
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, 1, 3, 1, 1),
                    )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        x = torch.sigmoid(x).squeeze(1)

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    img_vert_anchors, img_horz_anchors, 
                    lidar_vert_anchors, lidar_horz_anchors,
                    seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config, use_velocity=True):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * img_vert_anchors * img_horz_anchors + self.seq_len * lidar_vert_anchors * lidar_horz_anchors, n_embd))
        
        # velocity embedding
        self.use_velocity = use_velocity
        if(use_velocity == True):
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]
        
        assert self.seq_len == 1
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        # project velocity to n_embed
        if(self.use_velocity==True):
            velocity_embeddings = self.vel_emb(velocity) # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) #(B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)

        x = x.view(bz, self.seq_len*self.img_vert_anchors*self.img_horz_anchors + self.seq_len*self.lidar_vert_anchors*self.lidar_horz_anchors, self.n_embd)

        image_tensor_out = x[:, :self.seq_len*self.img_vert_anchors*self.img_horz_anchors, :].contiguous().view(bz * self.seq_len, -1, img_h, img_w)
        lidar_tensor_out = x[:, self.seq_len*self.img_vert_anchors*self.img_horz_anchors:, :].contiguous().view(bz * self.seq_len, -1, lidar_h, lidar_w)

        return image_tensor_out, lidar_tensor_out




def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x

class EfficientSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Combined qkv projection for better memory locality
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        B, T, C = x.size()
        
        # Single matrix multiplication for q, k, v
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (B, nh, T, hd)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output projection
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = EfficientSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            # nn.GELU(), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x