import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat
from .vit_utils import DropPath, trunc_normal_


class TLCA(nn.Module):
    def __init__(self):
        super(TLCA, self).__init__()

    def forward(self, x, pred, b):
        residual = x
        score = torch.sigmoid(pred)
        # dist = torch.abs(score - 0.5)
        # att = 1 - (dist / 0.5)

        l = x.shape[0] // b
        score = rearrange(score, '(b l) c h w -> b l c h w', b=b)
        dist = torch.mean(score, dim=1)
        dist = repeat(dist, 'b c h w -> b l c h w', l=l)
        dist = torch.abs(score - dist)
        att = dist
        # att = 1 - (dist / 0.5)
        att = rearrange(att, 'b l c h w -> (b l) c h w')

        att_x = x * att

        out = att_x + residual

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1)) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x + res_temporal

            ## Spatial
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = x + res
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PFSA(nn.Module):
    """ Vision Transformere
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=512, depth=1,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=11,
                 attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = 64

        ## Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, b):
        B, C, H, W = x.shape  # 44, 512, 8, 8
        T = B // b
        B = b
        # x, T, W = self.patch_embed(x)
        x = rearrange(x, 'bt c h w -> bt (h w) c')

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            raise AssertionError('x.size(1) != self.pos_embed.size(1)!')
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                raise AssertionError("T != self.time_embed.size(1)")
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)
        x = self.norm(x)

        x = rearrange(x, 'b (h w t) c -> (b t) c h w', b=B, t=T, c=C, h=H, w=W)

        return x

    def forward(self, x, b):
        x = self.forward_features(x, b)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class VACSNet(nn.Module):
    """
    The video-based ACSNet
    Refer to: Adaptive Context Selection for Polyp Segmentation (MICCAI 2020)
    https://github.com/ReaFly/ACSNet
    """

    def __init__(self, num_classes=1, cfg=None):
        super(VACSNet, self).__init__()

        self.cfg = cfg
        resnet = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=768, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=192, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=128, out_channels=64)

        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                     nn.Dropout2d(0.1),
                                     nn.Conv2d(32, num_classes, 1))

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)

        # local context attention module
        self.lca1 = TLCA()
        self.lca2 = TLCA()
        self.lca3 = TLCA()
        self.lca4 = TLCA()
        self.use_tlca = True  # cfg.MODEL.TLCA

        # global context module
        self.plsa = PFSA(num_frames=cfg.video_time_clips)
        self.use_plsa = True  # cfg.MODEL.PFTSA

    def forward(self, x):
        # x 224
        b = x.shape[0]
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  # 56
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)  # 28
        e4 = self.encoder4(e3)  # 14
        e5 = self.encoder5(e4)  # 7

        # global_contexts = self.gcm(e5)
        if self.use_plsa:
            e5 = self.plsa(e5, b)

        d5 = self.decoder5(e5)  # 14
        out5 = self.sideout5(d5)
        if self.use_tlca:
            e4 = self.lca4(e4, out5, b)
        comb4 = torch.cat([d5, e4], dim=1)

        d4 = self.decoder4(comb4)  # 28
        out4 = self.sideout4(d4)
        if self.use_tlca:
            e3 = self.lca3(e3, out4, b)
        comb3 = torch.cat([d4, e3], dim=1)

        d3 = self.decoder3(comb3)  # 56
        out3 = self.sideout3(d3)
        if self.use_tlca:
            e2 = self.lca2(e2, out3, b)
        comb2 = torch.cat([d3, e2], dim=1)

        d2 = self.decoder2(comb2)  # 128
        out2 = self.sideout2(d2)
        if self.use_tlca:
            e1 = self.lca1(e1, out2, b)
        comb1 = torch.cat([d2, e1], dim=1)

        d1 = self.decoder1(comb1)  # 224*224*64
        out1 = self.outconv(d1)  # 224

        # return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
        #        torch.sigmoid(out4), torch.sigmoid(out5)
        out1 = rearrange(out1, '(b l) c h w -> b l c h w', b=b)
        out2 = rearrange(out2, '(b l) c h w -> b l c h w', b=b)
        out3 = rearrange(out3, '(b l) c h w -> b l c h w', b=b)
        out4 = rearrange(out4, '(b l) c h w -> b l c h w', b=b)
        out5 = rearrange(out5, '(b l) c h w -> b l c h w', b=b)

        return {"seg_final": out1, "out1": out2, "out2": out3, "out3": out4, "out4": out5}
