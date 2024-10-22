# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats, # 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])
            bev_queries, # 由nn.Embedding(2500, 256)生成
            bev_h, # 50
            bev_w, # 50
            grid_length=[0.512, 0.512], # (2.048,2.048)
            bev_pos=None, # bev_mask(全零初始化)经过LearnedPositionalEncoding之后，torch.Size([1, 256, 50, 50])
            prev_bev=None, # 第一帧时为None/ 之后为前一帧的bev特征 torch.Size([1, 2500, 256])
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)  # torch.Size([1, 6, 256, 15, 25])
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # torch.Size([2500, 256])->torch.Size([2500, 1, 256])
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # torch.Size([1, 256, 50, 50])->torch.Size([2500, 1, 256])

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy  # 生成和bev_queries数据类型一致的tensor，第一帧时为自身位置，值为tensor([[0., 0.]])，之后车辆移动，即与上一时刻相对偏移量

        if prev_bev is not None: # 有前一帧bev时
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2) # torch.Size([1, 2500, 256])->torch.Size([2500, 1, 256])
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1) # prev_bev[:, i]:torch.Size([2500, 256]) tmp_prev_bev:torch.Size([256, 50, 50])
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center) # F_t.rotate(img, matrix=matrix, interpolation=interpolation.value, expand=expand, fill=fill)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1) # torch.Size([256, 50, 50])->torch.Size([2500, 1, 256])
                    prev_bev[:, i] = tmp_prev_bev[:, 0] # 根据车辆转角，转动bev特征平面

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :] # torch.Size([1, 18])自车位姿信息
        can_bus = self.can_bus_mlp(can_bus)[None, :, :] # Linear(in_features=18, out_features=128, bias=True) Linear(in_features=128, out_features=256, bias=True)  torch.Size([1, 18])->torch.Size([1, 1, 256])
        bev_queries = bev_queries + can_bus * self.use_can_bus  # torch.Size([2500, 1, 256])

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # torch.Size([1, 6, 256, 15, 25])->torch.Size([6, 1, 375, 256])
            if self.use_cams_embeds: # cams_embeds nn.Parameter:torch.Size([6, 256])
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype) # level_embeds nn.Parameter:torch.Size([4, 256])
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) torch.Size([6, 1, 375, 256])->torch.Size([6, 375, 1, 256])

        bev_embed = self.encoder(
            bev_queries, # torch.Size([2500, 1, 256]) 没有前一帧的情况下就是由nn.Embedding(2500, 256)初始化的bev_queries，增加了相机位姿信息
            feat_flatten, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
            feat_flatten, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos, # torch.Size([2500, 1, 256]) bev_mask经过LearnedPositionalEncoding之后，torch.Size([1, 256, 50, 50])
            spatial_shapes=spatial_shapes, # 多层情况为每一层特征图的h和w
            level_start_index=level_start_index, # 多层情况为每一层的特征图展平后的索引
            prev_bev=prev_bev, # 第一帧为None/ 之后会根据当前帧自车朝向，转动bev特征，torch.Size([2500, 1, 256])
            shift=shift, # torch.Size([1, 2]) 与上一时刻相对偏移量
            **kwargs
        )

        return bev_embed # torch.Size([1, 2500, 256])

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats, # 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])
                bev_queries, # 由nn.Embedding(2500, 256)生成
                object_query_embed, # 由nn.Embedding(900, 512)生成
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512], # (2.048, 2.048)
                bev_pos=None, # bev_mask(全零初始化)经过LearnedPositionalEncoding之后，torch.Size([1, 256, 50, 50])
                reg_branches=None,
                cls_branches=None,
                prev_bev=None, # 由encoder生成的前N帧的bev特征 torch.Size([1, 2500, 256])
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats, # # 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])
            bev_queries, # 由nn.Embedding(2500, 256)生成
            bev_h,
            bev_w,
            grid_length=grid_length, # (2.048, 2.048)
            bev_pos=bev_pos, # bev_mask(全零初始化)经过LearnedPositionalEncoding之后，torch.Size([1, 256, 50, 50])
            prev_bev=prev_bev, # 由encoder生成的前N帧的bev特征 torch.Size([1, 2500, 256])
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims :torch.Size([1, 2500, 256])

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1) # query_pos:torch.Size([900, 256]) query:torch.Size([900, 256]) 由nn.Embedding(900, 512)生成并拆分为两组
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1) # torch.Size([900, 256])->torch.Size([1, 900, 256])
        query = query.unsqueeze(0).expand(bs, -1, -1) # torch.Size([900, 256])->torch.Size([1, 900, 256])
        reference_points = self.reference_points(query_pos) # self.reference_points:Linear(in_features=256, out_features=3, bias=True) torch.Size([1, 900, 256])->torch.Size([1, 900, 3])
        reference_points = reference_points.sigmoid() # torch.Size([1, 900, 3])
        init_reference_out = reference_points

        query = query.permute(1, 0, 2) # torch.Size([1, 900, 256])->torch.Size([900, 1, 256])
        query_pos = query_pos.permute(1, 0, 2) # torch.Size([1, 900, 256])->torch.Size([900, 1, 256])
        bev_embed = bev_embed.permute(1, 0, 2) # torch.Size([1, 2500, 256])->torch.Size([2500, 1, 256])

        inter_states, inter_references = self.decoder(
            query=query, # 由nn.Embedding(900, 512)初始化生成并拆分为两组取其中一组 torch.Size([900, 1, 256])
            key=None,
            value=bev_embed, # 由encoder生成的bev特征 torch.Size([2500, 1, 256])
            query_pos=query_pos, # 由nn.Embedding(900, 512)初始化生成并拆分为两组取其中另一组 torch.Size([900, 1, 256])
            reference_points=reference_points, # query_pos经过线性层得到 torch.Size([1, 900, 3])
            reg_branches=reg_branches, # 6 x [Linear(in_features=256, out_features=256, bias=True) Linear(in_features=256, out_features=256, bias=True) Linear(in_features=256, out_features=10, bias=True)]
            cls_branches=cls_branches, # None
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device), # torch.tensor([[50, 50]]
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
