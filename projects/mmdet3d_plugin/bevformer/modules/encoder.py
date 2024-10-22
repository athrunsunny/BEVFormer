
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1) # torch.Size([4, 50, 50, 3])
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # torch.Size([4, 2500, 3])
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1) # torch.Size([1, 4, 2500, 3])
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H # torch.Size([1, 2500])
            ref_x = ref_x.reshape(-1)[None] / W # torch.Size([1, 2500])
            ref_2d = torch.stack((ref_x, ref_y), -1) # torch.Size([1, 2500, 2])
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2) # torch.Size([1, 2500, 1, 2])
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # lidar2img相当于相机外参
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])  # mg_meta['lidar2img']:[[4,4]*6]
        lidar2img = np.asarray(lidar2img) # (1, 6, 4, 4)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()
        # reference_points = ref_3d # normalize 0~1 # torch.Size([1, 4, 50*50, 3])
        # pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # reference_points映射到自车坐标系下, 尺度被缩放为真实尺度
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        # bs, num_points_in_pillar, bev_h * bev_w, xyz1
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # torch.Size([1, 4, 2500, 4]) 齐次
        
        reference_points = reference_points.permute(1, 0, 2, 3) # torch.Size([1, 4, 2500, 4])->torch.Size([4, 1, 2500, 4])
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1) # 6
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, xyz1, None
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  #torch.Size([4, 1, 2500, 4])->torch.Size([4, 1, 6, 2500, 4, 1])
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, 4, 4
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1) #  (1, 6, 4, 4)->torch.Size([4, 1, 6, 2500, 4, 4])
        # Z*(u,v,1)^T = K*W2C*(X,Y,Z,1)^T
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, xys1
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1) # 将3d点转换坐标系
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps) # 取位于相机前方的点 torch.Size([4, 1, 6, 2500, 1])
        # 齐次坐标下除以比例系数得到图像平面的坐标真值
        # num_points_in_pillar, bs, num_cam, bev_h * bev_w, uv
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)  # 取二维点并归一化

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1] # 800
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0] # 480
        #从相机感受野的角度再删选一次
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)) # 过滤掉位于图像外的点
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # torch.Size([4, 1, 6, 2500, 2])->torch.Size([6, 1, 2500, 4, 2])
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # torch.Size([4, 1, 6, 2500, 1])->torch.Size([6, 1, 2500, 4])

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query, # torch.Size([2500, 1, 256]) 没有前一帧的情况下就是由nn.Embedding(2500, 256)初始化的bev_queries，增加了相机位姿信息
                key, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                value, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None, # torch.Size([2500, 1, 256]) bev_mask(全零初始化)经过LearnedPositionalEncoding之后，torch.Size([1, 256, 50, 50])
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas']) # pc_range:[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # # 相机坐标系下的参考点reference_points_cam:torch.Size([6, 1, 2500, 4, 2]) bev_mask:torch.Size([6, 1, 2500, 4])
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :] # shift为tensor([[0., 0.]])

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2) # torch.Size([2500, 1, 256])->torch.Size([1, 2500, 256])
        bev_pos = bev_pos.permute(1, 0, 2) #torch.Size([2500, 1, 256])->torch.Size([1, 2500, 256]) # (全零初始化)经过LearnedPositionalEncoding
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2) # 没有前一帧bev的情况下，合并两个当前的2d参考点torch.Size([1, 2500, 1, 2])->torch.Size([2, 2500, 1, 2])

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query, # torch.Size([2500, 1, 256]) 没有前一帧的情况下就是由nn.Embedding(2500, 256)初始化的bev_queries，增加了相机位姿信息
                key, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                value, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                *args,
                bev_pos=bev_pos, # (全零初始化)经过LearnedPositionalEncoding
                ref_2d=hybird_ref_2d,# 区分是否存在前一帧bev
                ref_3d=ref_3d, # 3d参考点
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam, # 相机坐标系下的参考点
                bev_mask=bev_mask, # 相机坐标系下生成的bev_mask，主要用于判断点是否位于相机前方以及相机成像平面内
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query, # torch.Size([1, 2500, 256]) 没有前一帧的情况下就是由nn.Embedding(2500, 256)初始化的bev_queries，增加了相机位姿信息
                key=None, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                value=None, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                bev_pos=None, # (全零初始化)经过LearnedPositionalEncoding torch.Size([1, 2500, 256])
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None, # 区分是否存在前一帧bev torch.Size([2, 2500, 1, 2])
                ref_3d=None, # 3d参考点 torch.Size([1, 4, 2500, 3])
                bev_h=None,
                bev_w=None,
                reference_points_cam=None, # 相机坐标系下的参考点 torch.Size([6, 1, 2500, 4, 2])
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs): # 相机坐标系下生成的bev_mask，主要用于判断点是否位于相机前方以及相机成像平面内 torch.Size([6, 1, 2500, 4])
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)] # [None, None]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query, # torch.Size([1, 2500, 256])没有前一帧的情况下就是由nn.Embedding(2500, 256)初始化的bev_queries，增加了相机位姿信息/
                    prev_bev, # 没有前一帧的情况下None/堆叠了前一帧和当前帧（）的bev特征，torch.stack([prev_bev, bev_query], 1):torch.Size([2, 2500, 256])
                    prev_bev, # 没有前一帧的情况下None/堆叠了前一帧和当前帧（）的bev特征，torch.stack([prev_bev, bev_query], 1):torch.Size([2, 2500, 256])
                    identity if self.pre_norm else None,
                    query_pos=bev_pos, # torch.Size([1, 2500, 256])
                    key_pos=bev_pos, # torch.Size([1, 2500, 256])
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d, # torch.Size([2, 2500, 1, 2])
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device), #(50,50)
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs) # query:torch.Size([1, 2500, 256])
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, # 经过TemporalSelfAttention之后的bev_query torch.Size([1, 2500, 256])
                    key, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                    value, # torch.Size([6, 375, 1, 256]) 六个相机的图像特征torch.Size([1, 6, 256, 15, 25])增加了相机embed和特征层embed
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d, # 3d参考点 torch.Size([1, 4, 2500, 3])
                    reference_points_cam=reference_points_cam, # 相机坐标系下的参考点torch.Size([6, 1, 2500, 4, 2])
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)  # 相机坐标系下生成的bev_mask，主要用于判断点是否位于相机前方以及相机成像平面内 torch.Size([6, 1, 2500, 4])
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
