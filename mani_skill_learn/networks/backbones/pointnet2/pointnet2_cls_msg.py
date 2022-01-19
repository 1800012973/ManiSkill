import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

import copy
#from mani_skill_learn.utils.data import dict_to_seq
#from mani_skill_learn.utils.torch import masked_average, masked_max
class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class PointBackbone(nn.Module):
    def __init__(self):
        super(PointBackbone, self).__init__()

    def forward(self, pcd):
        pcd = pcd.copy()
        if isinstance(pcd, dict):
            if 'pointcloud' in pcd:
                pcd['pcd'] = pcd['pointcloud']
                del pcd['pointcloud']
            assert 'pcd' in pcd
            return self.forward_raw(**pcd)
        else:
            return self.forward_raw(pcd)

    def forward_raw(self, pcd, state=None):
        raise NotImplementedError("")
#
# class PointNet2(PointBackbone):
#     def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False):
#         """
#         PointNet that processes multiple consecutive frames of pcd data.
#         :param conv_cfg: configuration for building point feature extractor
#         :param mlp_cfg: configuration for building global feature extractor
#         :param stack_frame: num of stacked frames in the input
#         :param subtract_mean_coords: subtract_mean_coords trick
#             subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
#             we found concatenating the mean pretty crucial
#         :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
#         """
#         super(PointNet2, self).__init__()
#         self.stack_frame = stack_frame
#         self.max_mean_mix_aggregation = max_mean_mix_aggregation
#         self.subtract_mean_coords = subtract_mean_coords
#
#         in_channel = (conv_cfg["mlp_spec"][0] + int(subtract_mean_coords) * 3)
#         self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel, [[16, 16, 32], [32, 32, 64]])
#         self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
#         self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
#         self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
#         self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
#         self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
#         self.conv1 = nn.Conv1d(128, 128, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         #self.conv2 = nn.Conv1d(128, num_classes, 1)
#
#     def forward_raw(self, pcd, state, mask=None):
#         """
#         :param pcd: point cloud
#                 xyz: shape (l, n_points, 3)
#                 rgb: shape (l, n_points, 3)
#                 seg: shape (l, n_points, n_seg) (unused in this function)
#         :param state: shape (l, state_shape) agent state and other information of robot
#         :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
#         :return: [B, F] ([batch size, final output dim])
#         """
#         if isinstance(pcd, dict):
#             pcd = pcd.copy()
#             mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
#             if self.subtract_mean_coords:
#                 # Use xyz - mean xyz instead of original xyz
#                 xyz = pcd['xyz']  # [B, N, 3]
#                 mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
#                 pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
#                 pcd['xyz'] = xyz - mean_xyz
#                 temp_pcd = copy.deepcopy(pcd['xyz'])
#             # Concat all elements like xyz, rgb, seg mask, mean_xyz
#             pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
#         else:
#             mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
#
#         B, N = pcd.shape[:2]
#         state = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]
#
#         l0_points = state.permute(0,2,1)
#         l0_xyz = xyz.permute(0,2,1)
#
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#
#         l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
#
#         x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)))) #B C N
#
#         return x

if __name__=="__main__":
    # x = PointNet2(dict(
    #                 type='ConvMLP',
    #                 norm_cfg=None,
    #                 mlp_spec=[38+3+3+3, 256, 256],
    #                 bias='auto',
    #                 inactivated_output=True,
    #                 conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
    #             ), None, 1, True)
    # pcd={}
    # pcd['xyz'] = torch.randn([2,100,3])
    # pcd['rgb'] = torch.randn([2,100,3])
    # pcd['seg'] = torch.randn([2,100,3])
    # state=torch.randn([2,38])
    #
    # x.forward_raw(pcd,state)
    model = get_model(10)
    from torchsummary import    summary
    summary(model, (3,1200), -1)