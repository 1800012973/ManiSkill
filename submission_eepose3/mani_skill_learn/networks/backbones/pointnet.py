import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone

from mani_skill_learn.utils.ee import EndEffectorInterface
from mani_skill_learn.utils.osc import OperationalSpaceControlInterface
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


@BACKBONES.register_module()
class PointNetV0(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """
        super(PointNetV0, self).__init__()
        conv_cfg = conv_cfg.deepcopy()
        conv_cfg.mlp_spec[0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)

    def forward_raw(self, pcd, state, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function)
        :param state: shape (l, state_shape) agent state and other information of robot
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """
        if isinstance(pcd, dict):
            pcd = pcd.copy()
            mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
            if self.subtract_mean_coords:
                # Use xyz - mean xyz instead of original xyz
                xyz = pcd['xyz']  # [B, N, 3]
                mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
                pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
                pcd['xyz'] = xyz - mean_xyz
            # Concat all elements like xyz, rgb, seg mask, mean_xyz

            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]

        B, N = pcd.shape[:2]
        state = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]
        point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)
        return self.global_mlp(global_feature)


@BACKBONES.register_module()
class PointNetWithInstanceInfoV0(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithInstanceInfoV0, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
        assert self.num_objs > 0

    def forward_raw(self, pcd, state):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg:  shape (l, n_points, n_seg)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = []
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)
        x = self.global_mlp(global_feature)
        # print(x)
        return x


@BACKBONES.register_module()
class PointNetWithEEPose2(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithEEPose2, self).__init__()
        self.use_osc = False

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        #self.pcd_pns = nn.ModuleList([PointNet2(pcd_pn_cfg.conv_cfg,None,1,subtract_mean_coords=True) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs

        assert self.num_objs > 0
        if final_mlp_cfg['mlp_spec'][-1]==13:
            self.ee = EndEffectorInterface("OpenCabinet")
            self.osc = OperationalSpaceControlInterface("OpenCabinet")
            self.env="OpenCabinet"
            self.pose_mlp = nn.Sequential(
                nn.Linear(256+7, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128,64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(128, 7)
            )
            final_mlp_cfg["mlp_spec"][0]+=7
            if self.use_osc:
                final_mlp_cfg["mlp_spec"][-1] = 12+7
        elif final_mlp_cfg['mlp_spec'][-1]==22:
            self.ee = EndEffectorInterface("PushChair MoveBucket")
            self.env="PushChair MoveBucket"
            self.osc = OperationalSpaceControlInterface("PushChair MoveBucket")
            self.pose_mlp = nn.Sequential(
                nn.Linear(256+2*7, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128,64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(128, 2*7)
            )
            final_mlp_cfg["mlp_spec"][0] += 2*7
            if self.use_osc:
                final_mlp_cfg["mlp_spec"][-1] = 20+2*7
        self.global_mlp = build_backbone(final_mlp_cfg)

    def forward_raw(self, pcd, state):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg:  shape (l, n_points, n_seg)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = []
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)

        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]

        ee_pose = []
        if self.env == "OpenCabinet":
            for i in range(state.shape[0]):
                temp = self.ee.get_ee_pose_from_obs(dict(agent=state[i].cpu().numpy()))
                # print(temp)
                ee_pose.append(np.concatenate((temp.p, temp.q), axis=0))
            ee_pose = torch.Tensor(np.array(ee_pose)).cuda()  # B 7
            pose_feature = torch.cat((ee_pose, global_feature), dim=1)
        elif self.env == "PushChair MoveBucket":
            for i in range(state.shape[0]):
                tempr, templ = self.ee.get_ee_pose_from_obs(dict(agent=state[i].cpu().numpy()))
                tempr = np.concatenate((tempr.p, tempr.q), axis=0)  # 7
                templ = np.concatenate((templ.p, templ.q), axis=0)  # 7
                ee_pose.append(np.concatenate((tempr, templ), axis=0))  # Right Left 14
            ee_pose = torch.Tensor(np.array(ee_pose)).cuda()  # B 14
            pose_feature = torch.cat((ee_pose, global_feature), dim=1)
        pose = self.pose_mlp(pose_feature)
        ee_pose += pose

        # print('Y', global_feature.shape)
        x = self.global_mlp(torch.cat((global_feature,ee_pose),dim=1))

        # print(x)
        if self.use_osc:
            qpos = []
            for i in range(state.shape[0]):
                qpos.append(self.osc.get_robot_qpos_from_obs(dict(agent=state[i].cpu().numpy())))
            qpos = torch.Tensor(np.array(qpos)).cuda()
                # if self.env == "OpenCabinet":
                #     action.append(self.osc.operational_space_and_null_space_to_joint_space(
                #         #qpos, x[i][:12].cpu().detach().numpy(), x[i][12:].cpu().detach().numpy(), True
                #     ))# input 13,12,7 or 22,20,14 output 13or22
                # elif self.env == "PushChair MoveBucket":
                #     action.append(self.osc.operational_space_and_null_space_to_joint_space(
                #         qpos, xx[i][:20].cpu().detach().numpy(), xx[i][20:].cpu().detach().numpy(), True
                #     ))  # input 13,12,7 or 22,20,14 output 13or22
            x = torch.cat([qpos, x],dim=1)

        return x,ee_pose


@BACKBONES.register_module()
class PointNetWithEEPose(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithEEPose, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs

        assert self.num_objs > 0
        if final_mlp_cfg['mlp_spec'][-1]==13:
            self.ee = EndEffectorInterface("OpenCabinet")
            self.env="OpenCabinet"
            self.pose_mlp = nn.Sequential(
                nn.Linear(3 * 256 + 7, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128,64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(128, 7)
            )
        elif final_mlp_cfg['mlp_spec'][-1]==22:
            self.ee = EndEffectorInterface("PushChair MoveBucket")
            self.env="PushChair MoveBucket"
            self.pose_mlp = nn.Sequential(
                nn.Linear(2 * 256 + 2*7, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128,64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(128, 2*7)
            )
    def forward_raw(self, pcd, state):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg:  shape (l, n_points, n_seg)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = []
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)

        ee_pose=[]
        if self.env == "OpenCabinet":
            for i in range(state.shape[0]):
                temp = self.ee.get_ee_pose_from_obs(dict(agent=state[i].cpu().numpy()))
                #print(temp)
                ee_pose.append(np.concatenate((temp.p,temp.q), axis=0))
            ee_pose = torch.Tensor(np.array(ee_pose)).cuda() #B 7
            pose_feature = torch.cat((ee_pose, obj_features[1],obj_features[2],obj_features[3]),dim=1)
        elif self.env=="PushChair MoveBucket":
            for i in range(state.shape[0]):
                tempr, templ = self.ee.get_ee_pose_from_obs(dict(agent=state[i].cpu().numpy()))
                tempr = np.concatenate((tempr.p,tempr.q),axis=0) #7
                templ = np.concatenate((templ.p, templ.q), axis=0) #7
                ee_pose.append(np.concatenate((tempr,templ), axis=0)) #Right Left 14
            ee_pose = torch.Tensor(np.array(ee_pose)).cuda() #B 14
            pose_feature = torch.cat((ee_pose, obj_features[1], obj_features[2]), dim=1)
        pose = self.pose_mlp(pose_feature)
        ee_pose += pose

        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)
        x = self.global_mlp(torch.cat((global_feature,ee_pose),dim=1))
        # print(x)
        return x,ee_pose


@BACKBONES.register_module()
class PointNetWithInstanceInfoWOGlobal(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithInstanceInfoWOGlobal, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs+3)])#original +2
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
        assert self.num_objs > 0

    def forward_raw(self, pcd, state):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg:  shape (l, n_points, n_seg)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        #obj_masks = []
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = []
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 1, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 1]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 1, NO + 1]
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)
        x = self.global_mlp(global_feature)
        # print(x)
        return x