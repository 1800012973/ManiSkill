"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F
import numpy as np
from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from ..builder import BRL
from mani_skill_learn.utils.ee import EndEffectorInterface
from mani_skill_learn.utils.osc import OperationalSpaceControlInterface
import copy
@BRL.register_module()
class BC(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super(BC, self).__init__()
        self.batch_size = batch_size
        self.use_osc = False
        policy_optim_cfg = policy_cfg.pop("optim_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.policy = build_model(policy_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.sheduler = torch.optim.lr_scheduler.StepLR(self.policy_optim, step_size=100000, gamma=0.2) #bs=128,
        #print(action_shape)
        if action_shape == 13:
            self.ee = EndEffectorInterface("OpenCabinet")
            self.env="OpenCabinet"
            self.osc = OperationalSpaceControlInterface("OpenCabinet")
        else:
            self.ee = EndEffectorInterface("MoveBucket PushChair")
            self.env="MoveBucket PushChair"
            self.osc=OperationalSpaceControlInterface("MoveBucket PushChair")


    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"], next_obs=sampled_batch['next_obs'])
        #print("bc",sampled_batch['next_obs']['state'].shape)
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        #pred_action = self.policy(sampled_batch['obs'], mode='eval')

        # 新加对next_ee_pose 和 next_state预测
        pred_action, pred_next_ee_pose = self.policy(sampled_batch['obs'], mode='eval')
        sampled_batch['next_obs']["agent"]=sampled_batch['next_obs']['state']
        length = sampled_batch['next_obs']["agent"].shape[0]

        temp=[]
        for i in range(length):
            if self.env=="OpenCabinet":
                x = self.ee.get_ee_pose_from_obs(dict(agent=np.array(sampled_batch['next_obs']['state'][i].cpu())))
                temp.append(np.concatenate((x.p,x.q), axis=0))#7
            elif self.env=="MoveBucket PushChair":
                r,l = self.ee.get_ee_pose_from_obs(dict(agent=np.array(sampled_batch['next_obs']['state'][i].cpu())))
                temp.append(np.concatenate((r.p, r.q, l.p, l.q), axis=0))#14
        temp = torch.Tensor(np.array(temp)).cuda()

        if self.use_osc:
            osc_actions=[]
            for i in range(length):
                qpos = self.osc.get_robot_qpos_from_obs(sampled_batch['obs'][i].cpu().numpy())
                os_action, null_action = self.osc.joint_space_to_operational_space_and_null_space(qpos, sampled_batch["actions"][i])
                osc_actions.append(np.concatenate((os_action, null_action)))
            sampled_batch["actions"] = torch.Tensor(np.array(osc_actions)).cuda()

            policy_loss = F.mse_loss(pred_action[13:], sampled_batch['actions']) + F.l1_loss(pred_next_ee_pose, temp)\
                      #+F.l1_loss(pred_next_state, sampled_batch['next_obs']['state'])
        else:
            policy_loss = F.mse_loss(pred_action, sampled_batch['actions']) + F.l1_loss(pred_next_ee_pose, temp) \
                # +F.l1_loss(pred_next_state, sampled_batch['next_obs']['state'])
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return {
            'policy_abs_error': torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item(),
            'policy_loss': policy_loss.item()
        }
