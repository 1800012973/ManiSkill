_base_ = ['./td3.py']
stack_frame = 1
num_heads = 4

train_mfrl_cfg = dict(
    total_steps=1000000,
    warm_steps=0,
    n_eval=200000,
    n_checkpoint=200000,
    n_steps=8,
    n_updates=4,
    on_policy=False,
)

env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=stack_frame,
    obs_mode='pointcloud',
    reward_type='dense',
)

replay_cfg = dict(
    type='ReplayMemory',
    capacity=1000000,
)


rollout_cfg = dict(
    type='BatchRollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=8,
)

eval_cfg = dict(
    type='BatchEvaluation',
    num=10,
    num_procs=2,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
)


agent = dict(
    type='TD3',
    batch_size=1024,
    gamma=0.95,
    policy_cfg=dict(
        type='ContinuousPolicyRL',
        policy_head_cfg=dict(
            type='DeterministicHead',
            noise_std=1e-5,
        ),
        nn_cfg=dict(
            type='PointNetWithInstanceInfoWOGlobal',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[256 * 1, 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 256, 256],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=256,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[256, 1024, 256],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=256,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=6,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[256, 256, 'action_shape'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),

    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetWithInstanceInfoWOGlobal_Value',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[256 * 1, 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 256, 256],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=256,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[256, 1024, 256],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=256,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=6,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[256*2, 256, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=0.0003),
    ),
)

