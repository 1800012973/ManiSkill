_base_ = ['./bc.py']

agent = dict(
    type='BC_gaussion',
    batch_size=1024,
    policy_cfg=dict(
        type='ContinuousPolicyRL',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=['obs_shape', 256, 256, 256, 'action_shape*2'],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        optim_cfg=dict(type='Adam', lr=1e-3),
    ),
)

env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='state',
    reward_type='dense',
)

