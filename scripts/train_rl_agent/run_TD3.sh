python -m tools.run_rl configs/td3/td3_transformer.py --seed=11 --gpu-ids=1 --clean-up \
      --work-dir=./work_dirs/td3/td3_transformer_drawer/ \
      --cfg-options "train_mfrl_cfg.total_steps=300000" "train_mfrl_cfg.init_replay_buffers=" \
	    "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=150000" \
	    "agent.batch_size=128"
