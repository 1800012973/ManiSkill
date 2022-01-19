#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'bucket_models.yml')))")


python -m tools.run_rl configs/bc/EEPose.py --gpu-ids=5 \
	--work-dir=./work_dirs/EE_Pose_bucket/ \
	--cfg-options "train_mfrl_cfg.total_steps=300000" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/MoveBucket/\",\"$model_list\"]" \
	"env_cfg.env_name=MoveBucket-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=150000"
