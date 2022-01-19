# README

This repository is a submission of ManiSkill Challenge.

Both no interaction and no external annotation tracks are contained. 

For **no interaction** track, the main idea is to: 

1. utilize the *End-Effector Pose* for supervision
2. combine *multi time-steps point clouds*  to exploit the multi-view information for visual inputs.

We process the dataset and use the given attributes to generate above labels for supervision.



For **no external annotation**  track, we simply use the models trained in no interaction track and do RL training in the environment.



###### Usage

The environment requirement is the same as the original ManiSkill-Learn repository.

For training, execute the `run_bc_EEPOSE.sh` or `run_SAC.sh` in `/scripts`

For testing, execute the test command shown [here](https://github.com/haosulab/ManiSkill-Learn#download-data---quick-example) in directory `submission_eepose3` or `submission_RL`.



