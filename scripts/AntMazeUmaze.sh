SEED=$1
GROUP=$2
GPU=0


python DHRL/main.py \
--env_name 'AntMaze_UMaze-eval-v5' \
--test_env_name 'AntMaze_UMaze-eval-v5' \
--high_future_step 15 \
--subgoal_freq 40 \
--subgoal_scale 2.5 2.5 \
--subgoal_offset 0.0 0.0 \
--low_future_step 150 \
--subgoaltest_threshold 0.45 \
--cutoff 30 \
--n_initial_rollouts 142 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--subgoal_noise_eps 1 \
--cuda_num ${GPU} \
--seed ${SEED} \
--n_epochs 96 \
--n_cycles 15 \
--group $GROUP 

# epoch * max_ep_steps * cycles = total train timesteps
#96*700*15=1_008_000 steps