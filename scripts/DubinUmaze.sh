SEED=$1
GROUP=$2
GPU=0


python DHRL/main.py \
--env_name 'DubinsUMaze-v0' \
--test_env_name 'DubinsUMaze-v0' \
--high_future_step 15 \
--subgoal_freq 40 \
--subgoal_scale 1.0 1.0 \
--subgoal_offset 0.0 0.0 \
--low_future_step 150 \
--subgoaltest_threshold 0.1 \
--cutoff 30 \
--n_initial_rollouts 72 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--subgoal_noise_eps 1 \
--cuda_num ${GPU} \
--seed ${SEED} \
--n_epochs 143 \ 
--n_cycles 15 \
--group $GROUP 

# epoch * max_ep_steps * cycles = total train timesteps
 #143*70*15= 150_150 steps