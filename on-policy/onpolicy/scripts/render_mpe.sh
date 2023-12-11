#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}"
# for seed in `seq ${seed_max}`
# do
#     CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
#     --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
#     --n_training_threads 1 --n_rollout_threads 1 --use_render True --episode_length 25 --render_episodes 10 \
#     --model_dir "xxx" --use_valuenorm False --use_wandb False
# done

for seed in `seq ${seed_max}`; do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_value_active_masks False --use_policy_active_masks False \
    --use_sd True --save_dir "SD_MAPPO_MPE3" --sd_delta 0.125 --cuda False --use_render True --render_epsiodes 30 --use_valuenorm False --model_dir "xxx" --use_wandb false
done
