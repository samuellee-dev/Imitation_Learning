for N in 1 2 4 8 16 32 64 100
do
  echo "===== ACT (aloha) n_action_steps = $N ====="
  lerobot-eval \
    --policy.path=outputs/aloha_act_100k/checkpoints/last/pretrained_model \
    --env.type=aloha \
    --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=$N \
    --output_dir=outputs/eval_aloha_act_chunk$N
done

for N in 1 2 4 8 16 32
do
  echo "===== Diffusion (aloha) n_action_steps = $N ====="
  lerobot-eval \
    --policy.path=outputs/aloha_diffusion_100k/checkpoints/last/pretrained_model \
    --env.type=aloha \
    --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=$N \
    --output_dir=outputs/eval_aloha_diff_chunk$N
done