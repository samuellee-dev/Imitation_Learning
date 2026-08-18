# ACT: 1부터 100까지
for N in 1 2 4 8 16 32 64 100
do
  echo "===== ACT n_action_steps = $N ====="
  lerobot-eval \
    --policy.path=outputs/act_100k/checkpoints/last/pretrained_model \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=$N \
    --output_dir=outputs/eval_act_chunk$N
done

# Diffusion: 1부터 32까지
for N in 1 2 4 8 16 32
do
  echo "===== DIFFUSION n_action_steps = $N ====="
  lerobot-eval \
    --policy.path=outputs/diffusion_100k/checkpoints/last/pretrained_model \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=$N \
    --output_dir=outputs/eval_diff_chunk$N
done