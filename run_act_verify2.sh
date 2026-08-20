echo "===== ACT step 100000 (200 episodes) ====="
lerobot-eval \
  --policy.path=outputs/act_100k/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=200 --eval.batch_size=1 \
  --eval.use_async_envs=false --policy.device=cuda \
  --policy.n_action_steps=8 \
  --output_dir=outputs/eval_act_verify2_100000

echo "===== ACT step 150000 (200 episodes) ====="
lerobot-eval \
  --policy.path=outputs/act_200k/checkpoints/050000/pretrained_model \
  --env.type=pusht --eval.n_episodes=200 --eval.batch_size=1 \
  --eval.use_async_envs=false --policy.device=cuda \
  --policy.n_action_steps=8 \
  --output_dir=outputs/eval_act_verify2_150000

echo "===== ACT step 200000 (200 episodes) ====="
lerobot-eval \
  --policy.path=outputs/act_200k/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=200 --eval.batch_size=1 \
  --eval.use_async_envs=false --policy.device=cuda \
  --policy.n_action_steps=8 \
  --output_dir=outputs/eval_act_verify2_200000