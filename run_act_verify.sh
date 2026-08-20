for S in 070000 080000 100000
do
  echo "===== ACT step $S (200 episodes) ====="
  lerobot-eval \
    --policy.path=outputs/act_100k/checkpoints/$S/pretrained_model \
    --env.type=pusht --eval.n_episodes=200 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=8 \
    --output_dir=outputs/eval_act_verify_$S
done