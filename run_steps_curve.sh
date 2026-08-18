for S in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000
do
  echo "===== DIFFUSION step $S ====="
  lerobot-eval \
    --policy.path=outputs/diffusion_100k/checkpoints/$S/pretrained_model \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=8 \
    --output_dir=outputs/eval_curve_diff_$S
done