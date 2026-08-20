for S in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000
do
  echo "===== ACT (200k series) step $S ====="
  lerobot-eval \
    --policy.path=outputs/act_200k/checkpoints/$S/pretrained_model \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=8 \
    --output_dir=outputs/eval_act200k_curve_$S
done