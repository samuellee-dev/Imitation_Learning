for S in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000
do
  echo "===== ACT (aloha) step $S ====="
  lerobot-eval \
    --policy.path=outputs/aloha_act_100k/checkpoints/$S/pretrained_model \
    --env.type=aloha --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=100 \
    --output_dir=outputs/eval_aloha_act_curve_$S
done

for S in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000
do
  echo "===== Diffusion (aloha) step $S ====="
  lerobot-eval \
    --policy.path=outputs/aloha_diffusion_100k/checkpoints/$S/pretrained_model \
    --env.type=aloha --eval.n_episodes=50 --eval.batch_size=1 \
    --eval.use_async_envs=false --policy.device=cuda \
    --policy.n_action_steps=16 \
    --output_dir=outputs/eval_aloha_diff_curve_$S
done