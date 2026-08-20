import subprocess
import json
import os
import time

def run_policy(policy_key, config, n_episodes=20):
    timestamp = int(time.time())
    output_dir = f"outputs/agent_run_{policy_key}_{timestamp}"

    cmd = [
        "lerobot-eval",
        f"--policy.path={config['path']}",
        f"--env.type={config['env_type']}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--policy.device=cuda",
        f"--policy.n_action_steps={config['n_action_steps']}",
        f"--output_dir={output_dir}",
    ]

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"

    print(f"\n[실행 중] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    info_path = os.path.join(output_dir, "eval_info.json")
    if not os.path.exists(info_path):
        return {"success": False, "error": result.stderr[-500:]}

    with open(info_path) as f:
        data = json.load(f)

    success_rate = data["overall"]["pc_success"]
    return {"success": True, "policy": policy_key, "pc_success": success_rate}