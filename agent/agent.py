import json
import requests
import sys
import os

sys.path.append(os.path.dirname(__file__))
from knowledge import BENCHMARK_KNOWLEDGE, AVAILABLE_POLICIES
from tools import run_policy

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SUCCESS_THRESHOLD = 30  # 이 성공률(%) 미만이면 재시도를 고려

def ask_llama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })
    return json.loads(response.json()["response"])

def decide_policy(task_request, history=""):
    prompt = f"""너는 로봇 policy를 선택하는 에이전트다. 아래 지식과 사용 가능한 policy 목록을 참고해서,
사용자 요청에 가장 적합한 policy를 하나만 골라라.

[지식]
{BENCHMARK_KNOWLEDGE}

[사용 가능한 policy 목록]
{list(AVAILABLE_POLICIES.keys())}

[사용자 요청]
{task_request}

{history}

반드시 아래 JSON 형식으로만 답해라. 다른 말은 하지 마라.
{{"policy": "선택한 policy 이름", "reasoning": "선택한 이유 한 문장"}}
"""
    return ask_llama(prompt)

def decide_retry(task_request, tried_policy, pc_success):
    prompt = f"""너는 로봇 policy 실행 결과를 검토하는 에이전트다.

[지식]
{BENCHMARK_KNOWLEDGE}

[사용 가능한 policy 목록]
{list(AVAILABLE_POLICIES.keys())}

[상황]
사용자 요청: "{task_request}"
방금 "{tried_policy}"를 실행했고 성공률은 {pc_success}%였다.
이 성공률은 기대보다 낮다.

이 상황에서 어떻게 해야 할지 판단해라. 선택지:
- 같은 policy를 다시 시도한다 (표본이 작아 우연일 수 있음)
- 다른 policy로 바꿔본다 (지식을 참고해서)
- 더 시도해도 소용없다고 판단하고 멈춘다

반드시 아래 JSON 형식으로만 답해라.
{{"action": "retry_same" 또는 "switch_policy" 또는 "stop", "policy": "다음에 쓸 policy 이름(action이 stop이면 생략 가능)", "reasoning": "판단 이유 한 문장"}}
"""
    return ask_llama(prompt)

def execute(policy_key, n_episodes=20):
    if policy_key not in AVAILABLE_POLICIES:
        return {"success": False, "error": f"알 수 없는 policy: {policy_key}"}
    config = AVAILABLE_POLICIES[policy_key]
    return run_policy(policy_key, config, n_episodes=n_episodes)

def main():
    task_request = input("어떤 작업을 원하시나요? (예: aloha 삽입 작업 해줘)\n> ")

    print("\n[1단계: Llama 3가 판단 중...]")
    decision = decide_policy(task_request)
    policy_key = decision["policy"]
    print(f"선택: {policy_key}")
    print(f"이유: {decision['reasoning']}\n")

    result = execute(policy_key)
    if not result.get("success"):
        print("[결과] 실행 자체가 실패했습니다.")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    pc_success = result["pc_success"]
    print(f"[1차 실행 결과] {policy_key} → 성공률 {pc_success}%\n")

    attempt = 1
    while pc_success is not None and pc_success < SUCCESS_THRESHOLD and attempt < 3:
        print(f"[성공률이 낮습니다 ({pc_success}% < {SUCCESS_THRESHOLD}%). Llama 3에게 다음 행동을 묻는 중...]")
        retry_decision = decide_retry(task_request, policy_key, pc_success)
        action = retry_decision.get("action")
        print(f"판단: {action}")
        print(f"이유: {retry_decision.get('reasoning')}\n")

        if action == "stop":
            print("[에이전트가 재시도를 멈추기로 결정했습니다.]")
            break

        next_policy = retry_decision.get("policy", policy_key)
        print(f"[{attempt+1}차 실행: {next_policy}]")
        result = execute(next_policy)
        if not result.get("success"):
            print("[결과] 실행 자체가 실패했습니다.")
            break

        policy_key = next_policy
        pc_success = result["pc_success"]
        print(f"[{attempt+1}차 실행 결과] {policy_key} → 성공률 {pc_success}%\n")
        attempt += 1

    print("=" * 40)
    print("[최종 결과]")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()