import json
import requests
import sys
import os

sys.path.append(os.path.dirname(__file__))
from knowledge import BENCHMARK_KNOWLEDGE, AVAILABLE_POLICIES
from tools import run_policy

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def ask_llama(task_request):
    prompt = f"""너는 로봇 policy를 선택하는 에이전트다. 아래 지식과 사용 가능한 policy 목록을 참고해서,
사용자 요청에 가장 적합한 policy를 하나만 골라라.

[지식]
{BENCHMARK_KNOWLEDGE}

[사용 가능한 policy 목록]
{list(AVAILABLE_POLICIES.keys())}

[사용자 요청]
{task_request}

반드시 아래 JSON 형식으로만 답해라. 다른 말은 하지 마라.
{{"policy": "선택한 policy 이름", "reasoning": "선택한 이유 한 문장"}}
"""

    response = requests.post(OLLAMA_URL, json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })
    return json.loads(response.json()["response"])

def main():
    task_request = input("어떤 작업을 원하시나요? (예: aloha 삽입 작업 해줘)\n> ")

    print("\n[Llama 3가 판단 중...]")
    decision = ask_llama(task_request)
    print(f"선택: {decision['policy']}")
    print(f"이유: {decision['reasoning']}\n")

    policy_key = decision["policy"]
    if policy_key not in AVAILABLE_POLICIES:
        print(f"경고: '{policy_key}'는 알 수 없는 policy입니다. 종료합니다.")
        return

    config = AVAILABLE_POLICIES[policy_key]
    result = run_policy(policy_key, config, n_episodes=20)

    print("\n[결과]")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()