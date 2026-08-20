# 1부 실험에서 확인된 사실들. 에이전트가 판단할 때 참고하는 "지식"입니다.

BENCHMARK_KNOWLEDGE = """
[pusht 태스크 - T자 블록 밀기]
- Diffusion Policy 성공률 68%, ACT 성공률 33% (200회 검증)
- 최적 n_action_steps는 8

[aloha_insertion 태스크 - 핀을 소켓에 삽입]
- ACT 성공률 24%, Diffusion Policy 성공률 8%
- 최적 n_action_steps는 100

[두 모델의 작동 방식 차이]
- Diffusion Policy는 화면을 한 번 볼 때마다 신경망을 100번 통과시켜 답을 정제한다.
  느리지만, 애매한 상황에서 여러 가능한 답 중 하나를 유연하게 골라낼 수 있다.
- ACT는 화면을 한 번 볼 때마다 신경망을 1번만 통과시켜 답을 낸다.
  빠르지만, 정답이 여러 개일 때 그 평균값을 내버리는 경향이 있어 애매한 상황에 약하다.
- ACT는 대신 한 번에 긴 행동 순서를 계획해서, 순서가 정해진 긴 동작을 일관되게
  수행하는 데는 유리하다.
"""

# 사용 가능한 policy 체크포인트 경로
AVAILABLE_POLICIES = {
    "pusht_diffusion": {
        "path": "outputs/diffusion_100k/checkpoints/090000/pretrained_model",
        "env_type": "pusht",
        "n_action_steps": 8,
    },
    "pusht_act": {
        "path": "outputs/act_100k/checkpoints/last/pretrained_model",
        "env_type": "pusht",
        "n_action_steps": 8,
    },
    "aloha_diffusion": {
        "path": "outputs/aloha_diffusion_100k/checkpoints/last/pretrained_model",
        "env_type": "aloha",
        "n_action_steps": 16,
    },
    "aloha_act": {
        "path": "outputs/aloha_act_100k/checkpoints/last/pretrained_model",
        "env_type": "aloha",
        "n_action_steps": 100,
    },
}