# 1부 실험에서 확인된 사실들. 에이전트가 판단할 때 참고하는 "지식"입니다.

BENCHMARK_KNOWLEDGE = """
[pusht 태스크 - T자 블록 밀기]
- Diffusion Policy가 ACT보다 확실히 우세함 (성공률 68% vs 33%, 200회 검증)
- 최적 n_action_steps는 8 (너무 크거나 작으면 성능 급락)
- 학습 스텝: 70,000~100,000 사이에서 성능이 평평함 (과적합 시작점)

[aloha_insertion 태스크 - 핀을 소켓에 삽입]
- ACT가 Diffusion Policy보다 확실히 우세함 (성공률 24% vs 8%)
- 최적 n_action_steps는 100 (pusht와 정반대 경향! 길게 실행할수록 좋음)
- 학습 스텝을 늘려도(10만→20만) 성능 개선 없음, 100,000 스텝이면 충분

[일반 원칙]
- 반응성이 중요한 작업(pusht류)에는 Diffusion Policy
- 긴 동작을 정확히 재현해야 하는 작업(삽입류)에는 ACT
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