import torch
from fastapi import APIRouter

from app import config
from app.core.job_store import job_store
from app.models.schemas import HealthResponse
from app.services.whisperx_service import whisperx_service

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="서버 상태 확인",
    description=(
        "서버 상태, GPU 정보, WhisperX 모델 로딩 여부를 반환합니다.\n\n"
        "### 활용 방법\n\n"
        "- **모니터링**: `status` 필드로 서버 생존 확인\n"
        "- **준비 상태**: `model_loaded`가 `true`가 될 때까지 STT 요청을 보내지 마세요\n"
        "- **GPU 확인**: `gpu` 필드로 사용 중인 GPU 모델 확인\n"
        "- **큐 포화도**: `queue.utilization_pct`로 큐 사용률 관측. "
        "`queue.active >= queue.max_active`이면 신규 POST는 503 반환\n\n"
        "### 응답 시간\n\n"
        "이 엔드포인트는 GPU 상태 조회를 포함하므로, 첫 호출 시 약 100ms 소요될 수 있습니다."
    ),
    responses={
        200: {
            "description": "서버 정상 동작 중",
            "content": {
                "application/json": {
                    "examples": {
                        "GPU 서버 (모델 로딩 완료)": {
                            "summary": "정상 상태 — GPU 사용, 모델 준비 완료",
                            "value": {
                                "status": "ok",
                                "service": "WhisperX STT Server",
                                "version": "2.0.0",
                                "model": "large-v3",
                                "device": "cuda",
                                "gpu": "NVIDIA GeForce RTX 4090",
                                "model_loaded": True,
                                "queue": {"active": 0, "max_active": 5, "utilization_pct": 0.0},
                            },
                        },
                        "서버 시작 중 (모델 로딩 전)": {
                            "summary": "서버 시작됨 — 모델 아직 로딩 중",
                            "value": {
                                "status": "ok",
                                "service": "WhisperX STT Server",
                                "version": "2.0.0",
                                "model": "large-v3",
                                "device": "cuda",
                                "gpu": "NVIDIA GeForce RTX 4090",
                                "model_loaded": False,
                                "queue": {"active": 0, "max_active": 5, "utilization_pct": 0.0},
                            },
                        },
                        "큐 사용률 60% (3/5 처리 중)": {
                            "summary": "정상 + 큐 일부 사용 중",
                            "value": {
                                "status": "ok",
                                "service": "WhisperX STT Server",
                                "version": "2.0.0",
                                "model": "large-v3",
                                "device": "cuda",
                                "gpu": "NVIDIA GeForce RTX 4090",
                                "model_loaded": True,
                                "queue": {"active": 3, "max_active": 5, "utilization_pct": 60.0},
                            },
                        },
                        "CPU 모드": {
                            "summary": "CPU 전용 모드 (GPU 없음)",
                            "value": {
                                "status": "ok",
                                "service": "WhisperX STT Server",
                                "version": "2.0.0",
                                "model": "large-v3",
                                "device": "cpu",
                                "gpu": None,
                                "model_loaded": True,
                                "queue": {"active": 0, "max_active": 5, "utilization_pct": 0.0},
                            },
                        },
                    },
                },
            },
        },
    },
)
async def health_check():
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    active = job_store.active_count()
    max_active = config.MAX_ACTIVE_JOBS
    utilization = round(active * 100 / max_active, 1) if max_active > 0 else 0.0

    return {
        "status": "ok",
        "service": config.SERVICE_NAME,
        "version": config.VERSION,
        "model": config.MODEL_SIZE,
        "device": config.DEVICE,
        "gpu": gpu_name,
        "model_loaded": whisperx_service.is_model_loaded(),
        "queue": {
            "active": active,
            "max_active": max_active,
            "utilization_pct": utilization,
        },
    }
