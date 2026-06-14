"""Judge Provider 查询 API。"""

from typing import List

from fastapi import APIRouter

from api.schemas import JudgeProviderItem
from core.evaluator import list_available_judge_providers

router = APIRouter(prefix="/judge", tags=["judge"])


@router.get("/providers", response_model=List[JudgeProviderItem])
def get_judge_providers() -> List[JudgeProviderItem]:
    return [JudgeProviderItem(**p) for p in list_available_judge_providers()]
