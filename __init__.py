# __init__.py
from .batch_processor import (
    BatchLoadImages,
    BatchAestheticScorer,
    ScoreWebDisplay
)

NODE_CLASS_MAPPINGS = {
    "BatchLoadImages": BatchLoadImages,
    "BatchAestheticScorer": BatchAestheticScorer,
    "ScoreWebDisplay": ScoreWebDisplay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImages": "批量加载图片",
    "BatchAestheticScorer": "批量美学评分",
    "ScoreWebDisplay": "生成评分网页"
}