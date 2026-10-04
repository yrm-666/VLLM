"""Query-aware temporal frame selection for Molmo2 video inference."""

from .config import SelectorConfig
from .encoders import DualEncoder, Siglip2Encoder, cosine_similarities
from .pipeline import PipelineResult, QueryAwareSelectionPipeline
from .query import build_query_text
from .sampling import official_style_candidate_indices, uniform_candidate_indices
from .selector import QueryAwareFrameSelector, SelectionResult, SelectionStep

__all__ = [
    "QueryAwareFrameSelector",
    "QueryAwareSelectionPipeline",
    "PipelineResult",
    "SelectionResult",
    "SelectionStep",
    "SelectorConfig",
    "DualEncoder",
    "Siglip2Encoder",
    "build_query_text",
    "cosine_similarities",
    "uniform_candidate_indices",
    "official_style_candidate_indices",
]
