"""票房异常监管识别包。"""

from .config import DEFAULT_CONFIG, DetectorConfig
from .pipeline import PipelineResult, run_pipeline, write_reports
from .report import ReportBundle

__all__ = [
    "DEFAULT_CONFIG",
    "DetectorConfig",
    "PipelineResult",
    "ReportBundle",
    "run_pipeline",
    "write_reports",
]

__version__ = "0.1.0"
