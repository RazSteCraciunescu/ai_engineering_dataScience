from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    root: Path
    raw_dir: Path
    processed_path: Path
    database_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    images_dir: Path
    knowledge_dir: Path
    semantic_index_path: Path
    random_seed: int = 42

    def ensure(self) -> None:
        for path in (self.raw_dir, self.processed_path.parent, self.database_path.parent, self.models_dir, self.reports_dir, self.figures_dir, self.images_dir, self.knowledge_dir, self.semantic_index_path.parent):
            path.mkdir(parents=True, exist_ok=True)

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_settings(root: Path | None = None) -> Settings:
    base=(root or project_root()).resolve()
    return Settings(
        root=base,
        raw_dir=base/'data'/'raw',
        processed_path=base/'data'/'processed'/'feedback.csv',
        database_path=base/'artifacts'/'database'/'feedback_intelligence.sqlite',
        models_dir=base/'artifacts'/'models',
        reports_dir=base/'artifacts'/'reports',
        figures_dir=base/'artifacts'/'figures',
        images_dir=base/'data'/'raw'/'images',
        knowledge_dir=base/'data'/'knowledge',
        semantic_index_path=base/'artifacts'/'models'/'semantic_index.joblib',
    )
