from __future__ import annotations
import hashlib
import json
from datetime import UTC,datetime
from pathlib import Path
from feedback_intelligence.collection import collect_sources
from feedback_intelligence.config import Settings,get_settings
from feedback_intelligence.cv_quality import analyze_directory
from feedback_intelligence.database import load_database
from feedback_intelligence.embeddings import build_index
from feedback_intelligence.modeling import train_models
from feedback_intelligence.preparation import prepare,validate
from feedback_intelligence.reporting import build_dashboard,build_reports

def _digest(path:Path)->str:
    hasher=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):hasher.update(chunk)
    return hasher.hexdigest()

def run(settings:Settings|None=None)->dict[str,object]:
    s=settings or get_settings();s.ensure()
    sources=sorted(path for path in s.raw_dir.iterdir() if path.suffix.lower() in {'.csv','.json','.html'})
    raw=collect_sources(sources);frame,cleaning=prepare(raw);quality=validate(frame)
    if not quality['ok']:raise ValueError(f'Validation failed: {quality}')
    frame.to_csv(s.processed_path,index=False)
    data_quality={'sources':[path.name for path in sources],'source_rows':raw.groupby('source_type').size().astype(int).to_dict(),'cleaning':cleaning,'validation':quality}
    (s.reports_dir/'data-quality.json').write_text(json.dumps(data_quality,indent=2)+'\n',encoding='utf-8')
    load_database(frame,s);eda=build_reports(frame,s);models=train_models(frame,s,s.random_seed);retrieval=build_index(s);images=analyze_directory(s);build_dashboard(frame,models,s)
    outputs=[s.processed_path,s.database_path,s.models_dir/'priority_classifier.joblib',s.models_dir/'resolution_regressor.joblib',s.semantic_index_path,s.reports_dir/'dashboard.html']
    manifest={'completed_at':datetime.now(UTC).replace(microsecond=0).isoformat(),'rows':len(frame),'data_quality':data_quality,'eda':eda,'model_metrics':models,'retrieval':retrieval,'image_count':len(images),'outputs':{str(path.relative_to(s.root)):{'bytes':path.stat().st_size,'sha256':_digest(path)} for path in outputs}}
    (s.root/'artifacts'/'pipeline-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest
