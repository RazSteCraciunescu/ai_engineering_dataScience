from __future__ import annotations
import json
from typing import Any
import pandas as pd
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import FileResponse,RedirectResponse
from feedback_intelligence import __version__
from feedback_intelligence.config import get_settings
from feedback_intelligence.embeddings import search
from feedback_intelligence.modeling import predict_priority,predict_resolution
from feedback_intelligence.rag import answer
from feedback_intelligence.schemas import FeedbackInput,SearchInput

S=get_settings()
app=FastAPI(title='Public Feedback Intelligence API',version=__version__,description='Local inference, analytics and retrieval endpoints.')

@app.get('/',include_in_schema=False)
def root(): return RedirectResponse('/dashboard')
@app.get('/dashboard',include_in_schema=False)
def dashboard(): return FileResponse(S.reports_dir/'dashboard.html',media_type='text/html')
@app.get('/health')
def health()->dict[str,Any]:
    required=[S.processed_path,S.models_dir/'priority_classifier.joblib',S.models_dir/'resolution_regressor.joblib',S.semantic_index_path]
    missing=[str(path.relative_to(S.root)) for path in required if not path.exists()]
    return {'status':'ok' if not missing else 'degraded','version':__version__,'missing':missing}
@app.get('/metadata')
def metadata():
    paths={'data_quality':S.reports_dir/'data-quality.json','model_metrics':S.reports_dir/'model-metrics.json','retrieval':S.reports_dir/'semantic-index.json'}
    return {'version':__version__,**{name:json.loads(path.read_text()) if path.exists() else None for name,path in paths.items()}}
@app.get('/feedback')
def feedback(limit:int=Query(20,ge=1,le=200),category:str|None=None,priority:str|None=None):
    frame=pd.read_csv(S.processed_path)
    if category:frame=frame[frame['category']==category.lower()]
    if priority:frame=frame[frame['priority']==priority.lower()]
    columns=['record_id','created_at','channel','product','category','priority','title']
    return {'count':min(limit,len(frame)),'total_matching':len(frame),'items':frame[columns].head(limit).to_dict('records')}
@app.post('/predict/priority')
def priority(payload:FeedbackInput):
    try:return predict_priority(payload.model_dump())
    except FileNotFoundError as exc:raise HTTPException(503,'Classifier is unavailable') from exc
@app.post('/predict/resolution')
def resolution(payload:FeedbackInput):
    try:return predict_resolution(payload.model_dump())
    except FileNotFoundError as exc:raise HTTPException(503,'Regressor is unavailable') from exc
@app.post('/search')
def semantic(payload:SearchInput):
    items=search(payload.query,payload.limit)
    return {'query':payload.query,'results':[{key:value for key,value in item.items() if key!='text'}|{'excerpt':item['text'][:280]} for item in items]}
@app.post('/rag')
def rag(payload:SearchInput):return answer(payload.query,payload.limit)
@app.get('/vision')
def vision():
    path=S.reports_dir/'image-quality.json'
    if not path.exists():raise HTTPException(503,'Image report unavailable')
    return json.loads(path.read_text())
