from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from feedback_intelligence.config import Settings,get_settings

def load_documents(directory:Path)->list[dict[str,str]]:
    docs=[]
    for path in sorted(directory.glob('*.md')):
        text=path.read_text(encoding='utf-8').strip()
        title=text.splitlines()[0].lstrip('# ').strip() if text else path.stem
        docs.append({'id':path.stem,'title':title,'source':path.name,'text':text})
    if len(docs)<3: raise ValueError('At least three knowledge documents are required')
    return docs

def build_index(settings:Settings|None=None)->dict[str,Any]:
    s=settings or get_settings();s.ensure();docs=load_documents(s.knowledge_dir)
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1)
    tfidf=vectorizer.fit_transform([d['text'] for d in docs])
    components=max(2,min(12,tfidf.shape[0]-1,tfidf.shape[1]-1))
    projector=Pipeline([('svd',TruncatedSVD(n_components=components,random_state=s.random_seed)),('normalizer',Normalizer(copy=False))])
    vectors=projector.fit_transform(tfidf)
    payload={'documents':docs,'vectorizer':vectorizer,'projector':projector,'vectors':vectors}
    joblib.dump(payload,s.semantic_index_path)
    metadata={'documents':len(docs),'dimensions':int(vectors.shape[1]),'algorithm':'tfidf-truncated-svd-normalized','distance_metrics':['cosine','euclidean'],'sources':[d['source'] for d in docs]}
    import json
    (s.reports_dir/'semantic-index.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata

def load_index(settings:Settings|None=None)->dict[str,Any]:
    s=settings or get_settings();return joblib.load(s.semantic_index_path)

def search(query:str,limit:int=3,index:dict[str,Any]|None=None,settings:Settings|None=None)->list[dict[str,Any]]:
    if not query.strip(): raise ValueError('Query must not be blank')
    payload=index or load_index(settings)
    query_vector=payload['projector'].transform(payload['vectorizer'].transform([query]))
    scores=cosine_similarity(query_vector,payload['vectors'])[0]
    distances=np.linalg.norm(payload['vectors']-query_vector,axis=1)
    order=np.argsort(-scores)[:max(1,min(limit,len(scores)))]
    results=[]
    for rank,index_value in enumerate(order,start=1):
        doc=payload['documents'][int(index_value)]
        results.append({'rank':rank,'id':doc['id'],'title':doc['title'],'source':doc['source'],'text':doc['text'],'cosine_similarity':float(scores[index_value]),'euclidean_distance':float(distances[index_value])})
    return results
