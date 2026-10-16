from __future__ import annotations
import re
from typing import Any
from feedback_intelligence.embeddings import search

def _sentence(text:str)->str:
    cleaned=re.sub(r'^#+\s+.*?$','',text,flags=re.MULTILINE).strip()
    parts=re.split(r'(?<=[.!?])\s+',cleaned)
    return parts[0].strip() if parts else cleaned[:260]

def answer(query:str,limit:int=3)->dict[str,Any]:
    evidence=search(query,limit)
    useful=[item for item in evidence if item['cosine_similarity']>0]
    if not useful:
        return {'question':query,'answer':'The local knowledge base does not contain enough matching evidence.','evidence':evidence,'mode':'deterministic-local-rag'}
    text=' '.join(_sentence(item['text']) for item in useful[:2])
    return {'question':query,'answer':text,'evidence':[{'rank':item['rank'],'title':item['title'],'source':item['source'],'cosine_similarity':round(item['cosine_similarity'],4)} for item in evidence],'mode':'deterministic-local-rag'}
