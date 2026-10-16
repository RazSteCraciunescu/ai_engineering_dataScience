from __future__ import annotations
import re
from typing import Any
import pandas as pd

REQUIRED={'source_id','created_at','channel','product','category','title','message','priority','satisfaction','resolution_hours','status','language','affected_users','attachment_count'}
CATEGORY={'billing':'billing','bill':'billing','account':'account','login':'account','performance':'performance','slow':'performance','usability':'usability','ux':'usability','feature':'feature request','feature request':'feature request','reliability':'reliability','crash':'reliability'}
CHANNEL={'web':'web','website':'web','mobile':'mobile','app':'mobile','community':'community','forum':'community'}
PRIORITY={'low':'low','p3':'low','minor':'low','medium':'medium','p2':'medium','normal':'medium','high':'high','p1':'high','urgent':'high'}

def _text(value:object)->str:
    return re.sub(r'\s+',' ','' if pd.isna(value) else str(value)).strip()
def _map(series:pd.Series,mapping:dict[str,str],fallback:str)->pd.Series:
    return series.fillna('').astype(str).str.strip().str.lower().map(mapping).fillna(fallback)

def prepare(raw:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    missing=REQUIRED-set(raw.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    frame=raw.copy();raw_rows=len(frame)
    frame['source_id']=frame['source_id'].map(_text)
    frame['created_at']=pd.to_datetime(frame['created_at'],utc=True,errors='coerce')
    for c in ['title','message','product','status','language']: frame[c]=frame[c].map(_text)
    frame['product']=frame['product'].str.lower();frame['status']=frame['status'].str.lower();frame['language']=frame['language'].str.lower().str[:2]
    frame['category']=_map(frame['category'],CATEGORY,'other');frame['channel']=_map(frame['channel'],CHANNEL,'other');frame['priority']=_map(frame['priority'],PRIORITY,'medium')
    for c in ['satisfaction','resolution_hours','affected_users','attachment_count']: frame[c]=pd.to_numeric(frame[c],errors='coerce')
    frame['satisfaction_raw']=frame['satisfaction']
    group_median=frame.groupby('category')['satisfaction'].transform('median');global_median=float(frame['satisfaction'].median())
    frame['satisfaction']=frame['satisfaction'].fillna(group_median).fillna(global_median)
    frame['satisfaction_was_missing']=frame['satisfaction_raw'].isna()
    frame['resolution_hours_raw']=frame['resolution_hours']
    values=frame['resolution_hours'].dropna();q1=float(values.quantile(.25));q3=float(values.quantile(.75));iqr=q3-q1;low=max(0,q1-1.5*iqr);high=q3+1.5*iqr
    frame['is_outlier']=frame['resolution_hours'].notna() & ((frame['resolution_hours']<low)|(frame['resolution_hours']>high))
    frame['resolution_hours_model']=frame['resolution_hours'].clip(low,high)
    frame['affected_users']=frame['affected_users'].fillna(1).clip(lower=1).astype(int)
    frame['attachment_count']=frame['attachment_count'].fillna(0).clip(lower=0).astype(int)
    frame['text']=(frame['title']+'. '+frame['message']).str.strip();frame['text_length']=frame['text'].str.len().astype(int);frame['word_count']=frame['text'].str.split().str.len().astype(int)
    frame=frame.sort_values(['created_at','source_type','source_id'],na_position='last')
    duplicates=frame.duplicated('source_id',keep='first');removed=int(duplicates.sum());frame=frame.loc[~duplicates].copy();frame['record_id']=frame['source_id']
    frame['created_at']=frame['created_at'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    columns=['record_id','source_id','created_at','collected_at','source_type','source_file','channel','product','category','title','message','text','language','priority','status','satisfaction_raw','satisfaction','satisfaction_was_missing','resolution_hours_raw','resolution_hours','resolution_hours_model','is_outlier','affected_users','attachment_count','text_length','word_count']
    frame=frame[columns].reset_index(drop=True)
    report={'raw_rows':raw_rows,'canonical_rows':len(frame),'duplicates_removed':removed,'missing_satisfaction_imputed':int(frame['satisfaction_was_missing'].sum()),'resolution_outliers_flagged':int(frame['is_outlier'].sum()),'resolution_lower_fence':round(low,3),'resolution_upper_fence':round(high,3)}
    return frame,report

def validate(frame:pd.DataFrame)->dict[str,Any]:
    required={'record_id','created_at','source_type','category','text','priority','satisfaction','affected_users'}
    checks={
      'required_columns':not(required-set(frame.columns)),
      'unique_record_id':frame['record_id'].is_unique,
      'priority_domain':set(frame['priority'].dropna())<= {'low','medium','high'},
      'satisfaction_range':bool(frame['satisfaction'].between(1,5).all()),
      'non_blank_text':bool(frame['text'].fillna('').str.strip().ne('').all()),
      'positive_users':bool((frame['affected_users']>=1).all()),
    }
    return {'ok':all(checks.values()),'rows':len(frame),'columns':len(frame.columns),'checks':checks,'missing_by_column':{k:int(v) for k,v in frame.isna().sum().items() if v}}
