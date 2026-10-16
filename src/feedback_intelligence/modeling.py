from __future__ import annotations
import json
from functools import lru_cache
from typing import Any
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,mean_absolute_error,mean_squared_error,precision_score,r2_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from feedback_intelligence.config import Settings,get_settings

CLASS_FEATURES=['text','category','channel','product','language','status','affected_users','attachment_count','text_length']
REG_FEATURES=CLASS_FEATURES+['priority']
CATS=['category','channel','product','language','status'];NUM=['affected_users','attachment_count','text_length']

def preprocessor(include_priority=False):
    cats=CATS+(['priority'] if include_priority else [])
    return ColumnTransformer([('text',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=2200),'text'),('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),NUM)])

def train_models(frame:pd.DataFrame,settings:Settings|None=None,seed:int=42)->dict[str,Any]:
    s=settings or get_settings();s.ensure()
    train,test=train_test_split(frame,test_size=.25,random_state=seed,stratify=frame['priority'])
    classifier=Pipeline([('features',preprocessor()),('classifier',LogisticRegression(max_iter=1600,class_weight='balanced',random_state=seed))])
    classifier.fit(train[CLASS_FEATURES],train['priority']);pred=classifier.predict(test[CLASS_FEATURES]);labels=['low','medium','high']
    class_metrics={'model':'tfidf-logistic-regression','seed':seed,'train_rows':len(train),'test_rows':len(test),'features':CLASS_FEATURES,'accuracy':float(accuracy_score(test['priority'],pred)),'macro_precision':float(precision_score(test['priority'],pred,average='macro',zero_division=0)),'macro_recall':float(recall_score(test['priority'],pred,average='macro',zero_division=0)),'macro_f1':float(f1_score(test['priority'],pred,average='macro')),'labels':labels,'confusion_matrix':confusion_matrix(test['priority'],pred,labels=labels).tolist(),'classification_report':classification_report(test['priority'],pred,labels=labels,output_dict=True,zero_division=0)}
    joblib.dump(classifier,s.models_dir/'priority_classifier.joblib')
    resolved=frame.dropna(subset=['resolution_hours_model']).copy();rtrain,rtest=train_test_split(resolved,test_size=.25,random_state=seed)
    regressor=Pipeline([('features',preprocessor(True)),('regressor',RandomForestRegressor(n_estimators=140,max_depth=14,min_samples_leaf=2,random_state=seed,n_jobs=-1))])
    regressor.fit(rtrain[REG_FEATURES],rtrain['resolution_hours_model']);rpred=regressor.predict(rtest[REG_FEATURES])
    reg_metrics={'model':'tfidf-random-forest-regressor','seed':seed,'train_rows':len(rtrain),'test_rows':len(rtest),'features':REG_FEATURES,'mae_hours':float(mean_absolute_error(rtest['resolution_hours_model'],rpred)),'rmse_hours':float(mean_squared_error(rtest['resolution_hours_model'],rpred)**.5),'r2':float(r2_score(rtest['resolution_hours_model'],rpred))}
    joblib.dump(regressor,s.models_dir/'resolution_regressor.joblib')
    vect=TfidfVectorizer(stop_words='english',min_df=2,max_features=800);matrix=vect.fit_transform(frame['text']);clusters=KMeans(n_clusters=5,random_state=seed,n_init=20).fit_predict(matrix)
    numeric=frame[['satisfaction','affected_users','attachment_count','text_length']].fillna(frame[['satisfaction','affected_users','attachment_count','text_length']].median());detector=IsolationForest(contamination=.05,random_state=seed);anomaly=detector.fit_predict(numeric)
    terms=np.asarray(vect.get_feature_names_out());model=KMeans(n_clusters=5,random_state=seed,n_init=20).fit(matrix);top={str(i):terms[c.argsort()[-8:][::-1]].tolist() for i,c in enumerate(model.cluster_centers_)}
    unsup={'clusters':pd.Series(clusters).value_counts().sort_index().astype(int).to_dict(),'top_terms':top,'anomalies':int((anomaly==-1).sum())}
    metrics={'packages':{'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'priority_classifier':class_metrics,'resolution_regressor':reg_metrics,'unsupervised':unsup}
    (s.reports_dir/'model-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame({'record_id':frame['record_id'],'cluster':clusters,'is_anomaly':anomaly==-1}).to_csv(s.reports_dir/'unsupervised-results.csv',index=False)
    return metrics

def _feature_frame(payload:dict[str,Any])->pd.DataFrame:
    text=f"{payload.get('title','')}. {payload.get('message','')}".strip()
    return pd.DataFrame([{'text':text,'category':payload.get('category','other'),'channel':payload.get('channel','web'),'product':payload.get('product','workspace'),'language':payload.get('language','en'),'status':payload.get('status','open'),'affected_users':int(payload.get('affected_users',1)),'attachment_count':int(payload.get('attachment_count',0)),'text_length':len(text),'priority':payload.get('priority','medium')}])
@lru_cache(maxsize=4)
def _load(path:str): return joblib.load(path)
def predict_priority(payload:dict[str,Any],settings:Settings|None=None)->dict[str,Any]:
    s=settings or get_settings();model=_load(str(s.models_dir/'priority_classifier.joblib'));frame=_feature_frame(payload);label=str(model.predict(frame[CLASS_FEATURES])[0]);probs=model.predict_proba(frame[CLASS_FEATURES])[0];return {'priority':label,'probabilities':{str(k):float(v) for k,v in zip(model.classes_,probs,strict=True)}}
def predict_resolution(payload:dict[str,Any],settings:Settings|None=None)->dict[str,float]:
    s=settings or get_settings();model=_load(str(s.models_dir/'resolution_regressor.joblib'));frame=_feature_frame(payload);return {'estimated_resolution_hours':round(max(0,float(model.predict(frame[REG_FEATURES])[0])),2)}
