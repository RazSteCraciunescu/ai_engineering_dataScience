from __future__ import annotations
from collections.abc import Iterable
import numpy as np

def minmax(values:Iterable[float])->np.ndarray:
    a=np.asarray(list(values),dtype=float);lo=float(np.nanmin(a));hi=float(np.nanmax(a));return np.zeros_like(a) if hi==lo else (a-lo)/(hi-lo)
def standardize(values:Iterable[float])->np.ndarray:
    a=np.asarray(list(values),dtype=float);std=float(np.nanstd(a));return np.zeros_like(a) if std==0 else (a-float(np.nanmean(a)))/std
def euclidean(left:Iterable[float],right:Iterable[float])->float:
    a=np.asarray(list(left),dtype=float);b=np.asarray(list(right),dtype=float)
    if a.shape!=b.shape: raise ValueError('Vector shapes must match')
    return float(np.linalg.norm(a-b))
def summarize(values:Iterable[float])->dict[str,float]:
    a=np.asarray([v for v in values if not np.isnan(v)],dtype=float)
    if a.size==0: raise ValueError('At least one finite value is required')
    mean=float(np.mean(a));sd=float(np.std(a,ddof=1)) if a.size>1 else 0.0;skew=float(np.mean((a-mean)**3)/(np.std(a)**3)) if np.std(a) else 0.0
    return {'count':float(a.size),'mean':mean,'median':float(np.median(a)),'variance':float(np.var(a,ddof=1)) if a.size>1 else 0.0,'standard_deviation':sd,'minimum':float(np.min(a)),'maximum':float(np.max(a)),'skewness':skew}
def gradient_descent(x:Iterable[float],y:Iterable[float],learning_rate:float=.01,iterations:int=1000)->dict[str,object]:
    xa=np.asarray(list(x),dtype=float);ya=np.asarray(list(y),dtype=float)
    if xa.shape!=ya.shape or xa.size==0: raise ValueError('x and y must match and be non-empty')
    w=b=0.0;history=[];n=float(len(xa))
    for step in range(iterations):
        pred=w*xa+b;err=pred-ya;w-=learning_rate*float(2*np.dot(err,xa)/n);b-=learning_rate*float(2*np.sum(err)/n)
        if step%max(1,iterations//20)==0 or step==iterations-1: history.append(float(np.mean(err**2)))
    return {'weight':w,'bias':b,'loss_history':history}
def binary_metrics(truth:Iterable[int],prediction:Iterable[int])->dict[str,float|int]:
    t=np.asarray(list(truth));p=np.asarray(list(prediction))
    if t.shape!=p.shape: raise ValueError('Shapes must match')
    tp=int(np.sum((t==1)&(p==1)));tn=int(np.sum((t==0)&(p==0)));fp=int(np.sum((t==0)&(p==1)));fn=int(np.sum((t==1)&(p==0)))
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':(tp+tn)/max(1,len(t)),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)}
