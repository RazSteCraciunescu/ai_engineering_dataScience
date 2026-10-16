from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
from feedback_intelligence.config import Settings,get_settings

def analyze_image(path:Path)->dict[str,Any]:
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'Could not read {path}')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    brightness=float(np.mean(gray));contrast=float(np.std(gray));sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    smooth=cv2.GaussianBlur(gray,(3,3),0);noise=float(np.mean(np.abs(gray.astype(float)-smooth.astype(float))))
    return {'file':path.name,'width':int(image.shape[1]),'height':int(image.shape[0]),'channels':int(image.shape[2]),'brightness':round(brightness,3),'contrast':round(contrast,3),'laplacian_variance':round(sharpness,3),'noise_score':round(noise,3),'underexposed':brightness<100,'overexposed':brightness>235,'low_contrast':contrast<28,'blurred':sharpness<85}

def analyze_directory(settings:Settings|None=None)->list[dict[str,Any]]:
    s=settings or get_settings();results=[analyze_image(path) for path in sorted(s.images_dir.glob('*.png'))]
    (s.reports_dir/'image-quality.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(results).to_csv(s.reports_dir/'image-quality.csv',index=False)
    return results
