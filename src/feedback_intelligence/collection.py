from __future__ import annotations
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from collections.abc import Iterable
import pandas as pd

class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headers:list[str]=[]; self.rows:list[list[str]]=[]
        self._tag:str|None=None; self._buf:list[str]=[]; self._row:list[str]=[]
    def handle_starttag(self,tag,attrs):
        if tag=='tr': self._row=[]
        if tag in {'th','td'}: self._tag=tag; self._buf=[]
    def handle_data(self,data):
        if self._tag: self._buf.append(data)
    def handle_endtag(self,tag):
        if tag in {'th','td'} and self._tag==tag:
            value=' '.join(''.join(self._buf).split())
            if tag=='th': self.headers.append(value)
            else: self._row.append(value)
            self._tag=None; self._buf=[]
        elif tag=='tr' and self._row:
            self.rows.append(self._row); self._row=[]

def _provenance(frame:pd.DataFrame,path:Path,kind:str)->pd.DataFrame:
    out=frame.copy();out['source_type']=kind;out['source_file']=path.name;out['collected_at']=datetime.now(UTC).replace(microsecond=0).isoformat();return out

def collect_csv(path:Path)->pd.DataFrame: return _provenance(pd.read_csv(path),path,'csv')
def collect_json(path:Path)->pd.DataFrame:
    payload=json.loads(path.read_text(encoding='utf-8'))
    records=payload.get('export',{}).get('records') if isinstance(payload,dict) else payload
    if not isinstance(records,list): raise ValueError(f'Expected records list in {path}')
    return _provenance(pd.DataFrame(records),path,'json')
def collect_html(path:Path)->pd.DataFrame:
    parser=TableParser();parser.feed(path.read_text(encoding='utf-8'))
    if not parser.headers or not parser.rows: raise ValueError(f'No table data in {path}')
    rows=[r for r in parser.rows if len(r)==len(parser.headers)]
    return _provenance(pd.DataFrame(rows,columns=parser.headers),path,'html')
def collect_sources(paths:Iterable[Path])->pd.DataFrame:
    dispatch={'.csv':collect_csv,'.json':collect_json,'.html':collect_html}
    frames=[]
    for path in paths:
        if path.suffix.lower() not in dispatch: raise ValueError(f'Unsupported source: {path.suffix}')
        frames.append(dispatch[path.suffix.lower()](path))
    if not frames: raise ValueError('No source files supplied')
    return pd.concat(frames,ignore_index=True,sort=False)
