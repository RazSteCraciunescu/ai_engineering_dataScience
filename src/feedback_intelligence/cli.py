from __future__ import annotations
import argparse
import json
import subprocess
import sys
from feedback_intelligence.config import get_settings
from feedback_intelligence.embeddings import search
from feedback_intelligence.pipeline import run
from feedback_intelligence.rag import answer

def main():
    parser=argparse.ArgumentParser(prog='feedback-lab');sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('pipeline');sub.add_parser('validate')
    search_parser=sub.add_parser('search');search_parser.add_argument('query')
    ask_parser=sub.add_parser('ask');ask_parser.add_argument('query')
    serve_parser=sub.add_parser('serve');serve_parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    if args.command=='pipeline':result=run()
    elif args.command=='validate':result={'returncode':subprocess.call([sys.executable,str(get_settings().root/'scripts'/'validate_repository.py')])}
    elif args.command=='search':result=search(args.query)
    elif args.command=='ask':result=answer(args.query)
    else:
        import uvicorn
        uvicorn.run('feedback_intelligence.api:app',host='127.0.0.1',port=args.port,reload=False)
        return
    print(json.dumps(result,indent=2,default=str))
if __name__=='__main__':main()
