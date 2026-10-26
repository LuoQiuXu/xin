"""Read-only primary PDF collection; local outputs are confined to this run."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

RUN=Path(__file__).resolve().parent
REPO=RUN.parents[3]
sys.path.append('C:/Users/xiaoju/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
from pypdf import PdfReader

def collect(item):
    key,url=item
    out={'key':key,'url':url,'status':'UNVERIFIED'}
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research evidence audit)'})
        with urllib.request.urlopen(request,timeout=40) as response:
            data=response.read(); out['final_url']=response.url
        if not data.startswith(b'%PDF'): raise ValueError('Response is not a PDF')
        path=RUN/'paper_cache'/f'{key}.pdf'
        with path.open('xb') as handle: handle.write(data)
        reader=PdfReader(path)
        pages=[{'page':i+1,'text':page.extract_text() or ''} for i,page in enumerate(reader.pages)]
        with (RUN/f'{key}_PAGES.json').open('x',encoding='utf-8') as handle:
            json.dump(pages,handle,ensure_ascii=False,indent=2)
        out.update(status='FULL_PDF_TEXT_OPENED',sha256=hashlib.sha256(data).hexdigest(),
                   bytes=len(data),pages=len(pages),local_pdf=str(path.relative_to(RUN)))
    except Exception as exc:
        out['error']=f'{type(exc).__name__}: {exc}'
    return out

if __name__=='__main__':
    registry=json.loads((REPO/'H2T_repro_audit/research_gap_20261026/PAPER_SOURCE_REGISTRY.json').read_text(encoding='utf-8'))
    items=[(p['key'],p['fulltext_url']) for p in registry['papers']]
    items=[(k,'https://arxiv.org/pdf/2506.00625v2' if k=='PI_H2T' else u) for k,u in items]
    items.append(('H2T_ARXIV_V3','https://arxiv.org/pdf/2306.06963v3'))
    (RUN/'paper_cache').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        result=list(pool.map(collect,items))
    with (RUN/'PAPER_ACCESS.json').open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
