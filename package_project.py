"""Package reviewed outputs and reusable sources; exclude credentials and caches."""
from pathlib import Path
import argparse, zipfile
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',default='../Generative_AI_2027_Teaching_Pack.zip');p.add_argument('--include-history',action='store_true');args=p.parse_args()
out=(ROOT/args.output).resolve()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file() or f.resolve()==out:continue
        rel=f.relative_to(ROOT)
        if any(x in rel.parts for x in ('.venv','__pycache__','.git','drafts','backups')):continue
        if f.suffix in ('.pyc','.key') or f.name.startswith('.env') or f.name.endswith('.local.json'):continue
        if not args.include_history and rel.parts[:2]==('outputs','runs'):continue
        # Actual provider config is machine-specific; ship examples without credentials.
        if rel.as_posix()=='config/providers.json':continue
        z.write(f,Path(ROOT.name)/rel)
    z.write(ROOT/'config/providers.example.json',Path(ROOT.name)/'config/providers.json')
with zipfile.ZipFile(out) as z:
    if z.testzip():raise RuntimeError('ZIP integrity check failed')
print(out,round(out.stat().st_size/1048576,2),'MiB')
