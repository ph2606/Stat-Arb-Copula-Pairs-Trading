"""Package the local research submission, excluding environment files and source PDFs."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'Panagiotis_Housos_Copula_Pairs_Trading.zip'
EXCLUDE={'.git','.cache','.venv','.packages','.quality','__pycache__','.pytest_cache','.yf_cache','.ipynb_checkpoints'}
ALLOWED={'.py','.md','.tex','.pdf','.png','.ipynb','.json','.csv','.txt'}

def main():
    files=[]
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in EXCLUDE for part in path.relative_to(ROOT).parts):
            continue
        if path.name=='submission_manifest.json':
            continue
        if path.parent.name=='references' and path.suffix in {'.txt','.png','.pdf'}:
            continue
        if path.suffix not in ALLOWED and path.name not in {'.gitignore','.gitattributes'}:
            continue
        files.append(path)
    hashes={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    manifest=ROOT/'submission_manifest.json'
    manifest.write_text(json.dumps({'files':hashes,'excluded':'Installed dependencies, temporary files, extracted/full supplied papers.'},indent=2)+'\n')
    with ZipFile(ARCHIVE,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files+[manifest]:
            archive.write(path,arcname=str(Path(ROOT.name)/path.relative_to(ROOT)))
    with ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    print(f'{ARCHIVE.name}: {len(files)+1} files, {ARCHIVE.stat().st_size/1e6:.1f} MB; CRC verified')

if __name__=='__main__':
    main()
