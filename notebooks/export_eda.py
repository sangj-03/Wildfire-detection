"""eda.ipynb를 공유용 묶음으로 내보낸다: 코드 뺀 HTML → 인쇄용 CSS → Chrome PDF(A4 가로) → zip(eda.ipynb, eda_lib.py, eda_view.pdf).
사용: ../.venv_eda/bin/python export_eda.py  →  export/eda_view.html, eda_view.pdf, team7_aihub_eda.zip
"""
import os, subprocess, sys, zipfile

NB = os.path.dirname(os.path.abspath(__file__))
OUT = f'{NB}/export'
STYLE = '''<style>
@page { size: A4 landscape; margin: 10mm 12mm; }
body { font-family: "Noto Sans CJK KR", sans-serif; }
.jp-Notebook { padding: 0 !important; }
.jp-OutputArea-output img { max-width: 100% !important; height: auto; break-inside: avoid; }
.jp-RenderedHTMLCommon table { font-size: 9px; break-inside: auto; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding: 2px 5px; }
.jp-OutputArea-output pre { font-size: 9px; white-space: pre-wrap; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
</style>
'''
os.makedirs(OUT, exist_ok=True)
subprocess.run([sys.executable, '-m', 'nbconvert', '--to', 'html', '--no-input', f'{NB}/eda.ipynb',
                '--output', 'eda_view', '--output-dir', OUT], check=True, capture_output=True)
html = f'{OUT}/eda_view.html'
s = open(html).read()
s = s.replace('</head>', STYLE + '</head>', 1)
open(html, 'w').write(s)
subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--no-pdf-header-footer',
                f'--print-to-pdf={OUT}/eda_view.pdf', f'file://{html}'], check=True, capture_output=True)
with zipfile.ZipFile(f'{OUT}/team7_aihub_eda.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f, arc in [(f'{NB}/eda.ipynb', 'eda.ipynb'), (f'{NB}/eda_lib.py', 'eda_lib.py'), (f'{OUT}/eda_view.pdf', 'eda_view.pdf')]:
        z.write(f, arc)
print('saved', OUT, {f: os.path.getsize(f'{OUT}/{f}') for f in ('eda_view.pdf', 'team7_aihub_eda.zip')})
