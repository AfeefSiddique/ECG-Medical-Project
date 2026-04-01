import os, io, sys, base64, json
from datetime import datetime
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import beats_to_features
from preprocess import clean_signal

clf = joblib.load(os.path.join(os.path.dirname(__file__), 'model.joblib'))

app = FastAPI()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CardioScan — ECG Arrhythmia Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root {
  --bg:#f4f6f9; --surface:#ffffff; --surface2:#f0f3f7;
  --border:#dde2ea; --border2:#c8cfd9;
  --text:#1a2233; --muted:#6b7a94; --faint:#a8b4c4;
  --teal:#0d7c6e; --teal-lt:#e6f4f2; --teal-mid:#1aaa97;
  --blue:#1e5fa8; --blue-lt:#e8f0fb;
  --red:#c0392b; --red-lt:#fdf0ee; --red-mid:#e05a4a;
  --amber:#b45309; --amber-lt:#fef3e2;
  --sans:'IBM Plex Sans',sans-serif;
  --mono:'IBM Plex Mono',monospace;
  --serif:'Instrument Serif',serif;
  --radius:10px;
  --shadow:0 1px 3px rgba(0,0,0,0.07),0 4px 16px rgba(0,0,0,0.05);
}
html{background:var(--bg);color:var(--text);font-family:var(--sans);}
body{min-height:100vh;}
.page{display:grid;grid-template-columns:260px 1fr;min-height:100vh;}

/* Sidebar */
.sidebar{background:var(--surface);border-right:1px solid var(--border);
  padding:2rem 1.5rem;position:sticky;top:0;height:100vh;
  display:flex;flex-direction:column;}
.logo{display:flex;align-items:center;gap:10px;margin-bottom:2.5rem;}
.logo-mark{width:34px;height:34px;border-radius:8px;background:var(--teal);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.logo-mark svg{width:18px;height:18px;}
.logo-name{font-family:var(--serif);font-size:1.2rem;color:var(--text);}
.logo-name span{color:var(--teal);font-style:italic;}
.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:0.12em;
  color:var(--faint);margin-bottom:8px;padding-left:8px;}
.nav-item{display:flex;align-items:center;gap:10px;padding:8px 10px;
  border-radius:7px;margin-bottom:2px;font-size:13px;color:var(--muted);
  cursor:pointer;transition:background 0.15s,color 0.15s;}
.nav-item:hover{background:var(--surface2);color:var(--text);}
.nav-item.active{background:var(--teal-lt);color:var(--teal);}
.nav-dot{width:7px;height:7px;border-radius:50%;background:currentColor;
  opacity:0.6;flex-shrink:0;}
.sidebar-footer{margin-top:auto;padding-top:1.5rem;
  border-top:1px solid var(--border);font-size:11px;color:var(--faint);line-height:1.6;}
.sidebar-footer a{color:var(--teal);text-decoration:none;}

/* Main */
.main{padding:2.5rem 2.5rem 4rem;max-width:860px;}
.page-header{margin-bottom:2rem;}
.page-header h1{font-family:var(--serif);font-size:1.9rem;font-weight:400;
  color:var(--text);margin-bottom:4px;}
.page-header h1 em{font-style:italic;color:var(--teal);}
.page-header p{font-size:13px;color:var(--muted);line-height:1.7;}

/* Stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:2rem;}
.stat-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.1rem 1.2rem;box-shadow:var(--shadow);}
.stat-val{font-family:var(--serif);font-size:1.7rem;color:var(--teal);
  line-height:1;margin-bottom:3px;}
.stat-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}

/* Section title */
.section-title{font-size:11px;text-transform:uppercase;letter-spacing:0.12em;
  color:var(--faint);margin-bottom:10px;}

/* Upload */
.upload-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:var(--shadow);margin-bottom:1.2rem;overflow:hidden;}
.upload-zone{padding:2.5rem 2rem;text-align:center;
  border-bottom:1px dashed var(--border);cursor:pointer;position:relative;
  transition:background 0.15s;}
.upload-zone:hover,.upload-zone.drag{background:var(--teal-lt);}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.upload-icon-wrap{width:44px;height:44px;border-radius:10px;border:1px solid var(--border2);
  background:var(--surface2);display:flex;align-items:center;justify-content:center;
  margin:0 auto 12px;font-size:18px;}
.upload-title{font-size:14px;font-weight:500;color:var(--text);margin-bottom:4px;}
.upload-hint{font-size:12px;color:var(--muted);}
.upload-filename{display:none;margin-top:10px;font-size:12px;color:var(--teal);font-family:var(--mono);}
.upload-actions{padding:1rem 1.2rem;display:flex;gap:10px;align-items:center;}

/* Buttons */
.btn-primary{padding:9px 20px;background:var(--teal);color:#fff;border:none;
  border-radius:7px;font-family:var(--sans);font-size:13px;font-weight:500;
  cursor:pointer;transition:background 0.15s;}
.btn-primary:hover{background:#0a6860;}
.btn-primary:disabled{background:var(--faint);cursor:not-allowed;}
.btn-secondary{padding:9px 18px;background:var(--surface);color:var(--muted);
  border:1px solid var(--border2);border-radius:7px;font-family:var(--sans);
  font-size:13px;cursor:pointer;transition:all 0.15s;
  display:flex;align-items:center;gap:7px;}
.btn-secondary:hover{border-color:var(--teal);color:var(--teal);}
.btn-secondary:disabled{opacity:0.4;cursor:not-allowed;}

/* Loading */
#loading{display:none;padding:2.5rem;background:var(--surface);
  border:1px solid var(--border);border-radius:var(--radius);
  text-align:center;box-shadow:var(--shadow);margin-bottom:1.2rem;}
.spinner{width:28px;height:28px;border:2px solid var(--border);
  border-top-color:var(--teal);border-radius:50%;
  animation:spin 0.7s linear infinite;margin:0 auto 12px;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-txt{font-size:13px;color:var(--muted);}

/* Result */
#result{display:none;}
.result-banner{border-radius:var(--radius) var(--radius) 0 0;
  padding:1.2rem 1.5rem;display:flex;align-items:center;gap:14px;
  border:1px solid;border-bottom:none;}
.result-banner.normal{background:var(--teal-lt);border-color:#a8d8d3;}
.result-banner.ventricular{background:var(--red-lt);border-color:#f0b8b2;}
.status-badge{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.normal .status-badge{background:var(--teal);box-shadow:0 0 0 3px rgba(13,124,110,0.15);}
.ventricular .status-badge{background:var(--red);box-shadow:0 0 0 3px rgba(192,57,43,0.15);}
.result-title{font-family:var(--serif);font-size:1.3rem;}
.normal .result-title{color:var(--teal);}
.ventricular .result-title{color:var(--red);}
.result-conf-wrap{margin-left:auto;text-align:right;}
.result-conf-label{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--faint);}
.result-conf-val{font-family:var(--serif);font-size:1.2rem;color:var(--text);}
.result-body{background:var(--surface);border:1px solid;border-top:none;
  border-radius:0 0 var(--radius) var(--radius);box-shadow:var(--shadow);overflow:hidden;}
.normal .result-body{border-color:#a8d8d3;}
.ventricular .result-body{border-color:#f0b8b2;}
#ecg-plot{width:100%;display:block;}

/* Report grid */
.report-grid{display:grid;grid-template-columns:1fr 1fr 1fr;
  border-top:1px solid var(--border);}
.report-cell{padding:1rem 1.2rem;border-right:1px solid var(--border);}
.report-cell:last-child{border-right:none;}
.report-cell-key{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;
  color:var(--faint);margin-bottom:4px;}
.report-cell-val{font-size:13px;color:var(--text);font-family:var(--mono);}

/* Features */
.feature-section{border-top:1px solid var(--border);padding:1.2rem 1.5rem;}
.feature-section-title{font-size:11px;text-transform:uppercase;
  letter-spacing:0.12em;color:var(--faint);margin-bottom:12px;}
.feature-rows{display:grid;grid-template-columns:1fr 1fr;gap:6px 2rem;}
.feature-row{display:flex;justify-content:space-between;align-items:center;}
.feature-name{font-size:12px;color:var(--muted);min-width:90px;}
.feature-bar-wrap{flex:1;margin:0 10px;height:3px;
  background:var(--surface2);border-radius:2px;}
.feature-bar{height:3px;border-radius:2px;background:var(--teal);}
.feature-val{font-size:11px;font-family:var(--mono);color:var(--text);
  min-width:52px;text-align:right;}

/* Download bar */
.download-bar{display:flex;gap:10px;align-items:center;
  padding:1rem 1.5rem;border-top:1px solid var(--border);background:var(--surface2);}
.download-bar span{font-size:12px;color:var(--muted);margin-right:auto;}

/* Clinical note */
.clinical-note{margin-top:1.2rem;padding:1rem 1.2rem;
  background:var(--amber-lt);border:1px solid #f0d09a;
  border-radius:var(--radius);font-size:12px;color:var(--amber);
  line-height:1.6;display:none;}
.clinical-note strong{font-weight:500;}

.fade{animation:fd 0.4s ease;}
@keyframes fd{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>
<div class="page">
  <aside class="sidebar">
    <div class="logo">
      <div class="logo-mark">
        <svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 9h3l2-6 3 12 2-8 2 4h4" stroke="white" stroke-width="1.5"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="logo-name">Cardio<span>Scan</span></div>
    </div>
    <div class="nav-label">Tools</div>
    <div class="nav-item active"><div class="nav-dot"></div> Beat Analyser</div>
    <div class="nav-item"><div class="nav-dot"></div> About the Model</div>
    <div class="nav-item"><div class="nav-dot"></div> Dataset Info</div>
    <div style="margin-top:1.5rem">
      <div class="nav-label">Model Info</div>
      <div style="font-size:12px;color:var(--muted);line-height:1.8;padding-left:8px">
        Algorithm: Random Forest<br>
        Trees: 100<br>Features: 18<br>
        Training: MIT-BIH DB<br>
        Records: 100, 106, 119, 200<br>
        Test patient: 208
      </div>
    </div>
    <div class="sidebar-footer">
      Built by Afeef Siddique<br>
      BSc ETE · RUET · 2026<br>
      <a href="https://github.com/AfeefSiddique/ECG-Medical-Project" target="_blank">GitHub ↗</a>
    </div>
  </aside>

  <main class="main">
    <div class="page-header">
      <h1>ECG Beat <em>Analysis</em></h1>
      <p>Upload a 200-sample ECG beat window (single-column CSV, 360 Hz) to classify as Normal or Ventricular arrhythmia.</p>
    </div>

    <div class="stats">
      <div class="stat-card"><div class="stat-val">99%</div><div class="stat-label">F1-score</div></div>
      <div class="stat-card"><div class="stat-val">11k+</div><div class="stat-label">Training beats</div></div>
      <div class="stat-card"><div class="stat-val">0</div><div class="stat-label">Missed arrhythmias</div></div>
      <div class="stat-card"><div class="stat-val">18</div><div class="stat-label">Signal features</div></div>
    </div>

    <div class="section-title">Input</div>
    <div class="upload-card">
      <div class="upload-zone" id="dropzone">
        <input type="file" id="file-input" accept=".csv"/>
        <div class="upload-icon-wrap">↑</div>
        <div class="upload-title">Drop ECG CSV file here</div>
        <div class="upload-hint">Single column · 200 samples · 360 Hz sampling rate</div>
        <div class="upload-filename" id="file-name"></div>
      </div>
      <div class="upload-actions">
        <button class="btn-primary" id="submit-btn" disabled onclick="submitFile()">Analyse beat</button>
        <button class="btn-secondary" onclick="clearAll()">Clear</button>
        <span style="font-size:11px;color:var(--faint);margin-left:auto">
          MIT-BIH Arrhythmia Database · PhysioNet
        </span>
      </div>
    </div>

    <div id="loading">
      <div class="spinner"></div>
      <div class="loading-txt">Processing signal…</div>
    </div>

    <div id="result">
      <div class="section-title" style="margin-top:1.5rem">Analysis Result</div>
      <div id="result-wrap">
        <div class="result-banner" id="result-banner">
          <div class="status-badge"></div>
          <div class="result-title" id="result-label"></div>
          <div class="result-conf-wrap">
            <div class="result-conf-label">Confidence</div>
            <div class="result-conf-val" id="result-conf"></div>
          </div>
        </div>
        <div class="result-body" id="result-body">
          <img id="ecg-plot" alt="ECG waveform"/>
          <div class="report-grid">
            <div class="report-cell">
              <div class="report-cell-key">Analysis time</div>
              <div class="report-cell-val" id="meta-time">—</div>
            </div>
            <div class="report-cell">
              <div class="report-cell-key">Sampling rate</div>
              <div class="report-cell-val">360 Hz</div>
            </div>
            <div class="report-cell">
              <div class="report-cell-key">Window size</div>
              <div class="report-cell-val">200 samples</div>
            </div>
          </div>
          <div class="feature-section">
            <div class="feature-section-title">Top signal features</div>
            <div class="feature-rows" id="feature-rows"></div>
          </div>
          <div class="download-bar">
            <span>Report ready — waveform · prediction · confidence · feature breakdown</span>
            <button class="btn-secondary" onclick="downloadReport()">↓ Download PDF report</button>
            <button class="btn-secondary" onclick="downloadJSON()">↓ JSON</button>
          </div>
        </div>
      </div>
      <div class="clinical-note" id="clinical-note">
        <strong>Clinical note:</strong> This tool is for research and educational purposes only.
        It is not a certified medical device and should not be used for clinical diagnosis.
        Always consult a qualified cardiologist for medical decisions.
      </div>
    </div>
  </main>
</div>

<script>
const fileInput = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const submitBtn  = document.getElementById('submit-btn');
const dropzone   = document.getElementById('dropzone');
let selectedFile = null;
let lastResult   = null;

fileInput.addEventListener('change', e => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    fileNameEl.textContent = selectedFile.name;
    fileNameEl.style.display = 'block';
    submitBtn.disabled = false;
  }
});
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag');
  selectedFile = e.dataTransfer.files[0];
  if (selectedFile) {
    fileNameEl.textContent = selectedFile.name;
    fileNameEl.style.display = 'block';
    submitBtn.disabled = false;
  }
});

function clearAll() {
  selectedFile = null;
  fileNameEl.style.display = 'none';
  fileNameEl.textContent = '';
  submitBtn.disabled = true;
  document.getElementById('result').style.display = 'none';
  document.getElementById('clinical-note').style.display = 'none';
  fileInput.value = '';
}

async function submitFile() {
  if (!selectedFile) return;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display  = 'none';
  submitBtn.disabled = true;
  const form = new FormData();
  form.append('file', selectedFile);
  try {
    const res  = await fetch('/predict', { method:'POST', body:form });
    const data = await res.json();
    lastResult = data;
    lastResult.filename = selectedFile.name;
    renderResult(data);
  } catch(err) {
    document.getElementById('loading').style.display = 'none';
    alert('Error: ' + err.message);
  }
  submitBtn.disabled = false;
}

function renderResult(data) {
  const isV    = data.prediction === 'Ventricular (Arrhythmia)';
  const cls    = isV ? 'ventricular' : 'normal';
  const banner = document.getElementById('result-banner');
  const body   = document.getElementById('result-body');
  const wrap   = document.getElementById('result-wrap');
  banner.className = 'result-banner ' + cls;
  body.className   = 'result-body '   + cls;
  wrap.className   = cls;
  document.getElementById('result-label').textContent = data.prediction;
  document.getElementById('result-conf').textContent  = data.confidence;
  document.getElementById('ecg-plot').src = 'data:image/png;base64,' + data.plot;
  document.getElementById('meta-time').textContent = new Date().toLocaleTimeString();

  const rows   = document.getElementById('feature-rows');
  rows.innerHTML = '';
  const feats  = data.features.slice(0, 8);
  const maxVal = Math.max(...feats.map(f => Math.abs(f.value)));
  feats.forEach(f => {
    const pct = Math.min(100, (Math.abs(f.value) / maxVal) * 100).toFixed(0);
    rows.innerHTML += `
      <div class="feature-row">
        <span class="feature-name">${f.name}</span>
        <div class="feature-bar-wrap">
          <div class="feature-bar" style="width:${pct}%"></div>
        </div>
        <span class="feature-val">${f.display}</span>
      </div>`;
  });

  document.getElementById('loading').style.display = 'none';
  const resultEl = document.getElementById('result');
  resultEl.style.display = 'block';
  resultEl.classList.add('fade');
  document.getElementById('clinical-note').style.display = 'block';
}

async function downloadReport() {
  if (!lastResult) return;
  const res  = await fetch('/report', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(lastResult)
  });
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'cardioscan_report.pdf'; a.click();
  URL.revokeObjectURL(url);
}

function downloadJSON() {
  if (!lastResult) return;
  const clean = {...lastResult}; delete clean.plot;
  const blob  = new Blob([JSON.stringify(clean, null, 2)], {type:'application/json'});
  const url   = URL.createObjectURL(blob);
  const a     = document.createElement('a');
  a.href = url; a.download = 'cardioscan_result.json'; a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""


def make_ecg_plot(beat_clean, color):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfc')
    ax.plot(beat_clean, color=color, linewidth=1.4)
    ax.axvline(x=90, color='#c8cfd9', linestyle='--', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#dde2ea')
        spine.set_linewidth(0.6)
    ax.tick_params(colors='#6b7a94', labelsize=8)
    ax.set_xlabel('Sample', color='#6b7a94', fontsize=9)
    ax.set_ylabel('Amplitude (mV)', color='#6b7a94', fontsize=9)
    ax.grid(True, alpha=0.4, color='#e8edf3', linewidth=0.5)
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, facecolor='#ffffff')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return b64


def get_features_display(beat_clean):
    f    = beats_to_features(np.array([beat_clean]))
    cols = f.columns.tolist()
    vals = f.values[0].tolist()
    display_names = {
        'energy_twave':'T-wave energy','skewness':'Skewness',
        'min':'Signal min','qrs_width':'QRS width',
        'power_lf':'Low-freq power','kurtosis':'Kurtosis',
        'std':'Std deviation','rms':'RMS amplitude',
    }
    result = []
    for name, val in zip(cols, vals):
        result.append({
            'name': display_names.get(name, name),
            'value': float(val),
            'display': f'{val:.4f}' if abs(val) < 100 else f'{val:.1f}'
        })
    result.sort(key=lambda x: abs(x['value']), reverse=True)
    return result[:8], dict(zip(cols, vals))


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents   = await file.read()
    raw        = np.loadtxt(io.StringIO(contents.decode()), delimiter=',')
    if len(raw) < 200:
        return JSONResponse({"error": "Need at least 200 samples"}, status_code=400)
    beat       = raw[:200]
    beat_clean = clean_signal(beat, fs=360)
    feats      = beats_to_features(np.array([beat_clean]))
    pred       = clf.predict(feats)[0]
    prob       = clf.predict_proba(feats)[0]
    confidence = f"{prob[pred]*100:.1f}%"
    label      = "Ventricular (Arrhythmia)" if pred == 1 else "Normal"
    color      = "#c0392b" if pred == 1 else "#0d7c6e"
    plot_b64   = make_ecg_plot(beat_clean, color)
    features_display, features_raw = get_features_display(beat_clean)
    return JSONResponse({
        "prediction":   label,
        "confidence":   confidence,
        "pred_int":     int(pred),
        "prob_normal":  float(prob[0]),
        "prob_v":       float(prob[1]),
        "plot":         plot_b64,
        "features":     features_display,
        "features_raw": features_raw,
        "timestamp":    datetime.now().isoformat(),
    })


@app.post("/report")
async def generate_report(data: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib.units import cm
    except ImportError:
        return Response(content="Install reportlab: pip install reportlab", status_code=500)

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    teal   = colors.HexColor('#0d7c6e')
    slate  = colors.HexColor('#1a2233')
    muted  = colors.HexColor('#6b7a94')
    red    = colors.HexColor('#c0392b')
    border = colors.HexColor('#dde2ea')
    ltbg   = colors.HexColor('#f4f6f9')
    is_v   = data.get('pred_int', 0) == 1
    rc     = red if is_v else teal

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s = ps('t', fontSize=22, textColor=slate, fontName='Helvetica-Bold', spaceAfter=4)
    sub_s   = ps('s', fontSize=10, textColor=muted, fontName='Helvetica', spaceAfter=16)
    h2_s    = ps('h', fontSize=12, textColor=slate, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    body_s  = ps('b', fontSize=9, textColor=muted, fontName='Helvetica', leading=14)
    note_s  = ps('n', fontSize=8, textColor=colors.HexColor('#b45309'),
                 fontName='Helvetica', leading=12,
                 backColor=colors.HexColor('#fef3e2'),
                 borderPadding=(6, 8, 6, 8))

    story = []
    story.append(Paragraph("CardioScan — ECG Analysis Report", title_s))
    ts = data.get('timestamp', datetime.now().isoformat())[:19].replace('T', ' ')
    story.append(Paragraph(f"Generated: {ts}  ·  File: {data.get('filename','—')}", sub_s))
    story.append(Table([['']], colWidths=[17*cm],
                       style=[('LINEABOVE', (0,0), (-1,0), 0.5, border)]))
    story.append(Spacer(1, 12))

    pn   = f"{data.get('prob_normal',0)*100:.1f}%"
    pv   = f"{data.get('prob_v',0)*100:.1f}%"
    summary = [
        ['Prediction', data.get('prediction','—'), 'Confidence', data.get('confidence','—')],
        ['P(Normal)',  pn,  'P(Ventricular)', pv],
        ['Sampling rate', '360 Hz', 'Window', '200 samples'],
    ]
    t = Table(summary, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (-1,-1), muted),
        ('TEXTCOLOR', (1,0), (1,0), rc),
        ('FONTNAME',  (1,0), (1,0), 'Helvetica-Bold'),
        ('FONTSIZE',  (1,0), (1,0), 11),
        ('BACKGROUND',(0,0), (-1,-1), ltbg),
        ('GRID',      (0,0), (-1,-1), 0.4, border),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, ltbg, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("ECG Waveform", h2_s))
    if data.get('plot'):
        img_buf = io.BytesIO(base64.b64decode(data['plot']))
        story.append(RLImage(img_buf, width=17*cm, height=4.8*cm))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Signal Feature Analysis", h2_s))
    relevance = {
        'T-wave energy': 'High in V beats — broad T-wave morphology',
        'Skewness':      'V beats are morphologically asymmetric',
        'Signal min':    'Deep negative deflection in V beats',
        'QRS width':     'V beats are wider than normal (>120ms)',
        'Low-freq power':'Spectral content differs between beat types',
        'Kurtosis':      'Peakedness of the waveform distribution',
        'Std deviation': 'Amplitude variability across the beat',
        'RMS amplitude': 'Root mean square energy of the beat',
    }
    feat_data = [['Feature', 'Value', 'Clinical relevance']]
    for f in data.get('features', []):
        feat_data.append([f['name'], f['display'], relevance.get(f['name'], '—')])
    ft = Table(feat_data, colWidths=[4*cm, 3*cm, 10*cm])
    ft.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',  (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND',(0,0), (-1,0), teal),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ltbg]),
        ('GRID',      (0,0), (-1,-1), 0.3, border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(ft)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Model Information", h2_s))
    model_rows = [
        ['Algorithm',       'Random Forest (100 decision trees)'],
        ['Training data',   'MIT-BIH Arrhythmia Database (PhysioNet)'],
        ['Training records','100, 106, 119, 200'],
        ['Test record',     '208 — completely unseen patient'],
        ['F1-score',        '0.99 Normal / 0.99 Ventricular'],
        ['False negatives', '0 — no arrhythmias missed on test patient'],
        ['Features',        '18 hand-crafted signal features'],
    ]
    mt = Table(model_rows, colWidths=[5*cm, 12*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (0,0), (-1,-1), muted),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, ltbg]),
        ('GRID',      (0,0), (-1,-1), 0.3, border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an experimental ML model for "
        "research and educational purposes only. It is not a certified medical device "
        "and must not be used for clinical diagnosis or treatment decisions. "
        "Always consult a qualified cardiologist.",
        note_s))

    doc.build(story)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=cardioscan_report.pdf'}
    )


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)