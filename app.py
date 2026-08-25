"""
Flask web UI for the AI deblur agent.
Run: python app.py  then open http://localhost:5000
"""

import os
import uuid
from pathlib import Path
from flask import Flask, request, render_template_string, send_file, jsonify
import deblur_agent

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Deblur - 4K Image Restoration</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#0f0f1a;color:#e0e0f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem 1rem}
  h1{font-size:2rem;font-weight:700;margin-bottom:.4rem;background:linear-gradient(135deg,#7c6fff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  p.sub{color:#888;margin-bottom:2rem;font-size:.95rem}
  .card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:16px;padding:2rem;width:100%;max-width:620px;box-shadow:0 8px 40px rgba(0,0,0,.4)}
  label.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;border:2px dashed #3a3a6a;border-radius:12px;height:160px;cursor:pointer;transition:.2s;gap:.6rem;color:#666}
  label.drop:hover{border-color:#7c6fff;background:#1f1f3a}
  label.drop svg{width:40px;height:40px;stroke:#7c6fff}
  input[type=file]{display:none}
  input[type=text]{width:100%;padding:.7rem 1rem;margin-top:1rem;background:#0f0f1a;border:1px solid #2a2a4a;border-radius:8px;color:#e0e0f0;font-size:.95rem}
  button{margin-top:1.2rem;width:100%;padding:.85rem;background:linear-gradient(135deg,#7c6fff,#00d4ff);border:none;border-radius:10px;color:#fff;font-size:1rem;font-weight:600;cursor:pointer;transition:.2s}
  button:hover{opacity:.88}
  button:disabled{opacity:.4;cursor:not-allowed}
  #status{margin-top:1.2rem;font-size:.9rem;color:#aaa;min-height:1.4rem}
  #progress{width:100%;height:6px;background:#2a2a4a;border-radius:3px;margin-top:.8rem;overflow:hidden;display:none}
  #bar{height:100%;width:0;background:linear-gradient(90deg,#7c6fff,#00d4ff);border-radius:3px;transition:width .4s}
  .result{margin-top:1.5rem;display:none}
  .result img{width:100%;border-radius:10px;border:1px solid #2a2a4a}
  .result a{display:inline-block;margin-top:.8rem;padding:.55rem 1.4rem;background:#7c6fff;border-radius:8px;color:#fff;text-decoration:none;font-weight:600}
  .params{background:#0f0f1a;border-radius:8px;padding:1rem;margin-top:1rem;font-size:.82rem;color:#aaa;white-space:pre-wrap;word-break:break-all}
  #preview-name{margin-top:.6rem;font-size:.85rem;color:#7c6fff;text-align:center}
</style>
</head>
<body>
<h1>AI Deblur Agent</h1>
<p class="sub">Upload a blurry photo and get a sharp 4K image powered by Claude AI</p>
<div class="card">
  <label class="drop" id="drop-zone">
    <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg>
    <span>Click or drag and drop an image</span>
    <span style="font-size:.8rem">JPG, PNG, BMP, TIFF, WEBP</span>
    <input type="file" id="file-input" accept="image/*"/>
  </label>
  <div id="preview-name"></div>
  <input type="text" id="api-key" placeholder="ANTHROPIC_API_KEY (or set as env var)" value=""/>
  <button id="run-btn" onclick="runDeblur()">Remove Blur and Upscale to 4K</button>
  <div id="progress"><div id="bar"></div></div>
  <div id="status"></div>
  <div class="result" id="result-box">
    <h3 style="margin-bottom:.6rem;color:#7c6fff">Result</h3>
    <img id="result-img" src="" alt="deblurred"/>
    <br/>
    <a id="dl-link" href="#" download>Download 4K Image</a>
    <div class="params" id="params-box"></div>
  </div>
</div>
<script>
const fi = document.getElementById('file-input');
fi.addEventListener('change', () => {
  if(fi.files[0]) document.getElementById('preview-name').textContent = fi.files[0].name;
});
document.getElementById('drop-zone').addEventListener('dragover', e => e.preventDefault());
document.getElementById('drop-zone').addEventListener('drop', e => {
  e.preventDefault();
  fi.files = e.dataTransfer.files;
  if(fi.files[0]) document.getElementById('preview-name').textContent = fi.files[0].name;
});
function setStatus(msg){ document.getElementById('status').textContent = msg; }
function setBar(pct){ document.getElementById('bar').style.width = pct + '%'; }
async function runDeblur(){
  const file = fi.files[0];
  if(!file){ setStatus('Please select an image first.'); return; }
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  document.getElementById('progress').style.display = 'block';
  document.getElementById('result-box').style.display = 'none';
  setBar(10); setStatus('Uploading image...');
  const fd = new FormData();
  fd.append('image', file);
  const key = document.getElementById('api-key').value.trim();
  if(key) fd.append('api_key', key);
  try {
    setBar(25); setStatus('Analysing blur with Claude AI...');
    const res = await fetch('/deblur', { method:'POST', body:fd });
    setBar(75); setStatus('Running deblur pipeline...');
    const data = await res.json();
    if(!res.ok || data.error){ setStatus('Error: ' + (data.error||res.statusText)); btn.disabled=false; return; }
    setBar(100); setStatus('Done!');
    document.getElementById('result-img').src = data.output_url + '?t=' + Date.now();
    document.getElementById('dl-link').href = data.output_url;
    document.getElementById('dl-link').download = data.filename;
    document.getElementById('params-box').textContent = JSON.stringify(data.params, null, 2);
    document.getElementById('result-box').style.display = 'block';
  } catch(e){ setStatus('Network error: ' + e.message); }
  btn.disabled = false;
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/deblur", methods=["POST"])
def deblur():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    api_key = request.form.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not provided"}), 400
    os.environ["ANTHROPIC_API_KEY"] = api_key
    deblur_agent.client = deblur_agent.anthropic.Anthropic(api_key=api_key)
    uid = uuid.uuid4().hex
    ext = Path(file.filename).suffix or ".jpg"
    in_path = str(UPLOAD_DIR / f"{uid}_in{ext}")
    out_name = f"{uid}_deblurred_4k.jpg"
    out_path = str(OUTPUT_DIR / out_name)
    file.save(in_path)
    try:
        img = deblur_agent.load_image(in_path)
        params = deblur_agent.analyse_blur_with_claude(img)
        deblurred = deblur_agent.deblur_pipeline(img, params, verbose=False)
        final = deblur_agent.resize_to_4k(deblurred)
        deblur_agent.save_image(final, out_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"output_url": f"/output/{out_name}", "filename": out_name, "params": params})


@app.route("/output/<filename>")
def serve_output(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
