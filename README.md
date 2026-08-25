# AI Agent - Remove Blur & Upscale to 4K

An AI-powered image deblurring agent that uses **Claude** to analyse blur type and severity, then applies adaptive signal-processing to produce a sharp 4K output.

## How it works

```
Blurry image
    |
    v
[Claude AI] -- analyses blur type (motion / gaussian / out-of-focus)
    |         -- estimates kernel parameters & severity
    v
[Wiener Deconvolution] -- inverts the blur mathematically
    |
    v
[NL-Means Denoising] -- removes artefacts from deconvolution
    |
    v
[Unsharp Masking] -- fine sharpening tuned by Claude recommendations
    |
    v
[CLAHE Enhancement] -- adaptive contrast boost for crisp detail
    |
    v
[Lanczos Upscaling] -- resizes to 4K (3840x2160 max, aspect-preserved)
    |
    v
Sharp 4K image
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Command line

```bash
python deblur_agent.py blurry_photo.jpg
python deblur_agent.py blurry_photo.jpg -o sharp_result.jpg
```

### Web UI

```bash
python app.py
# Open http://localhost:5000
```

Upload your blurry image, paste your Anthropic API key, and click **Remove Blur & Upscale to 4K**.

## Blur types handled

| Type | Example cause | Algorithm |
|------|--------------|----------|
| Motion blur | Camera shake / fast subject | Directional Wiener deconvolution |
| Gaussian blur | Soft-focus, mild defocus | Gaussian Wiener deconvolution |
| Out-of-focus | Wrong focal plane | Gaussian Wiener deconvolution |

## Output

- Format: JPEG, quality 97
- Resolution: up to **3840 x 2160** (4K), aspect-ratio preserved
- Lanczos-4 interpolation for highest quality upscaling
