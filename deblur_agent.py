"""
AI Agent for removing blur from images and upscaling to 4K quality.
Uses classical signal processing (Wiener deconvolution, unsharp masking)
combined with AI-driven analysis via Claude to adaptively tune parameters.
"""

import os
import sys
import base64
import argparse
import json
import numpy as np
import cv2
from scipy.signal import wiener
import anthropic

TARGET_4K = (3840, 2160)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(img: np.ndarray, path: str) -> None:
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 97])


def encode_image_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf).decode("utf-8")


def resize_to_4k(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    tw, th = TARGET_4K
    scale = min(tw / w, th / h)
    if scale > 1.0:
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    return img


def analyse_blur_with_claude(img: np.ndarray) -> dict:
    """Ask Claude to analyse the blur type and recommend deblur parameters."""
    b64 = encode_image_b64(img)
    prompt = """You are an expert image processing engineer. Analyse this image for blur.

Return ONLY a JSON object (no markdown, no explanation) with these exact keys:
{
  "blur_type": one of "motion", "gaussian", "out_of_focus", "none",
  "severity": a float 0.0-1.0 (0=no blur, 1=extreme blur),
  "motion_angle": angle in degrees if motion blur else null,
  "motion_length": estimated kernel length in pixels (5-50) if motion blur else null,
  "gaussian_sigma": estimated sigma (1-15) if gaussian/focus blur else null,
  "wiener_noise_ratio": suggested wiener noise-to-signal ratio (0.001-0.05),
  "sharpen_strength": unsharp mask strength 0.5-3.0,
  "sharpen_radius": unsharp mask radius 1-5,
  "notes": short string with any extra observations
}"""
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "blur_type": "gaussian",
            "severity": 0.5,
            "motion_angle": None,
            "motion_length": None,
            "gaussian_sigma": 2.0,
            "wiener_noise_ratio": 0.01,
            "sharpen_strength": 1.5,
            "sharpen_radius": 2,
            "notes": "Could not parse Claude response; using defaults.",
        }


def wiener_deconvolve_channel(channel: np.ndarray, kernel: np.ndarray, noise_ratio: float) -> np.ndarray:
    H = np.fft.fft2(kernel, s=channel.shape)
    G = np.fft.fft2(channel.astype(np.float64))
    H_conj = np.conj(H)
    W = H_conj / (np.abs(H) ** 2 + noise_ratio)
    F_hat = W * G
    restored = np.fft.ifft2(F_hat).real
    return np.clip(restored, 0, 255).astype(np.uint8)


def build_motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length))
    center = length // 2
    rad = np.deg2rad(angle)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(rad)))
        y = int(round(center + offset * np.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


def apply_wiener(img: np.ndarray, kernel: np.ndarray, noise_ratio: float) -> np.ndarray:
    channels = cv2.split(img)
    deblurred = [wiener_deconvolve_channel(c, kernel, noise_ratio) for c in channels]
    return cv2.merge(deblurred)


def unsharp_mask(img: np.ndarray, strength: float, radius: int) -> np.ndarray:
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
    sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)


def enhance_details(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def deblur_pipeline(img: np.ndarray, params: dict, verbose: bool = True) -> np.ndarray:
    blur_type = params.get("blur_type", "gaussian")
    severity = float(params.get("severity", 0.5))
    noise_ratio = float(params.get("wiener_noise_ratio", 0.01))
    sharpen_strength = float(params.get("sharpen_strength", 1.5))
    sharpen_radius = int(params.get("sharpen_radius", 2))

    if verbose:
        print(f"  Blur type   : {blur_type}")
        print(f"  Severity    : {severity:.2f}")
        print(f"  Notes       : {params.get('notes', '')}")

    result = img.copy()

    if blur_type == "motion" and severity > 0.15:
        angle = float(params.get("motion_angle") or 0)
        length = int(params.get("motion_length") or 15)
        length = max(5, min(length, 50))
        if verbose:
            print(f"  Motion kernel: length={length}, angle={angle} deg")
        kernel = build_motion_kernel(length, angle)
        result = apply_wiener(result, kernel, noise_ratio)

    elif blur_type in ("gaussian", "out_of_focus") and severity > 0.15:
        sigma = float(params.get("gaussian_sigma") or 2.0)
        sigma = max(0.5, min(sigma, 10.0))
        ksize = int(sigma * 6) | 1
        if verbose:
            print(f"  Gaussian kernel: sigma={sigma:.1f}, size={ksize}")
        kernel = build_gaussian_kernel(ksize, sigma)
        result = apply_wiener(result, kernel, noise_ratio)

    if severity > 0.3:
        if verbose:
            print("  Applying denoising ...")
        result = denoise(result)

    if verbose:
        print(f"  Sharpening: strength={sharpen_strength:.1f}, radius={sharpen_radius}")
    result = unsharp_mask(result, sharpen_strength, sharpen_radius)
    result = enhance_details(result)

    return result


def run(input_path: str, output_path: str, verbose: bool = True) -> str:
    if verbose:
        print(f"\n[1/5] Loading image: {input_path}")
    img = load_image(input_path)
    h, w = img.shape[:2]
    if verbose:
        print(f"      Original size: {w}x{h}")

    if verbose:
        print("[2/5] Analysing blur with Claude AI ...")
    params = analyse_blur_with_claude(img)

    if verbose:
        print("[3/5] Running deblur pipeline ...")
    deblurred = deblur_pipeline(img, params, verbose=verbose)

    if verbose:
        print("[4/5] Upscaling to 4K resolution ...")
    final = resize_to_4k(deblurred)
    fh, fw = final.shape[:2]
    if verbose:
        print(f"      Output size: {fw}x{fh}")

    if verbose:
        print(f"[5/5] Saving -> {output_path}")
    save_image(final, output_path)

    summary = (
        f"Done! Blur type={params['blur_type']}, severity={params['severity']:.2f}. "
        f"Input {w}x{h} -> Output {fw}x{fh} saved to {output_path}"
    )
    if verbose:
        print(f"\n{summary}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent: Remove blur from an image and upscale to 4K"
    )
    parser.add_argument("input", help="Path to the blurry input image")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: <input>_deblurred_4k.jpg)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_deblurred_4k.jpg"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    run(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
