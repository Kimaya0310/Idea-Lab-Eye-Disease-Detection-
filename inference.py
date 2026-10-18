"""
inference.py -- single-image prediction.   python inference.py path/to/image.jpg

RESEARCH OUTPUT ONLY -- not a clinical diagnosis or medical device.
The decision threshold is the validation-derived operating point.
"""
import argparse
import os

import cv2
import numpy as np
import torch

import config
import utils
from engine import build_model_from_checkpoint
from explainability import generate_multi_backbone_cam, save_gradcam_figure
from preprocessing import FundusPreprocessor


def predict_image(path: str, explain: bool = True, checkpoint: str = config.BEST_MODEL_PATH,
                  borderline_margin: float = 0.10) -> dict:
    device = utils.get_device()
    model, ck = build_model_from_checkpoint(checkpoint, device)
    thr = utils.load_json(os.path.join(os.path.dirname(checkpoint), "operating_threshold.json"))["threshold"]

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    pre = FundusPreprocessor(image_size=ck.get("image_size", config.IMAGE_SIZE), use_cache=False)
    proc = pre.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x = torch.from_numpy(np.ascontiguousarray(pre.normalize(proc).transpose(2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = float(torch.softmax(model(x).float(), 1)[0, 1])
    res = {"probability_positive": prob, "threshold": thr,
           "prediction": config.POSITIVE_CLASS if prob >= thr else config.NEGATIVE_CLASS,
           "borderline": abs(prob - thr) < borderline_margin}
    if explain:
        with torch.enable_grad():
            ov = generate_multi_backbone_cam(model, x, proc, class_idx=1)
        out = os.path.join(config.OUTPUT_DIR, f"gradcam_{os.path.splitext(os.path.basename(path))[0]}.png")
        save_gradcam_figure(proc, ov, f"P({config.POSITIVE_CLASS}) = {prob:.1%} | {res['prediction']}", out)
        res["gradcam"] = out
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    r = predict_image(a.image, not a.no_explain)
    print("=" * 56 + "\nRESEARCH OUTPUT -- NOT A CLINICAL DIAGNOSIS\n" + "=" * 56)
    print(f"P({config.POSITIVE_CLASS}) : {r['probability_positive']:.1%}  (operating threshold {r['threshold']:.2f})")
    print(f"Prediction            : {r['prediction']}" + ("   [borderline - needs review]" if r["borderline"] else ""))
    if "gradcam" in r:
        print(f"Grad-CAM              : {r['gradcam']}")
