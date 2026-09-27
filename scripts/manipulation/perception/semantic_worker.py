"""GPU-only YOLO-World and CLIP worker using a JSON-lines protocol."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def _text_features(
    prompts: dict[str, list[str]], model: Any, device: str, clip: Any, torch: Any
) -> tuple[list[str], Any]:
    labels = list(prompts)
    features = []
    with torch.inference_mode():
        for label in labels:
            tokens = clip.tokenize(prompts[label]).to(device)
            encoded = model.encode_text(tokens)
            encoded /= encoded.norm(dim=-1, keepdim=True)
            mean = encoded.mean(dim=0)
            features.append(mean / mean.norm())
    return labels, torch.stack(features)


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _cross_class_nms(detections: list[dict[str, Any]], threshold: float = 0.65):
    kept = []
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        if all(_iou(detection["bbox_xyxy"], other["bbox_xyxy"]) < threshold for other in kept):
            kept.append(detection)
    return kept


def main() -> int:
    args = _parse_args()
    import clip
    import numpy as np
    import torch
    from PIL import Image

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Semantic worker requires CUDA; refusing a silent CPU fallback. "
            f"requested={args.device!r}, torch={torch.__version__}, cuda={torch.version.cuda!r}"
        )
    from ultralytics import YOLOWorld

    device = args.device
    with contextlib.redirect_stdout(sys.stderr):
        clip_model, preprocess = clip.load(args.clip_model, device=device)
        detector = YOLOWorld(args.yolo_model)
    clip_model.eval()
    print(
        json.dumps(
            {
                "ready": True,
                "device": device,
                "gpu": torch.cuda.get_device_name(torch.device(device)),
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    cached_prompts = ""
    cached_labels: list[str] = []
    cached_features: Any = None
    detector_prompt_map: list[str] = []
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if request.get("command") == "close":
                return 0
            prompts = request["prompts"]
            signature = json.dumps(prompts, ensure_ascii=True, sort_keys=True)
            if signature != cached_prompts:
                cached_labels, cached_features = _text_features(
                    prompts, clip_model, device, clip, torch
                )
                detector_mapping = request["detector_prompts"]
                detector_prompt_map = list(detector_mapping)
                detector_prompts = [detector_mapping[label] for label in detector_prompt_map]
                with contextlib.redirect_stdout(sys.stderr):
                    detector.set_classes(detector_prompts)
                cached_prompts = signature
            images = [
                Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                for encoded in request["crops_png_base64"]
            ]
            batch = torch.stack([preprocess(image) for image in images]).to(device)
            with torch.inference_mode():
                image_features = clip_model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probabilities = (100.0 * image_features @ cached_features.T).softmax(dim=-1)
            full_image = Image.open(
                io.BytesIO(base64.b64decode(request["rgb_png_base64"]))
            ).convert("RGB")
            bgr = np.asarray(full_image)[:, :, ::-1]
            with contextlib.redirect_stdout(sys.stderr):
                result = detector.predict(
                    bgr,
                    device=0,
                    imgsz=args.imgsz,
                    conf=args.confidence,
                    iou=0.5,
                    max_det=50,
                    verbose=False,
                )[0]
            detections = []
            for box, cls, score in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf, strict=False
            ):
                detections.append(
                    {
                        "label": detector_prompt_map[int(cls)],
                        "score": float(score),
                        "bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()],
                    }
                )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "device": device,
                        "labels": cached_labels,
                        "probabilities": probabilities.float().cpu().tolist(),
                        "detections": _cross_class_nms(detections),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
