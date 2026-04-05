import json
import re
import sys
import inspect
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
MODEL_DIR = ROOT_DIR / "models" / "bert_multitask"
TOKENIZER_DIR = ROOT_DIR / "models" / "tokenizer"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model as model_module
from text_pipeline import build_model_text, build_explanation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def _find_model_class():
    candidates = []
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if obj.__module__ == model_module.__name__ and issubclass(obj, torch.nn.Module):
            candidates.append((name, obj))

    if not candidates:
        raise ImportError("No torch.nn.Module class found in src/model.py")

    priority = []
    for name, obj in candidates:
        score = 0
        lower = name.lower()
        for word in ["multitask", "bert", "sentiment", "classifier", "model"]:
            if word in lower:
                score += 1
        priority.append((score, name, obj))

    priority.sort(key=lambda x: x[0], reverse=True)
    chosen_name, chosen_cls = priority[0][1], priority[0][2]
    print(f"[INFO] Using model class: {chosen_name}")
    return chosen_cls


def _load_label_maps():
    label_map_file = MODEL_DIR / "label_maps.json"
    data = _load_json(label_map_file)

    if isinstance(data, dict):
        print(f"[INFO] label_maps top-level keys: {list(data.keys())}")

    def normalize_id2label(obj):
        if isinstance(obj, list):
            return {i: str(v) for i, v in enumerate(obj)}

        if isinstance(obj, dict):
            # id2label format
            id_to_label = {}
            for k, v in obj.items():
                if str(k).isdigit():
                    id_to_label[int(k)] = str(v)
            if id_to_label:
                return id_to_label

            # label2id format -> invert
            if all(isinstance(v, int) or str(v).isdigit() for v in obj.values()):
                return {int(v): str(k) for k, v in obj.items()}

        raise ValueError(f"Unsupported label map structure: {type(obj)}")

    polarity_section = data.get("id2polarity", data.get("polarity2id"))
    emotion_section = data.get("id2emotion", data.get("emotion2id"))
    tone_section = data.get("id2tone", data.get("tone2id"))

    if polarity_section is None or emotion_section is None or tone_section is None:
        raise KeyError(
            f"Could not locate id2polarity/id2emotion/id2tone or polarity2id/emotion2id/tone2id in {label_map_file}"
        )

    polarity_map = normalize_id2label(polarity_section)
    emotion_map = normalize_id2label(emotion_section)
    tone_map = normalize_id2label(tone_section)

    return polarity_map, emotion_map, tone_map

def _extract_outputs(outputs):
    def get_attr(obj, name):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    polarity_logits = get_attr(outputs, "polarity_logits")
    emotion_logits = get_attr(outputs, "emotion_logits")
    tone_logits = get_attr(outputs, "tone_logits")
    intensity = get_attr(outputs, "intensity")

    if polarity_logits is None and isinstance(outputs, (tuple, list)):
        if len(outputs) >= 3:
            polarity_logits, emotion_logits, tone_logits = outputs[:3]
        if len(outputs) >= 4:
            intensity = outputs[3]

    return polarity_logits, emotion_logits, tone_logits, intensity


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    polarity_map, emotion_map, tone_map = _load_label_maps()

    model_class = _find_model_class()

    config = None
    try:
        config = AutoConfig.from_pretrained(str(MODEL_DIR))
        print("[INFO] Loaded config from model folder.")
    except Exception:
        print("[INFO] No config.json in model folder. Falling back to bert-base-uncased config.")
        config = AutoConfig.from_pretrained("bert-base-uncased")

    # Make sure config can carry custom label sizes
    try:
        config.num_labels = max(len(polarity_map), len(emotion_map), len(tone_map))
    except Exception:
        pass

    model = None

    # First try HuggingFace-style loading
    try:
        model = model_class.from_pretrained(str(MODEL_DIR), config=config)
        print("[INFO] Loaded model using from_pretrained().")
    except Exception as e1:
        print(f"[INFO] from_pretrained() failed: {e1}")

        # Fallback: manual init with config
        try:
            model = model_class(config)
            print("[INFO] Loaded model using model_class(config).")
        except Exception as e2:
            raise TypeError(
                f"Could not initialize the model.\n"
                f"from_pretrained() error: {e1}\n"
                f"config init error: {e2}"
            )

        # Then load weights manually
        checkpoint = None
        for name in ["best_model.pt", "best_model.pth", "model.pt", "model.pth", "pytorch_model.bin"]:
            candidate = MODEL_DIR / name
            if candidate.exists():
                checkpoint = candidate
                break

        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint file found in {MODEL_DIR}. Expected .pt, .pth, or pytorch_model.bin"
            )

        ckpt = torch.load(checkpoint, map_location=DEVICE)
        state_dict = _resolve_state_dict(ckpt)
        state_dict = _strip_module_prefix(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Manual weight load done. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, polarity_map, emotion_map, tone_map


def predict(text, model, tokenizer, polarity_map, emotion_map, tone_map):
    model_text, special_tokens, cleaned = build_model_text(text)

    encoding = tokenizer(
        model_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            outputs = model(input_ids, attention_mask)

    polarity_logits, emotion_logits, tone_logits, intensity = _extract_outputs(outputs)

    if polarity_logits is None or emotion_logits is None or tone_logits is None:
        raise RuntimeError("Could not extract logits from model output. Check the output structure in src/model.py")

    polarity_probs = torch.softmax(polarity_logits, dim=-1)
    emotion_probs = torch.softmax(emotion_logits, dim=-1)
    tone_probs = torch.softmax(tone_logits, dim=-1)

    polarity_idx = int(torch.argmax(polarity_probs, dim=-1).item())
    emotion_idx = int(torch.argmax(emotion_probs, dim=-1).item())
    tone_idx = int(torch.argmax(tone_probs, dim=-1).item())

    polarity_conf = float(torch.max(polarity_probs).item())
    emotion_conf = float(torch.max(emotion_probs).item())
    tone_conf = float(torch.max(tone_probs).item())

    if isinstance(intensity, torch.Tensor):
        intensity_value = float(intensity.squeeze().item())
    elif intensity is None:
        intensity_value = None
    else:
        intensity_value = float(intensity)

    confidence = round((polarity_conf + emotion_conf + tone_conf) / 3, 3)
    explanation = build_explanation(text, model_text)

    return {
        "input_text": text,
        "model_text": model_text,
        "special_tokens": special_tokens,
        "polarity": polarity_map.get(polarity_idx, str(polarity_idx)),
        "emotion": emotion_map.get(emotion_idx, str(emotion_idx)),
        "tone": tone_map.get(tone_idx, str(tone_idx)),
        "intensity": None if intensity_value is None else round(intensity_value, 3),
        "confidence": confidence,
        "explanation": explanation
    }


def main():
    print("\n=== Multi-task Sentiment Inference ===")
    print("Type a sentence and press Enter.")
    print("Type 'exit' to stop.\n")

    model, tokenizer, polarity_map, emotion_map, tone_map = _load_model()

    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("Please enter some text.\n")
            continue

        try:
            result = predict(text, model, tokenizer, polarity_map, emotion_map, tone_map)

            print("\n--- Prediction ---")
            print(f"Polarity   : {result['polarity']}")
            print(f"Emotion    : {result['emotion']}")
            print(f"Tone       : {result['tone']}")
            print(f"Intensity  : {result['intensity']}")
            print(f"Confidence : {result['confidence']}")
            print("Explanation: " + (", ".join(result["explanation"]) if result["explanation"] else "No strong cues detected"))
            print()
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()