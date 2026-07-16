from __future__ import annotations

import io
import json
import time
import base64
import tempfile
import traceback
import math
from pathlib import Path
import importlib.util
import shutil
import urllib.request

import gradio as gr
from PIL import Image

from modules import scripts, shared, script_callbacks

try:
    from modules.paths import models_path as forge_models_path
except Exception:
    forge_models_path = None


# -----------------------------
# Safe import of taggers_core.py (no sys.path)
# -----------------------------
_CORE_PATH = Path(__file__).resolve().parent / "taggers_core.py"
_spec = importlib.util.spec_from_file_location("tagger_prompt_taggers_core", str(_CORE_PATH))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to create module spec for taggers_core.py")
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

WD14Tagger = _core.WD14Tagger
WDSwinV2V3Tagger = _core.WDSwinV2V3Tagger
WDViTV3Tagger = _core.WDViTV3Tagger
WDEVAV3Tagger = _core.WDEVAV3Tagger
WDConvV3Tagger = _core.WDConvV3Tagger
DeepDanbooruTagger = _core.DeepDanbooruTagger
E621Tagger = _core.E621Tagger


# -----------------------------
# Settings
# -----------------------------
def _norm_dir(p: str) -> str:
    return (p or "").strip().strip('"').strip("'")


def _get_models_dir() -> str:
    try:
        return _norm_dir(shared.opts.data.get("tagger_prompt_models_dir", "") or "")
    except Exception:
        return ""


def _get_default_negative_words() -> str:
    try:
        value = shared.opts.data.get("tagger_prompt_default_negative_words", _DEFAULT_NEGATIVE_WORDS) or ""
    except Exception:
        value = _DEFAULT_NEGATIVE_WORDS
    return str(value)


_MODEL_DOWNLOADS = {
    "wd14": {
        "folder": "wd14",
        "files": {
            "model.onnx": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/selected_tags.csv?download=true",
        },
    },
    "wd3": {
        "folder": "wd_swinv2_v3",
        "files": {
            "model.onnx": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv?download=true",
        },
    },
    "wd_vit_v3": {
        "folder": "wd_vit_v3",
        "files": {
            "model.onnx": "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv?download=true",
        },
    },
    "wd_eva_v3": {
        "folder": "wd_eva_v3",
        "files": {
            "model.onnx": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv?download=true",
        },
    },
    "wd_conv_v3": {
        "folder": "wd_conv_v3",
        "files": {
            "model.onnx": "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/selected_tags.csv?download=true",
        },
    },
    "ddb": {
        "folder": "deepdanbooru",
        "files": {
            "model.onnx": "https://huggingface.co/chinoll/deepdanbooru/resolve/main/deepdanbooru.onnx?download=true",
            "tags.txt": "https://huggingface.co/chinoll/deepdanbooru/resolve/main/tags.txt?download=true",
        },
    },
    "e621": {
        "folder": "e621",
        "files": {
            "model.onnx": "https://huggingface.co/silveroxides/Z3D-E621-Convnext/resolve/main/model.onnx?download=true",
            "selected_tags.csv": "https://huggingface.co/silveroxides/Z3D-E621-Convnext/resolve/main/selected_tags.csv?download=true",
        },
    },
}

_DEFAULT_NEGATIVE_WORDS = "logo, watermark, patreon logo, patreon username, artist name, web address"
_PROMPT_EXPANSION_FOLDER = "prompt_expansion/fooocus_expansion"
_PROMPT_EXPANSION_FILES = {
    "config.json": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/config.json",
    "merges.txt": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/merges.txt",
    "special_tokens_map.json": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/special_tokens_map.json",
    "tokenizer_config.json": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/tokenizer_config.json",
    "vocab.json": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/vocab.json",
    "positive.txt": "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/positive.txt",
    "pytorch_model.bin": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
}
_NEG_INF = -8192.0
_SEED_LIMIT_NUMPY = 2**32
_prompt_enhancer_singleton = None


def on_ui_settings():
    shared.opts.add_option(
        "tagger_prompt_models_dir",
        shared.OptionInfo(
            "",
            "Tagger models directory (WD14 / WD3 / WD ViT v3 / WD EVA v3 / WD Conv v3 / DDB / E621)",
            section=("tagger_prompt", "Tagger Prompt"),
        ),
    )
    shared.opts.add_option(
        "tagger_prompt_default_negative_words",
        shared.OptionInfo(
            _DEFAULT_NEGATIVE_WORDS,
            "Default Negative Words",
            section=("tagger_prompt", "Tagger Prompt"),
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


# -----------------------------
# Helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _default_models_dir() -> Path:
    if forge_models_path:
        return Path(forge_models_path) / "taggers_prompt_models"
    return Path.cwd() / "models" / "taggers_prompt_models"


def _resolve_models_dir() -> tuple[str, bool]:
    configured_dir = _get_models_dir()
    if configured_dir:
        return configured_dir, False
    return str(_default_models_dir()), True


def _download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as f:
        shutil.copyfileobj(resp, f)

    tmp_path.replace(dst)


def _ensure_model_available(tagger_key: str, models_dir: str):
    spec = _MODEL_DOWNLOADS.get(tagger_key)
    if spec is None:
        raise ValueError(f"Unknown tagger: {tagger_key}")

    model_dir = Path(models_dir) / spec["folder"]
    missing = [name for name in spec["files"] if not (model_dir / name).exists()]
    if not missing:
        return

    for name in missing:
        _download_file(spec["files"][name], model_dir / name)


def _safe_str(x) -> str:
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


def _ensure_prompt_expansion_assets(models_dir: str) -> tuple[Path, list[str]]:
    root = Path(models_dir) / _PROMPT_EXPANSION_FOLDER
    missing = [name for name in _PROMPT_EXPANSION_FILES if not (root / name).exists()]
    if not missing:
        return root, []

    downloaded: list[str] = []
    print(f"[Fooocus V2] Missing assets: {len(missing)}. Downloading to: {root}")
    for name in missing:
        print(f"[Fooocus V2] Downloading: {name}")
        _download_file(_PROMPT_EXPANSION_FILES[name], root / name)
        downloaded.append(name)
        print(f"[Fooocus V2] Downloaded: {name}")
    print("[Fooocus V2] Asset download complete.")
    return root, downloaded


class _FooocusPromptEnhancer:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.logits_bias = None
        self._loaded = False

    def ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir))
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        positive_words = (self.model_dir / "positive.txt").read_text(encoding="utf-8").splitlines()
        positive_words = {"Ġ" + w.lower() for w in positive_words if w}

        logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + _NEG_INF
        for token, token_id in self.tokenizer.vocab.items():
            if token in positive_words:
                logits_bias[0, token_id] = 0

        self.logits_bias = logits_bias
        self._loaded = True

    def _logits_processor(self, input_ids, scores):
        import torch

        bias = self.logits_bias.to(scores).clone()
        bias[0, input_ids[0].to(bias.device).long()] = _NEG_INF
        # Allow comma token, same as Fooocus expansion.py
        bias[0, 11] = 0
        return scores + bias

    def expand(self, prompt: str, seed: int) -> str:
        self.ensure_loaded()

        import torch
        from transformers import set_seed
        from transformers.generation.logits_process import LogitsProcessorList

        prompt = _safe_str(prompt)
        if not prompt:
            return ""

        seed = int(seed) % _SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = prompt + ","

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        tokenized.data["input_ids"] = tokenized.data["input_ids"].to(device)
        tokenized.data["attention_mask"] = tokenized.data["attention_mask"].to(device)

        current_token_length = int(tokenized.data["input_ids"].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length
        if max_new_tokens <= 0:
            return prompt[:-1]

        with torch.no_grad():
            features = self.model.generate(
                **tokenized,
                top_k=100,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self._logits_processor]),
            )
        decoded = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        return _safe_str(decoded[0])


def _get_prompt_enhancer(models_dir: str) -> _FooocusPromptEnhancer:
    global _prompt_enhancer_singleton

    model_dir, _ = _ensure_prompt_expansion_assets(models_dir)
    if _prompt_enhancer_singleton is None:
        _prompt_enhancer_singleton = _FooocusPromptEnhancer(model_dir)
    return _prompt_enhancer_singleton


def _maybe_enhance_prompt(prompt_text: str, models_dir: str) -> str:
    prompt_text = _safe_str(prompt_text)
    if not prompt_text:
        return ""

    enhancer = _get_prompt_enhancer(models_dir)
    seed = _now_ms()
    return enhancer.expand(prompt_text, seed=seed)


def _sync_negative_words_in_ui_config(default_negative_words: str):
    try:
        cfg_path = getattr(getattr(shared, "cmd_opts", None), "ui_config_file", None)
        if not cfg_path:
            return
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            return

        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return

        changed = False
        for tab in ("txt2img", "img2img"):
            key = f"{tab}/Negative words/value"
            if data.get(key) != default_negative_words:
                data[key] = default_negative_words
                changed = True

        if changed:
            cfg_file.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    except Exception:
        pass


def _build_tagger(tagger_key: str, models_dir: str):
    if not models_dir:
        raise RuntimeError("Models directory is not set. Go to Settings → Tagger Prompt and set it.")

    if tagger_key == "wd14":
        return WD14Tagger(models_dir)
    if tagger_key == "wd3":
        return WDSwinV2V3Tagger(models_dir)
    if tagger_key == "wd_vit_v3":
        return WDViTV3Tagger(models_dir)
    if tagger_key == "wd_eva_v3":
        return WDEVAV3Tagger(models_dir)
    if tagger_key == "wd_conv_v3":
        return WDConvV3Tagger(models_dir)
    if tagger_key == "ddb":
        return DeepDanbooruTagger(models_dir)
    if tagger_key == "e621":
        return E621Tagger(models_dir)

    raise ValueError(f"Unknown tagger: {tagger_key}")


def _save_pil_to_temp(pil_img: Image.Image) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "forge_tagger_prompt"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"tagger_prompt_{_now_ms()}.jpg"
    pil_img.save(path, format="JPEG", quality=95)
    return str(path)


def _parse_negative_words(negative_words: str) -> set[str]:
    return {
        " ".join(word.strip().lower().split())
        for word in (negative_words or "").split(",")
        if word.strip()
    }


def _filter_tags_text(tags_text: str, negative_words: str) -> str:
    blocked = _parse_negative_words(negative_words)
    if not blocked:
        return tags_text or ""

    tags = [tag.strip() for tag in (tags_text or "").split(",")]
    filtered = []
    for tag in tags:
        if not tag:
            continue
        normalized = " ".join(tag.lower().split())
        if normalized in blocked:
            continue
        filtered.append(tag)
    return ", ".join(filtered)


def _split_tags_csv(text: str) -> list[str]:
    return [tag.strip() for tag in (text or "").split(",") if tag.strip()]


def _normalize_tag_for_compare(tag: str) -> str:
    return " ".join((tag or "").lower().split())


def _apply_enhancement_strength(original_text: str, enhanced_text: str, strength: float) -> str:
    original_tags = _split_tags_csv(original_text)
    enhanced_tags = _split_tags_csv(enhanced_text)

    if not enhanced_tags:
        return ", ".join(original_tags)

    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return ", ".join(original_tags)
    if strength >= 1.0:
        return ", ".join(enhanced_tags)

    original_norm = {_normalize_tag_for_compare(t) for t in original_tags}
    extras = [t for t in enhanced_tags if _normalize_tag_for_compare(t) not in original_norm]
    if not extras:
        return ", ".join(original_tags)

    keep_count = int(round(len(extras) * strength))
    if keep_count <= 0:
        return ", ".join(original_tags)

    merged = original_tags + extras[:keep_count]
    return ", ".join(merged)


def _run_tagger_on_pil(
    tagger_key: str, pil_img: Image.Image, gen_th: float, char_th: float, negative_words: str
) -> str:
    models_dir, should_autodownload = _resolve_models_dir()
    if should_autodownload:
        _ensure_model_available(tagger_key, models_dir)
    tagger = _build_tagger(tagger_key, models_dir)

    if hasattr(tagger, "ensure_loaded"):
        tagger.ensure_loaded()

    if not hasattr(tagger, "predict"):
        raise RuntimeError("Tagger object has no predict() method.")

    img_path = _save_pil_to_temp(pil_img)

    # WD-style taggers / E621 support both general + character thresholds.
    # DeepDanbooru uses only general threshold.
    if tagger_key == "ddb":
        out = tagger.predict(img_path, float(gen_th))
    else:
        use_char = True
        out = tagger.predict(img_path, float(gen_th), use_char, float(char_th))

    if out is None:
        return ""
    if isinstance(out, str):
        return _filter_tags_text(out.strip(), negative_words)
    if isinstance(out, (list, tuple)):
        tags_text = ", ".join([str(x).strip() for x in out if str(x).strip()])
        return _filter_tags_text(tags_text, negative_words)
    if isinstance(out, dict):
        try:
            items = sorted(out.items(), key=lambda kv: float(kv[1]), reverse=True)
            tags_text = ", ".join([str(k).strip() for k, _ in items if str(k).strip()])
            return _filter_tags_text(tags_text, negative_words)
        except Exception:
            tags_text = ", ".join([str(k).strip() for k in out.keys() if str(k).strip()])
            return _filter_tags_text(tags_text, negative_words)
    return _filter_tags_text(str(out).strip(), negative_words)


# -----------------------------
# Script
# -----------------------------
class Script(scripts.Script):
    def title(self):
        return "Tagger Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        default_negative_words = _get_default_negative_words()
        _sync_negative_words_in_ui_config(default_negative_words)

        # unique suffix per tab to avoid ID collisions in DOM
        ui_suffix = "i2i" if is_img2img else "t2i"

        # ids
        eid_tagger_row = f"tp_tagger_row_{ui_suffix}"
        eid_main_layout = f"tp_main_layout_{ui_suffix}"
        eid_controls_col = f"tp_controls_col_{ui_suffix}"
        eid_image_col = f"tp_image_col_{ui_suffix}"
        eid_upload_bar = f"tp_upload_bar_{ui_suffix}"
        eid_sliders_row = f"tp_sliders_row_{ui_suffix}"
        eid_drop = f"tp_drop_{ui_suffix}"
        eid_preview = f"tp_preview_{ui_suffix}"
        eid_status = f"tp_status_{ui_suffix}"
        eid_paste_btn = f"tp_paste_btn_{ui_suffix}"
        eid_paste_pipe = f"tp_paste_pipe_{ui_suffix}"

        selected_tagger = gr.State("wd14")
        image_state = gr.State(None)  # PIL.Image or None

        with gr.Accordion("Tagger Prompt", open=False):
            gr.HTML(
                f"""
<style>
  #{eid_tagger_row}{{display:flex;gap:6px;width:100%;}}
  #{eid_tagger_row} > *{{min-width:min(116px, 100%) !important;}}
  #{eid_tagger_row} .gr-button{{flex:1 1 0;min-width:min(116px, 100%) !important;padding-left:clamp(8px, 1vw, 14px) !important;padding-right:clamp(8px, 1vw, 14px) !important;}}
  #{eid_tagger_row} .gr-button,
  #{eid_tagger_row} button,
  #{eid_upload_bar} .gr-button,
  #{eid_upload_bar} button{{border-radius:8px !important;}}

  /* make "primary" not screaming, just slightly lighter */
  #{eid_tagger_row} .gr-button.primary{{
    background: linear-gradient(180deg, #5e6a7a, #4b5563) !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
  }}

  #{eid_upload_bar}{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;align-items:stretch}}
  #{eid_upload_bar} .gr-button{{width:100%}}

  #{eid_main_layout}{{
    display:grid !important;
    grid-template-columns:minmax(0, 1fr) minmax(0, 1fr);
    gap:12px;
    align-items:start;
  }}
  #{eid_main_layout} > *{{min-width:0 !important;}}
  #{eid_controls_col}, #{eid_image_col}{{min-width:0 !important;}}

  #{eid_drop}{{position:relative;margin-top:0;min-height:84px !important;height:84px !important;overflow:hidden;}}
  #{eid_drop} .wrap, #{eid_drop} .file-wrap, #{eid_drop} .border, #{eid_drop} .container{{
    height:100% !important;min-height:100% !important;padding:0 !important;
    background:transparent !important;border:none !important;
  }}
  #{eid_drop} label, #{eid_drop} .label, #{eid_drop} .upload-text, #{eid_drop} .filetype{{display:none!important;}}
  #{eid_drop}::after{{
      content:'Drag an image here or click to choose one, or use "Paste from clipboard"';
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      padding:0 14px; font-size:13.5px; font-weight:600; opacity:.95;
      border:1.5px dashed var(--block-border-color); border-radius:8px;
      background:var(--body-background-fill); text-align:center;
      pointer-events:none; z-index:2;
  }}

  #{eid_preview} {{ height: 180px !important; max-height:180px !important; overflow:hidden !important; }}
  #{eid_preview} img {{ height: 170px !important; max-height:170px !important; width:100% !important; object-fit:contain !important; }}
  #{eid_status} {{ min-height: 1.2em; }}

  #{eid_sliders_row} {{ margin-top: 6px; }}

  @media (max-width: 900px){{
    #{eid_main_layout}{{grid-template-columns:1fr;}}
  }}
</style>

<script>
(function(){{
  // Make dropzone always clickable (open picker) - per tab instance
  function setupDrop(){{
    const drop = document.querySelector('#{eid_drop}');
    if(!drop) return false;
    drop.addEventListener('click', () => {{
      const input = drop.querySelector('input[type="file"]');
      if (input) input.click();
    }});
    return true;
  }}
  let tries = 0;
  const t = setInterval(() => {{ if (setupDrop() || ++tries > 120) clearInterval(t); }}, 100);
}})();
</script>
"""
            )

            with gr.Row(elem_id=eid_tagger_row):
                btn_wd14 = gr.Button("WD14", variant="primary")
                btn_wd3 = gr.Button("WD3", variant="secondary")
                btn_wd_vit_v3 = gr.Button("WD ViT v3", variant="secondary")
                btn_wd_eva_v3 = gr.Button("WD EVA v3", variant="secondary")
                btn_wd_conv_v3 = gr.Button("WD Conv v3", variant="secondary")
                btn_ddb = gr.Button("DDB", variant="secondary")
                btn_e621 = gr.Button("E621", variant="secondary")

            with gr.Row(elem_id=eid_main_layout):
                with gr.Column(elem_id=eid_controls_col, scale=1):
                    with gr.Row(elem_id=eid_upload_bar):
                        paste_btn = gr.Button("Paste from clipboard", elem_id=eid_paste_btn)
                        remove_btn = gr.Button("Remove")

                    # Threshold sliders (no custom styles)
                    with gr.Row(elem_id=eid_sliders_row):
                        gen_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.35, label="Gen")
                        char_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.90, label="Char")
                    with gr.Row():
                        prompt_enhance_enabled = gr.Checkbox(
                            label="Fooocus V2 Prompt Enhancement",
                            value=False,
                            scale=1,
                        )
                        enhance_strength = gr.Slider(
                            0.0,
                            1.0,
                            step=0.05,
                            value=1.0,
                            label="Strength",
                            scale=1,
                        )
                    negative_words = gr.Textbox(
                        label="Negative words",
                        lines=2,
                        value=default_negative_words,
                        placeholder="Comma-separated words/phrases to exclude from tags",
                    )

                    # IMPORTANT: unique per tab
                    paste_pipe = gr.Textbox(visible=False, elem_id=eid_paste_pipe)

                with gr.Column(elem_id=eid_image_col, scale=1):
                    drop_zone = gr.File(
                        label="",
                        show_label=False,
                        file_types=["image"],
                        file_count="single",
                        elem_id=eid_drop,
                    )

                    preview = gr.Image(label="Preview", type="pil", height=170, elem_id=eid_preview, interactive=False)

            # Paste: write into the tab-local hidden pipe and dispatch events
            paste_btn.click(
                fn=None,
                inputs=[],
                outputs=[],
                _js=f"""
                async () => {{
                  const root = (window.gradioApp ? gradioApp() : document);
                  const pipe = root.querySelector('#{eid_paste_pipe} textarea');
                  if (!pipe) return;

                  if (!(navigator.clipboard && navigator.clipboard.read)) {{
                    pipe.value = "";
                    pipe.dispatchEvent(new Event('input', {{ bubbles:true }}));
                    pipe.dispatchEvent(new Event('change', {{ bubbles:true }}));
                    return;
                  }}
                  try{{
                    const items = await navigator.clipboard.read();
                    for (const item of items){{
                      for (const type of item.types){{
                        if (type.startsWith('image/')){{
                          const blob = await item.getType(type);
                          const dataUrl = await new Promise(res=>{{
                            const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(blob);
                          }});
                          pipe.value = JSON.stringify([dataUrl]);
                          pipe.dispatchEvent(new Event('input', {{ bubbles:true }}));
                          pipe.dispatchEvent(new Event('change', {{ bubbles:true }}));
                          return;
                        }}
                      }}
                    }}
                    pipe.value = "";
                    pipe.dispatchEvent(new Event('input', {{ bubbles:true }}));
                    pipe.dispatchEvent(new Event('change', {{ bubbles:true }}));
                  }}catch(e){{
                    console.warn(e);
                    pipe.value = "";
                    pipe.dispatchEvent(new Event('input', {{ bubbles:true }}));
                    pipe.dispatchEvent(new Event('change', {{ bubbles:true }}));
                  }}
                }}
                """,
            )

            out_tags = gr.Textbox(label="Tags / Prompt", lines=4)
            send_btn = gr.Button("Insert into Prompt")
            status = gr.Markdown("Ready to work. Insert an image.", elem_id=eid_status, visible=False)

            # Insert ONLY into active tab prompt + scroll to that prompt
            js_replace_prompt_and_scroll = r"""
                (tags) => {
                    const val = (tags ?? "").toString();
                    const t2i = document.querySelector('#txt2img_prompt textarea');
                    const i2i = document.querySelector('#img2img_prompt textarea');

                    const isVisible = (el) => {
                      if (!el) return false;
                      const r = el.getClientRects();
                      if (!r || r.length === 0) return false;
                      const cs = window.getComputedStyle(el);
                      if (!cs) return false;
                      if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
                      let p = el;
                      while (p) {
                        if (p === document.body) break;
                        const ps = window.getComputedStyle(p);
                        if (ps && ps.display === 'none') return false;
                        p = p.parentElement;
                      }
                      return true;
                    };

                    const ae = document.activeElement;
                    const focusedIsT2i = (t2i && (ae === t2i || (ae && ae.closest && ae.closest('#txt2img_prompt'))));
                    const focusedIsI2i = (i2i && (ae === i2i || (ae && ae.closest && ae.closest('#img2img_prompt'))));

                    const tabIsVisible = (id) => {
                      const el = document.querySelector(id);
                      return isVisible(el);
                    };

                    let target = null;
                    if (focusedIsT2i) target = t2i;
                    else if (focusedIsI2i) target = i2i;
                    else if (typeof get_uiCurrentTabContent === 'function') {
                      try {
                        const currentTab = get_uiCurrentTabContent();
                        if (currentTab && currentTab.id === 'tab_img2img') target = i2i;
                        else if (currentTab && currentTab.id === 'tab_txt2img') target = t2i;
                      } catch(e) {}
                    }
                    if (!target && typeof activePromptTextarea === 'object') {
                      try {
                        if (activePromptTextarea.img2img === i2i) target = i2i;
                        else if (activePromptTextarea.txt2img === t2i) target = t2i;
                      } catch(e) {}
                    }
                    if (!target) {
                      if (tabIsVisible('#tab_img2img')) target = i2i;
                      else if (tabIsVisible('#tab_txt2img')) target = t2i;
                      else if (isVisible(t2i)) target = t2i;
                      else if (isVisible(i2i)) target = i2i;
                      else target = t2i || i2i;
                    }

                    if (!target) return "Could not find the active Prompt field.";

                    target.value = val;
                    if (typeof updateInput === 'function') {
                      try { updateInput(target); } catch(e) {
                        target.dispatchEvent(new Event('input', { bubbles: true }));
                      }
                    } else {
                      target.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    target.dispatchEvent(new Event('change', { bubbles: true }));

                    const isImg2Img = target === i2i;
                    const promptBlock = target.closest('#txt2img_prompt, #img2img_prompt') || target;
                    const phystonBlock = document.querySelector(isImg2Img ? '#phystonPrompt_img2img_prompt' : '#phystonPrompt_txt2img_prompt');
                    const scrollTarget = isVisible(promptBlock) ? promptBlock : (isVisible(phystonBlock) ? phystonBlock : null);
                    try {
                      if (scrollTarget) scrollTarget.scrollIntoView({ behavior: 'smooth', block: 'center' });
                      else window.scrollTo({ top: 0, behavior: 'smooth' });
                    } catch(e){
                      try { window.scrollTo({ top: 0, behavior: 'smooth' }); } catch(e2){}
                    }
                    if (isVisible(target)) {
                      setTimeout(() => {
                        try { target.focus({ preventScroll: true }); } catch(e){}
                      }, 700);
                    }
                    return "";
                }
            """

            def set_from_file(file_obj, current_pil):
                if not file_obj:
                    if current_pil is None:
                        return None, None, "No image selected."
                    return current_pil, current_pil, "Done."

                try:
                    pil = Image.open(file_obj.name).convert("RGB")
                    return pil, pil, "Image loaded."
                except Exception as e:
                    if current_pil is None:
                        return None, None, f"Could not open image: {e}"
                    return current_pil, current_pil, f"Could not open image: {e}"

            def set_from_paste(payload_json):
                if not payload_json:
                    return None, None, "Clipboard is empty (or the browser denied access)."
                try:
                    arr = json.loads(payload_json or "[]")
                except Exception:
                    arr = []
                if not arr:
                    return None, None, "Clipboard is empty (no image found)."
                try:
                    data_url = arr[0]
                    comma = data_url.find(",")
                    b64 = data_url[comma + 1 :] if comma != -1 else data_url
                    raw = base64.b64decode(b64)
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    return pil, pil, "Image pasted from clipboard."
                except Exception as e:
                    return None, None, f"Could not read clipboard: {e}"

            def autotag(pil_img, tagger_key, gen_th, char_th, negative_words_text, enhance_prompt, enhance_strength_value):
                if pil_img is None:
                    return "", "No image."
                try:
                    models_dir, _ = _resolve_models_dir()
                    tags = _run_tagger_on_pil(
                        tagger_key,
                        pil_img,
                        float(gen_th),
                        float(char_th),
                        negative_words_text,
                    )
                    if enhance_prompt:
                        try:
                            _, downloaded_files = _ensure_prompt_expansion_assets(models_dir)
                            enhanced = _maybe_enhance_prompt(tags, models_dir)
                            tags = _apply_enhancement_strength(tags, enhanced, float(enhance_strength_value))
                            if downloaded_files:
                                return tags, f"Done. Fooocus V2 downloaded {len(downloaded_files)} file(s)."
                            return tags, "Done. Fooocus V2 model cache is ready."
                        except Exception as e:
                            return tags, f"Tags done. Prompt enhancement skipped: {e}"
                    return tags, "Done."
                except Exception as e:
                    tb = traceback.format_exc()
                    return "", f"Tagger error: {e}\n\n{tb}"

            def set_from_file_and_autotag(
                file_obj,
                current_pil,
                tagger_key,
                gen_th,
                char_th,
                negative_words_text,
                enhance_prompt,
                enhance_strength_value,
            ):
                # Clearing the file component emits one more change event. Ignore
                # that event so it only restores the dropzone, without retagging.
                if not file_obj:
                    return current_pil, gr.update(), gr.update(), gr.update(), gr.update()

                pil, preview_value, load_status = set_from_file(file_obj, current_pil)
                if pil is current_pil:
                    return current_pil, preview_value, load_status, gr.update(), gr.update(value=None)

                tags, tag_status = autotag(
                    pil,
                    tagger_key,
                    gen_th,
                    char_th,
                    negative_words_text,
                    enhance_prompt,
                    enhance_strength_value,
                )
                return pil, preview_value, tag_status, tags, gr.update(value=None)

            def remove_image():
                return None, None, "Image removed.", "", gr.update(value=None)

            def select_tagger(key: str):
                return (
                    key,
                    gr.update(variant="primary" if key == "wd14" else "secondary"),
                    gr.update(variant="primary" if key == "wd3" else "secondary"),
                    gr.update(variant="primary" if key == "wd_vit_v3" else "secondary"),
                    gr.update(variant="primary" if key == "wd_eva_v3" else "secondary"),
                    gr.update(variant="primary" if key == "wd_conv_v3" else "secondary"),
                    gr.update(variant="primary" if key == "ddb" else "secondary"),
                    gr.update(variant="primary" if key == "e621" else "secondary"),
                )

            drop_zone.change(
                fn=set_from_file_and_autotag,
                inputs=[
                    drop_zone,
                    image_state,
                    selected_tagger,
                    gen_slider,
                    char_slider,
                    negative_words,
                    prompt_enhance_enabled,
                    enhance_strength,
                ],
                outputs=[image_state, preview, status, out_tags, drop_zone],
            )

            paste_pipe.change(
                fn=set_from_paste,
                inputs=[paste_pipe],
                outputs=[image_state, preview, status],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            remove_btn.click(
                fn=remove_image,
                inputs=[],
                outputs=[image_state, preview, status, out_tags, drop_zone],
            )

            btn_wd14.click(
                fn=lambda: select_tagger("wd14"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_wd3.click(
                fn=lambda: select_tagger("wd3"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_wd_vit_v3.click(
                fn=lambda: select_tagger("wd_vit_v3"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_wd_eva_v3.click(
                fn=lambda: select_tagger("wd_eva_v3"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_wd_conv_v3.click(
                fn=lambda: select_tagger("wd_conv_v3"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_ddb.click(
                fn=lambda: select_tagger("ddb"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            btn_e621.click(
                fn=lambda: select_tagger("e621"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_wd_vit_v3, btn_wd_eva_v3, btn_wd_conv_v3, btn_ddb, btn_e621],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            gen_slider.change(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )
            char_slider.change(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )
            negative_words.change(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )
            prompt_enhance_enabled.change(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )
            enhance_strength.change(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider, negative_words, prompt_enhance_enabled, enhance_strength],
                outputs=[out_tags, status],
            )

            send_btn.click(
                fn=None,
                inputs=[out_tags],
                outputs=[],
                js=js_replace_prompt_and_scroll,
            )

        return [out_tags]

