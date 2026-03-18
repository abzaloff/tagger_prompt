from __future__ import annotations

import io
import json
import time
import base64
import tempfile
import traceback
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


def on_ui_settings():
    shared.opts.add_option(
        "tagger_prompt_models_dir",
        shared.OptionInfo(
            "",
            "Tagger models directory (WD14 / WD3 / DDB / E621)",
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


def _model_missing_for_tagger(tagger_key: str, models_dir: str) -> bool:
    spec = _MODEL_DOWNLOADS.get(tagger_key)
    if spec is None:
        return False
    model_dir = Path(models_dir) / spec["folder"]
    return any(not (model_dir / name).exists() for name in spec["files"])


def _tagger_label(tagger_key: str) -> str:
    labels = {
        "wd14": "WD14",
        "wd3": "WD3",
        "ddb": "DDB",
        "e621": "E621",
    }
    return labels.get(tagger_key, tagger_key.upper())


def _build_tagger(tagger_key: str, models_dir: str):
    if not models_dir:
        raise RuntimeError("Models directory is not set. Go to Settings → Tagger Prompt and set it.")

    if tagger_key == "wd14":
        return WD14Tagger(models_dir)
    if tagger_key == "wd3":
        return WDSwinV2V3Tagger(models_dir)
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


def _run_tagger_on_pil(tagger_key: str, pil_img: Image.Image, gen_th: float, char_th: float) -> str:
    models_dir, should_autodownload = _resolve_models_dir()
    if should_autodownload:
        _ensure_model_available(tagger_key, models_dir)
    tagger = _build_tagger(tagger_key, models_dir)

    if hasattr(tagger, "ensure_loaded"):
        tagger.ensure_loaded()

    if not hasattr(tagger, "predict"):
        raise RuntimeError("Tagger object has no predict() method.")

    img_path = _save_pil_to_temp(pil_img)

    # WD14 / WD3 / E621 support both general + character thresholds.
    # DeepDanbooru uses only general threshold.
    if tagger_key == "ddb":
        out = tagger.predict(img_path, float(gen_th))
    else:
        use_char = True
        out = tagger.predict(img_path, float(gen_th), use_char, float(char_th))

    if out is None:
        return ""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, (list, tuple)):
        return ", ".join([str(x).strip() for x in out if str(x).strip()])
    if isinstance(out, dict):
        try:
            items = sorted(out.items(), key=lambda kv: float(kv[1]), reverse=True)
            return ", ".join([str(k).strip() for k, _ in items if str(k).strip()])
        except Exception:
            return ", ".join([str(k).strip() for k in out.keys() if str(k).strip()])
    return str(out).strip()


# -----------------------------
# Script
# -----------------------------
class Script(scripts.Script):
    def title(self):
        return "Tagger Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # unique suffix per tab to avoid ID collisions in DOM
        ui_suffix = "i2i" if is_img2img else "t2i"

        # ids
        eid_tagger_row = f"tp_tagger_row_{ui_suffix}"
        eid_upload_bar = f"tp_upload_bar_{ui_suffix}"
        eid_sliders_row = f"tp_sliders_row_{ui_suffix}"
        eid_drop = f"tp_drop_{ui_suffix}"
        eid_preview = f"tp_preview_{ui_suffix}"
        eid_paste_btn = f"tp_paste_btn_{ui_suffix}"
        eid_paste_pipe = f"tp_paste_pipe_{ui_suffix}"

        selected_tagger = gr.State("wd14")
        image_state = gr.State(None)  # PIL.Image or None

        with gr.Accordion("Tagger Prompt", open=False):
            gr.HTML(
                f"""
<style>
  #{eid_tagger_row}{{display:flex;gap:8px;width:100%;}}
  #{eid_tagger_row} .gr-button{{flex:1 1 0;min-width:0;}}

  /* make "primary" not screaming, just slightly lighter */
  #{eid_tagger_row} .gr-button.primary{{
    background: linear-gradient(180deg, #5e6a7a, #4b5563) !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
  }}

  #{eid_upload_bar}{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;align-items:stretch}}
  #{eid_upload_bar} .gr-button{{width:100%}}

  #{eid_drop}{{position:relative;margin-top:6px;min-height:84px !important;height:84px !important;overflow:hidden;}}
  #{eid_drop} .wrap, #{eid_drop} .file-wrap, #{eid_drop} .border, #{eid_drop} .container{{
    height:100% !important;min-height:100% !important;padding:0 !important;
    background:transparent !important;border:none !important;
  }}
  #{eid_drop} label, #{eid_drop} .label, #{eid_drop} .upload-text, #{eid_drop} .filetype{{display:none!important;}}
  #{eid_drop}::after{{
      content:"Drag an image here or click to choose one, or use “Paste from clipboard”";
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      padding:0 14px; font-size:13.5px; font-weight:600; opacity:.95;
      border:1.5px dashed var(--block-border-color); border-radius:8px;
      background:var(--body-background-fill); text-align:center;
      pointer-events:none; z-index:2;
  }}

  #{eid_preview} {{ height: 180px !important; max-height:180px !important; overflow:hidden !important; }}
  #{eid_preview} img {{ height: 170px !important; max-height:170px !important; width:100% !important; object-fit:contain !important; }}

  #{eid_sliders_row} {{ margin-top: 6px; }}
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
                btn_ddb = gr.Button("DDB", variant="secondary")
                btn_e621 = gr.Button("E621", variant="secondary")

            with gr.Row(elem_id=eid_upload_bar):
                paste_btn = gr.Button("Paste from clipboard", elem_id=eid_paste_btn)
                remove_btn = gr.Button("Remove")

            # Threshold sliders (no custom styles)
            with gr.Row(elem_id=eid_sliders_row):
                gen_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.35, label="Gen")
                char_slider = gr.Slider(0.0, 1.0, step=0.01, value=0.90, label="Char")

            # IMPORTANT: unique per tab
            paste_pipe = gr.Textbox(visible=False, elem_id=eid_paste_pipe)

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

            drop_zone = gr.File(
                label="",
                show_label=False,
                file_types=["image"],
                file_count="single",
                elem_id=eid_drop,
            )

            preview = gr.Image(label="Preview", type="pil", height=170, elem_id=eid_preview, interactive=False)
            status = gr.Markdown("Ready to work. Insert an image.")
            out_tags = gr.Textbox(label="Tags / Prompt", lines=4)
            send_btn = gr.Button("Insert into Prompt")

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

                    let target = null;
                    if (focusedIsT2i) target = t2i;
                    else if (focusedIsI2i) target = i2i;
                    else if (isVisible(t2i) && !isVisible(i2i)) target = t2i;
                    else if (isVisible(i2i) && !isVisible(t2i)) target = i2i;
                    else if (isVisible(t2i)) target = t2i;
                    else if (isVisible(i2i)) target = i2i;

                    if (!target) return "Could not find the active Prompt field.";

                    target.value = val;
                    target.dispatchEvent(new Event('input', { bubbles: true }));

                    try { target.focus(); } catch(e){}
                    try {
                      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    } catch(e){
                      try { window.scrollTo({ top: 0, behavior: 'smooth' }); } catch(e2){}
                    }
                    return "";
                }
            """

            # clear ONLY this tab's file input (avoid touching other tab)
            js_clear_file_frontend = f"""
                () => {{
                  const drop = document.querySelector('#{eid_drop}');
                  if (!drop) return;
                  const input = drop.querySelector('input[type="file"]');
                  if (input) input.value = '';
                }}
            """

            def set_from_file(file_obj, current_pil):
                if not file_obj:
                    if current_pil is None:
                        return None, None, "No image selected.", gr.update(value=None)
                    return current_pil, current_pil, "Done.", gr.update(value=None)

                try:
                    pil = Image.open(file_obj.name).convert("RGB")
                    return pil, pil, "Image loaded.", gr.update(value=None)
                except Exception as e:
                    if current_pil is None:
                        return None, None, f"Could not open image: {e}", gr.update(value=None)
                    return current_pil, current_pil, f"Could not open image: {e}", gr.update(value=None)

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

            def autotag(pil_img, tagger_key, gen_th, char_th):
                if pil_img is None:
                    return "", "No image."
                try:
                    tags = _run_tagger_on_pil(tagger_key, pil_img, float(gen_th), float(char_th))
                    return tags, "Done."
                except Exception as e:
                    tb = traceback.format_exc()
                    return "", f"Tagger error: {e}\n\n{tb}"

            def remove_image():
                return None, None, "Image removed.", "", gr.update(value=None)

            def prepare_autotag_status(pil_img, tagger_key, current_status):
                if pil_img is None:
                    return current_status
                models_dir, should_autodownload = _resolve_models_dir()
                if should_autodownload and _model_missing_for_tagger(tagger_key, models_dir):
                    return f"Downloading {_tagger_label(tagger_key)} model..."
                return current_status

            def select_tagger(key: str):
                return (
                    key,
                    gr.update(variant="primary" if key == "wd14" else "secondary"),
                    gr.update(variant="primary" if key == "wd3" else "secondary"),
                    gr.update(variant="primary" if key == "ddb" else "secondary"),
                    gr.update(variant="primary" if key == "e621" else "secondary"),
                )

            drop_zone.change(
                fn=set_from_file,
                inputs=[drop_zone, image_state],
                outputs=[image_state, preview, status, drop_zone],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider],
                outputs=[out_tags, status],
            ).then(
                fn=None,
                inputs=[],
                outputs=[],
                js=js_clear_file_frontend,
            )

            paste_pipe.change(
                fn=set_from_paste,
                inputs=[paste_pipe],
                outputs=[image_state, preview, status],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider],
                outputs=[out_tags, status],
            )

            remove_btn.click(
                fn=remove_image,
                inputs=[],
                outputs=[image_state, preview, status, out_tags, drop_zone],
            ).then(
                fn=None,
                inputs=[],
                outputs=[],
                js=js_clear_file_frontend,
            )

            btn_wd14.click(
                fn=lambda: select_tagger("wd14"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_ddb, btn_e621],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(fn=autotag, inputs=[image_state, selected_tagger, gen_slider, char_slider], outputs=[out_tags, status])

            btn_wd3.click(
                fn=lambda: select_tagger("wd3"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_ddb, btn_e621],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(fn=autotag, inputs=[image_state, selected_tagger, gen_slider, char_slider], outputs=[out_tags, status])

            btn_ddb.click(
                fn=lambda: select_tagger("ddb"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_ddb, btn_e621],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(fn=autotag, inputs=[image_state, selected_tagger, gen_slider, char_slider], outputs=[out_tags, status])

            btn_e621.click(
                fn=lambda: select_tagger("e621"),
                inputs=[],
                outputs=[selected_tagger, btn_wd14, btn_wd3, btn_ddb, btn_e621],
            ).then(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(fn=autotag, inputs=[image_state, selected_tagger, gen_slider, char_slider], outputs=[out_tags, status])

            gen_slider.change(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider],
                outputs=[out_tags, status],
            )
            char_slider.change(
                fn=prepare_autotag_status,
                inputs=[image_state, selected_tagger, status],
                outputs=[status],
            ).then(
                fn=autotag,
                inputs=[image_state, selected_tagger, gen_slider, char_slider],
                outputs=[out_tags, status],
            )

            send_btn.click(
                fn=None,
                inputs=[out_tags],
                outputs=[],
                js=js_replace_prompt_and_scroll,
            )

        return [out_tags]
