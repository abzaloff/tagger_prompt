# taggers_core.py
# Core taggers for Forge extension "Tagger Prompt"
# Expects models_root to contain subfolders:
#   wd14, wd_swinv2_v3, wd_vit_v3, wd_eva_v3, wd_conv_v3, deepdanbooru, e621

from __future__ import annotations

from pathlib import Path


# ---------------- WD14 ----------------
class WD14Tagger:
    def __init__(self, models_root: str | Path):
        root = Path(models_root)
        self.model_dir = root / "wd14"

        self.session = None
        self.input_name = None
        self.out_name = None
        self.in_h = 448
        self.in_w = 448

        self.tag_names: list[str] = []
        self.rating_idx: list[int] = []
        self.general_idx: list[int] = []
        self.char_idx: list[int] = []

    def ensure_loaded(self):
        if self.session is not None:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(f"WD14 folder not found: {self.model_dir}")

        import csv
        import onnxruntime as ort
        from onnxruntime import InferenceSession

        onnx_path = None
        for name in ["model.onnx", "wd-v1-4-vit-tagger-v2.onnx", "wd-v1-4-swinv2-tagger-v2.onnx"]:
            p = self.model_dir / name
            if p.exists():
                onnx_path = p
                break

        if onnx_path is None:
            files = sorted(self.model_dir.glob("*.onnx"))
            if not files:
                raise FileNotFoundError(f".onnx file not found in {self.model_dir}")
            onnx_path = files[0]

        csv_path = self.model_dir / "selected_tags.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"selected_tags.csv not found in {self.model_dir}")

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in set(ort.get_available_providers()):
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        self.session = InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4:
            _, h, w, _ = shape
            if isinstance(h, int) and isinstance(w, int):
                self.in_h, self.in_w = h, w

        self.tag_names.clear()
        self.rating_idx.clear()
        self.general_idx.clear()
        self.char_idx.clear()

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter=",", quotechar='"')
            for i, row in enumerate(rdr):
                cat = row.get("category", "")
                name = (row.get("name", "") or "").strip()
                if not name:
                    continue
                self.tag_names.append(name)
                if cat == "9":
                    self.rating_idx.append(i)
                elif cat == "0":
                    self.general_idx.append(i)
                elif cat == "4":
                    self.char_idx.append(i)

    def _prep(self, img_path: Path):
        import numpy as np
        from PIL import Image

        im = Image.open(img_path).convert("RGB").resize((self.in_w, self.in_h))
        arr = np.asarray(im, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        arr = arr[None, ...]
        return arr

    def predict(self, img_path: str | Path, gen_th: float = 0.35, use_char: bool = False, char_th: float = 0.90):
        return [t for t in self.predict_scores(img_path, gen_th, use_char, char_th).keys()]

    def predict_scores(
        self, img_path: str | Path, gen_th: float = 0.35, use_char: bool = False, char_th: float = 0.90
    ) -> dict[str, float]:
        self.ensure_loaded()
        p = Path(img_path)
        x = self._prep(p)
        probs = self.session.run(None, {self.input_name: x})[0][0].astype(float)

        out: dict[str, float] = {}
        for i in self.general_idx:
            pr = float(probs[i])
            if pr > gen_th:
                out[self.tag_names[i].replace("_", " ")] = pr

        if use_char:
            for i in self.char_idx:
                pr = float(probs[i])
                if pr > char_th:
                    out[self.tag_names[i].replace("_", " ")] = pr

        return out


# --- WD SwinV2 Tagger v3 (compatible with WD14Tagger by format) ---
class WDSwinV2V3Tagger(WD14Tagger):
    def __init__(self, models_root: str | Path):
        root = Path(models_root)
        self.model_dir = root / "wd_swinv2_v3"

        self.session = None
        self.input_name = None
        self.out_name = None
        self.in_h = 448
        self.in_w = 448

        self.tag_names: list[str] = []
        self.rating_idx: list[int] = []
        self.general_idx: list[int] = []
        self.char_idx: list[int] = []


class WDViTV3Tagger(WD14Tagger):
    def __init__(self, models_root: str | Path):
        super().__init__(models_root)
        self.model_dir = Path(models_root) / "wd_vit_v3"


class WDEVAV3Tagger(WD14Tagger):
    def __init__(self, models_root: str | Path):
        super().__init__(models_root)
        self.model_dir = Path(models_root) / "wd_eva_v3"


class WDConvV3Tagger(WD14Tagger):
    def __init__(self, models_root: str | Path):
        super().__init__(models_root)
        self.model_dir = Path(models_root) / "wd_conv_v3"


# --------------- DeepDanbooru ---------------
class DeepDanbooruTagger:
    def __init__(self, models_root: str | Path):
        root = Path(models_root)
        self.model_dir = root / "deepdanbooru"

        self.session = None
        self.input_name = None
        self.out_name = None
        self.in_h = 512
        self.in_w = 512
        self.tags: list[str] = []

    def ensure_loaded(self):
        if self.session is not None:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(f"DeepDanbooru folder not found: {self.model_dir}")

        import onnxruntime as ort
        from onnxruntime import InferenceSession

        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            files = sorted(self.model_dir.glob("*.onnx"))
            if not files:
                raise FileNotFoundError(f"model.onnx (or any .onnx file) not found in {self.model_dir}")
            onnx_path = files[0]

        tags_path = self.model_dir / "tags.txt"
        if not tags_path.exists():
            raise FileNotFoundError(f"tags.txt not found in {self.model_dir}")

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in set(ort.get_available_providers()):
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        self.session = InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4:
            _, h, w, _ = shape
            if isinstance(h, int) and isinstance(w, int):
                self.in_h, self.in_w = h, w

        self.tags = [
            line.strip()
            for line in tags_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

    def _prep(self, img_path: Path):
        import numpy as np
        from PIL import Image

        im = Image.open(img_path).convert("RGB").resize((self.in_w, self.in_h))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = arr[None, ...]
        return arr

    def predict(self, img_path: str | Path, gen_th: float = 0.5, *_):
        return [t for t in self.predict_scores(img_path, gen_th).keys()]

    def predict_scores(self, img_path: str | Path, gen_th: float = 0.5, *_ignore) -> dict[str, float]:
        self.ensure_loaded()
        p = Path(img_path)
        x = self._prep(p)
        probs = self.session.run(None, {self.input_name: x})[0][0]

        out: dict[str, float] = {}
        for i, pr in enumerate(probs):
            pr = float(pr)
            if pr >= gen_th:
                tag = (self.tags[i] if i < len(self.tags) else f"tag_{i}").replace("_", " ")
                out[tag] = pr
        return out


# ---------------- E621 ----------------
class E621Tagger:
    def __init__(self, models_root: str | Path):
        root = Path(models_root)
        self.model_dir = root / "e621"

        self.session = None
        self.input_name = None
        self.out_name = None
        self.in_h = 448
        self.in_w = 448

        self.tag_names: list[str] = []
        self.rating_idx: list[int] = []
        self.general_idx: list[int] = []
        self.char_idx: list[int] = []

    def ensure_loaded(self):
        if self.session is not None:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(f"E621 folder not found: {self.model_dir}")

        import csv
        import onnxruntime as ort
        from onnxruntime import InferenceSession

        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            files = sorted(self.model_dir.glob("*.onnx"))
            if not files:
                raise FileNotFoundError(f".onnx file not found in {self.model_dir}")
            onnx_path = files[0]

        csv_path = self.model_dir / "selected_tags.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"selected_tags.csv not found in {self.model_dir}")

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in set(ort.get_available_providers()):
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        self.session = InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4:
            _, h, w, _ = shape
            if isinstance(h, int) and isinstance(w, int):
                self.in_h, self.in_w = h, w

        self.tag_names.clear()
        self.rating_idx.clear()
        self.general_idx.clear()
        self.char_idx.clear()

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter=",", quotechar='"')
            for i, row in enumerate(rdr):
                cat = row.get("category", "")
                name = (row.get("name", "") or "").strip()
                if not name:
                    continue
                self.tag_names.append(name)
                if cat == "9":
                    self.rating_idx.append(i)
                elif cat == "0":
                    self.general_idx.append(i)
                elif cat == "4":
                    self.char_idx.append(i)

    def _prep(self, img_path: Path):
        import numpy as np
        from PIL import Image

        im = Image.open(img_path).convert("RGB").resize((self.in_w, self.in_h))
        arr = np.asarray(im, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        arr = arr[None, ...]
        return arr

    def predict(self, img_path: str | Path, gen_th: float = 0.35, use_char: bool = False, char_th: float = 0.90):
        return [t for t in self.predict_scores(img_path, gen_th, use_char, char_th).keys()]

    def predict_scores(
        self, img_path: str | Path, gen_th: float = 0.35, use_char: bool = False, char_th: float = 0.90
    ) -> dict[str, float]:
        self.ensure_loaded()
        p = Path(img_path)
        x = self._prep(p)
        probs = self.session.run(None, {self.input_name: x})[0][0].astype(float)

        out: dict[str, float] = {}
        for i in self.general_idx:
            pr = float(probs[i])
            if pr > gen_th:
                out[self.tag_names[i].replace("_", " ")] = pr

        if use_char:
            for i in self.char_idx:
                pr = float(probs[i])
                if pr > char_th:
                    out[self.tag_names[i].replace("_", " ")] = pr

        return out
