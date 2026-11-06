"""Report conflicting ONNX Runtime distributions in Forge's shared environment."""

from importlib.metadata import PackageNotFoundError, version
import os


def _installed_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    if os.environ.get("TAGGER_PROMPT_SKIP_ORT_INSTALL"):
        return

    cpu = _installed_version("onnxruntime")
    gpu = _installed_version("onnxruntime-gpu")
    if cpu and gpu:
        print(
            f"[Tagger Prompt] WARNING: onnxruntime {cpu} and onnxruntime-gpu {gpu} "
            "share the same Python module and can overwrite each other. "
            "Keep a single ONNX Runtime distribution compatible with your Forge setup."
        )
    elif not cpu and not gpu:
        print("[Tagger Prompt] No ONNX Runtime distribution found in the Forge environment.")


if __name__ == "__main__":
    main()
