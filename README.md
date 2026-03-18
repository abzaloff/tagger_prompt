# Tagger Prompt
`Tagger Prompt` is a small Forge extension that turns an input image into prompt tags inside the Stable Diffusion UI.

<img width="872" height="750" alt="tagger_prompt" src="https://github.com/user-attachments/assets/b686ef1a-ccf7-48b7-bf03-96e1db6bbc07" />

## What It Does

- Adds a `Tagger Prompt` panel to the Forge interface.
- Lets the user upload an image or paste one from the clipboard.
- Runs the selected tagger model and generates tags.
- Inserts the generated tags into the active prompt field.

## Supported Taggers

- `WD14`
- `WD3` (`WD SwinV2 v3`)
- `DeepDanbooru`
- `E621`

## Project Structure

- [scripts/tagger_prompt.py] contains the Forge/Gradio UI, settings, model auto-download logic, and the main user flow.
- [scripts/taggers_core.py] contains the ONNX-based tagger implementations.

## Model Storage

- If `tagger_prompt_models_dir` is set in Forge settings, the extension uses that directory.
- If it is empty, the extension uses `models/taggers_prompt_models`.
- On first use of a tagger, the extension downloads only the files required for the selected model.

## Typical Flow

1. Open the `Tagger Prompt` accordion in Forge.
2. Upload an image or paste one from the clipboard.
3. Select a tagger.
4. Wait for the tags to be generated.
5. Click `Insert into Prompt`.

## Notes

- Models are loaded through `onnxruntime`.
- WD-style taggers support both general and character thresholds.
- DeepDanbooru uses only the general threshold.
