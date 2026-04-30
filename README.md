# Tagger Prompt
`Tagger Prompt` is a small Forge extension that turns an input image into prompt tags inside the Stable Diffusion UI.

<img width="898" height="867" alt="777" src="https://github.com/user-attachments/assets/a710d8ed-4fc6-4a46-93df-d6ed0e5a930e" />

## What It Does

- Adds a `Tagger Prompt` panel to the Forge interface.
- Lets the user upload an image or paste one from the clipboard.
- Runs the selected tagger model and generates tags.
- Supports filtering unwanted tags with `Negative words`.
- Supports optional `Fooocus V2 Prompt Enhancement` with adjustable `Strength`.
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
- If `Fooocus V2 Prompt Enhancement` is enabled, the extension also auto-downloads prompt-expansion assets on first use.

## Prompt Controls

- `Negative words`: comma-separated words/phrases removed from generated tags.
- `Fooocus V2 Prompt Enhancement`: optional GPT-2 based prompt expansion.
- `Strength` (`0.0` to `1.0`): controls how much enhancement is applied.
  - `0.0` keeps original tagger output.
  - `1.0` applies full enhanced output.
  - Intermediate values blend original tags with added enhancement terms.

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
- `Fooocus V2 Prompt Enhancement` requires `transformers`/`torch` at runtime.
- During first enhancement run, download progress is shown in console with `[Fooocus V2]` logs, and UI status reports whether files were downloaded or loaded from cache.

## License

See [LICENSE.md](LICENSE.md).
