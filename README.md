### Macro Placement (AI Image Editing – Layout Planner)

This project is a focused module of a broader AI image editing pipeline. It performs macro placement: given an image and its segmented objects, it recomposes those objects on a newly generated canvas of a different aspect ratio. The system decides which object goes roughly where and in what order (containers/rows/columns), not the pixel-precise “micro” adjustments. It assumes object segmentation is already done, and that micro placement or fine alignment happens in a downstream step.

The core idea is to combine a deterministic compositor with a VLM-guided layout plan expressed in a simple Flex-DSL (row/column containers with justify/align/padding/gap). A Streamlit app wraps the pipeline so you can tweak persona-level design rules and see iteration-by-iteration results.

## Why this matters

- **Creative adaptation at scale**: Re-layout a single visual concept across formats (e.g., 9:16 → 1:1 → 16:9) while keeping brand intent.
- **Separation of concerns**: Macro placement handles grouping/order/rough location; later stages can handle typography, constraints, pixel-accurate nudges, and effects.
- **Prompt engineering surface**: Different “personas” (Initial layout, Critic, Refiner) let you explore instruction styles and context blocks that consistently yield better layouts for your brand or content type.

## What it does (high level)

1) Reads a pre-segmented bundle for an input image: `background.png`, `results.json`, and `objects/` (cutouts).

2) Computes a target canvas size for a chosen aspect ratio while preserving pixel budget, then synthesizes a solid background color based on the original.

3) Builds a labeled contact sheet and a shared context (object roles, normalized boxes, non‑nestable pairs by axis, best‑practice notes by aspect family).

4) VLM stage via Nebius (e.g., Qwen2.5-VL):
- Initial layout persona produces a Flex‑DSL JSON (one container tree) that covers every object exactly once.
- Critic persona reviews the composed draft and flags issues and hard‑rule violations.
- Refiner persona writes a new JSON addressing the critique (optionally with validator feedback when needed).

5) Deterministic compositor renders the placements. The process can iterate a few times (critique → refine → compose) to improve the layout.

### Visual overview

<p align="center" style="text-align:center;">
  <span style="display:inline-block; text-align:center; vertical-align:middle; horizontal-align:middle;">
    <img src="assets/squarespace.jpg" width="270" style="vertical-align:middle; object-fit:contain;" alt="Original input"><br/>
    <sub>Original</sub>
  </span>
</p>

<p align="center" style="font-size:28px; margin: 2px 0;">↓</p>

<p align="center" style="text-align:center;">
  <span style="display:inline-block; text-align:center; vertical-align:middle;">
    <img src="assets/annotated.png" width="270" style="vertical-align:middle; object-fit:contain;" alt="Annotated / segmented (upstream step)"><br/>
    <sub>Segmentation (outside this module)</sub>
  </span>
</p>

<p align="center" style="font-size:28px; margin: 2px 0;">↓</p>

<div align="center">
  <table style="border-collapse:collapse; border:0;">
    <tr style="border:0;">
      <td align="center" style="border:0;">
        <img src="assets/draft_macro_iter_00.png" height="200" alt="Iteration 0 (first draft)"><br/>
        <sub>Iter 0</sub>
      </td>
      <td align="center" style="font-size:28px; padding: 0 12px; border:0;">→</td>
      <td align="center" style="border:0;">
        <img src="assets/draft_macro_iter_01.png" height="200" alt="Iteration 1 (refined)"><br/>
        <sub>Iter 1</sub>
      </td>
      <td align="center" style="font-size:28px; padding: 0 12px;">→</td>
      <td align="center"><em>additional iterations…</em></td>
    </tr>
  </table>
</div>

## Prompting surface in the UI

The Streamlit UI exposes editable design-rule text areas per persona (not the full prompts):
- **Planner design rules**: guidance used by the initial layout planner when producing Flex‑DSL JSON.
- **Critic design rules**: guidance used by the critic when scoring and flagging issues.
- **Refiner design rules**: guidance for the refiner when making minor adjustments.

All other prompt context (object data, constraints, schema, etc.) is constructed automatically.

## Typical applications

- **Ad creative resizing/adaptation** (vertical, square, horizontal, ultra‑wide)
- **Social placements** (story, reel, post, banner) from a single master creative
- **Localization & variant generation** (same macro structure, different copy/assets)
- **A/B experimentation** on layout structure before micro‑polish stages

## Assumptions and limits

- Input is already segmented; this module does not perform segmentation.
- Outputs are “macro” placements; downstream tools should handle fine alignment, typography, and style.
- The Flex‑DSL is intentionally small (depth ≤ 2) to keep the search space tractable and the output deterministic.

## Running the app (local)

Prereqs: Python 3.10+.

Setup with venv and pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Alternative with uv:
```bash
uv pip install -r requirements.txt
uv run streamlit run app.py
```

In the UI:
- Enter your Nebius API key in the sidebar (used only for this session). You can also export it as `NEBIUS_API_KEY` for CLI runs.
- Pick an input image from `input/`.
- Ensure the segmentation bundle exists at `output/<image_stem>/` with `background.png`, `results.json`, and `objects/`.
- Set ratio, align, margin, temperature, and refine iterations.
- Optionally edit the persona design rules. Run the pipeline and explore iterations.

## Command-line usage (no UI)

You can run the pipeline directly via the CLI entry in `macro_placement_test.py`:

```bash
export NEBIUS_API_KEY=...  # or rely on Ollama if you configure --api ollama
python macro_placement_test.py \
  --image input/squarespace.jpg \
  --ratio 9:16 \
  --align center \
  --margin 0.05 \
  --api nebius \
  --temperature 0.8 \
  --refine-iters 3
```

Notes:
- The script expects the segmentation bundle at `output/<image_stem>/` next to `input/`.
- It writes results to `output_macro_placement/<image_stem>/` and clears any prior run for that stem.

## Inputs and outputs (expected layout)

- `input/<name>.png|jpg` – original input image (displayed in the UI)
- `output/<name>/` – segmentation bundle (required)
  - `background.png` – original background with holes/alpha
  - `results.json` – list of objects: `{object_id, filename, label, bounding_box}`
  - `objects/` – cutouts referenced by `results.json`
- `output_macro_placement/<name>/iteration_XX/`
  - `final_product/draft_macro_iter_XX.png` – composed result
  - `vlm_input_text/*` – prompts and validator outputs
  - `vlm_output/*` – raw VLM responses and layout JSONs
  - `layout_json/layout_macro_iter_XX.json` – final placements used by compositor
  - `time_log.txt` – step timings

## Code structure (brief)

- `macro_placement_test.py` – pipeline entry (`run_macro_only`), prompt building, VLM calls, validation, iteration orchestration
- `api_client.py` – unified Nebius/Ollama client (Nebius via OpenAI SDK base_url); accepts key from env or UI
- `layout_constraints.py` – canvas sizing, grid helpers, flow layout utilities
- `background_resizing.py` – background synthesis (solid median color)
- `compositor.py` – deterministic alpha compositing given placements
- `app.py` – Streamlit UI (parameters, persona design-rule editing, iteration viewer)
- `utils/` – small helpers (timing, labels)

### Agentic directory (experimental)

The `agentic/` directory contains an in-progress LangGraph-based workflow that aims to split the system into Macro, Micro, Critic, and Validator agents with tool-calling. It also includes a Streamlit UI at `agentic/app.py`. This path is not yet functional end-to-end; consider it a work-in-progress reference for future development.

## Roadmap

- Micro placement stage (pixel‑level nudges, overlaps, typography)
- Constraint solver integration for stricter guarantees
- Multi‑run comparison and presets for persona prompts
- Optional gradient/texture background synthesis

## Notes on privacy

The Nebius API key is collected in the Streamlit sidebar and used only in the current session. It is not logged or written to disk.


