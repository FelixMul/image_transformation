#This is just a placeholder for the app.py file.
import streamlit as st
from pathlib import Path
import os
import json
from typing import List

from macro_placement_test import (
    run_macro_only,
    SCRIPT_DIR,
    _compute_nesting_conflicts,
    _ratio_family,
    _best_practices_text,
    _build_shared_prompt_context,
)


def _list_inputs(input_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    if input_dir.exists():
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                imgs.append(p)
    return imgs


st.set_page_config(page_title="Macro Placement", layout="wide")

st.sidebar.header("Connection")
api_key = st.sidebar.text_input("Nebius API Key", type="password", help="Stored only in this session.")
connected = st.sidebar.button("Connect")
if connected and not api_key:
    st.sidebar.warning("Please provide an API key.")
if api_key:
    st.sidebar.success("API key ready for this session.")

st.sidebar.header("Parameters")
col_w, col_h = st.sidebar.columns(2)
with col_w:
    ratio_w = st.number_input("Ratio W", min_value=1, max_value=100, value=9)
with col_h:
    ratio_h = st.number_input("Ratio H", min_value=1, max_value=100, value=16)
ratio = f"{ratio_w}:{ratio_h}"

align = st.sidebar.radio("Align", options=["center", "edge"], index=0)
margin = st.sidebar.slider("Margin (0-0.3)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.2, value=1.0, step=0.1)
refine_iters = st.sidebar.slider("Refine iterations", min_value=0, max_value=10, value=2, step=1)


st.title("Macro Placement – Streamlit UI")

input_dir = SCRIPT_DIR / "input"
output_dir_base = SCRIPT_DIR / "output"

choices = _list_inputs(input_dir)
cols = st.columns(2)
if "selected_input" not in st.session_state:
    st.session_state["selected_input"] = choices[0].name if choices else None

for idx, p in enumerate(choices[:2]):
    with cols[idx % 2]:
        if p.exists():
            st.image(str(p), use_container_width=True)
            if st.button(f"Use {p.name}", key=f"select_{p.name}"):
                st.session_state["selected_input"] = p.name

selected_name = st.session_state.get("selected_input")
selected_path = (input_dir / selected_name) if selected_name else None
if selected_path:
    st.success(f"Selected: {selected_path.name}")

st.subheader("Prompts (full overrides)")
planner_override_key = "planner_prompt_override"
critic_override_key = "critic_prompt_override"
refiner_override_key = "refiner_prompt_override"

if planner_override_key not in st.session_state:
    st.session_state[planner_override_key] = ""
if critic_override_key not in st.session_state:
    st.session_state[critic_override_key] = ""
if refiner_override_key not in st.session_state:
    st.session_state[refiner_override_key] = ""

planner_prompt_override = st.text_area("Initial layout prompt (full)", value=st.session_state[planner_override_key], height=320, placeholder="Leave empty to use the default constructed planner prompt.")
critic_prompt_override = st.text_area("Critic prompt (full)", value=st.session_state[critic_override_key], height=240, placeholder="Leave empty to use the default constructed critic prompt.")
refiner_prompt_override = st.text_area("Refiner prompt (full)", value=st.session_state[refiner_override_key], height=240, placeholder="Leave empty to use the default constructed refiner prompt.")

st.session_state[planner_override_key] = planner_prompt_override
st.session_state[critic_override_key] = critic_prompt_override
st.session_state[refiner_override_key] = refiner_prompt_override

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_clicked = st.button("Run macro placement")
with run_col2:
    st.write("")

status_box = st.empty()
gallery_box = st.container()

def _bundle_dir_for(image_path: Path) -> Path:
    return output_dir_base / image_path.stem

def _build_default_planner_prompt(bundle: Path, ratio: str, canvas_size: tuple[int, int]) -> str:
    # Re-create the same logic as in _vlm_request_flex to show a default prompt
    results_json_path = bundle / "results.json"
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    iw, ih = (canvas_size[0], canvas_size[1])
    summary_lines = []
    role_lines = []
    for it in items:
        oid = int(it["object_id"])
        label = it.get("label", "")
        low = label.lower()
        if "logo" in low:
            role = "logo"
        elif "cta" in low:
            role = "cta"
        elif "text" in low or "copy" in low:
            role = "text"
        elif "design" in low or "shape" in low or "element" in low:
            role = "design"
        else:
            role = "image"
        x1, y1, x2, y2 = it.get("bounding_box", [0, 0, 0, 0])
        nx1 = round(x1 / max(1, iw), 4)
        ny1 = round(y1 / max(1, ih), 4)
        nx2 = round(x2 / max(1, iw), 4)
        ny2 = round(y2 / max(1, ih), 4)
        summary_lines.append(f"id={oid}, name='{label}', role={role}, bbox_norm=[{nx1},{ny1},{nx2},{ny2}]")
        role_lines.append(f"{oid}:{role}")
    summary_text = "\n".join(summary_lines)
    row_bad, col_bad = _compute_nesting_conflicts(str(bundle / "results.json"), canvas_size, margin)
    row_bad_str = ", ".join([f"({a}, {b})" for a, b in row_bad]) or "none"
    col_bad_str = ", ".join([f"({a}, {b})" for a, b in col_bad]) or "none"
    aspect_family = _ratio_family(ratio)
    best_practices = _best_practices_text(aspect_family)
    shared_context_block = _build_shared_prompt_context(best_practices, summary_text, role_lines, row_bad_str, col_bad_str)
    base_prompt = f"""### PERSONA

You are a pragmatic Layout Planner.
TASK

Your goal is to generate a valid first-draft layout in the Flex DSL JSON format. Analyze the original image to understand its visual intent and use the object data as your guide. Your layout must fit within the provided target canvas.

{shared_context_block}
OUTPUT INSTRUCTIONS

    Your output must be ONLY the valid JSON object.

    Do not include any explanations, comments, or markdown code fences."""
    return base_prompt


if run_clicked:
    if not api_key:
        st.error("Please enter a Nebius API key in the sidebar.")
    elif not selected_path:
        st.error("Please select an input image.")
    else:
        bundle = _bundle_dir_for(selected_path)
        bg_path = bundle / "background.png"
        res_json = bundle / "results.json"
        objects_dir = bundle / "objects"
        missing = [str(p) for p in [bg_path, res_json, objects_dir] if not p.exists()]
        if missing:
            st.error("Missing expected segmentation bundle items: " + ", ".join(missing))
        else:
            status_box.info("Running… this can take a few minutes depending on iterations.")
            try:
                run_macro_only(
                    output_dir=bundle,
                    ratio=ratio,
                    align=align,
                    margin=margin,
                    api_type="nebius",
                    samples=1,
                    temperature=temperature,
                    refine_iters=refine_iters,
                    original_input_path=str(selected_path),
                    api_key=api_key,
                    planner_prompt_override=planner_prompt_override.strip() or None,
                    critic_prompt_override=critic_prompt_override.strip() or None,
                    refiner_prompt_override=refiner_prompt_override.strip() or None,
                )
                status_box.success("Run completed.")
            except Exception as e:
                status_box.error(f"Run failed: {e}")

if selected_path:
    # Discover artifacts from the latest run (the pipeline currently purges and rewrites)
    base_out = SCRIPT_DIR / "output_macro_placement" / selected_path.stem
    if base_out.exists():
        iterations = sorted([p for p in base_out.iterdir() if p.is_dir() and p.name.startswith("iteration_")])
        if iterations:
            idx_max = len(iterations) - 1
            # Show the composed image smaller and centered above the slider
            top = iterations[0]  # placeholder for image sizing reference
            iter_idx = st.slider("Iteration", min_value=0, max_value=idx_max, value=0)
            cur = iterations[iter_idx]
            fp = cur / "final_product"
            vi = cur / "vlm_input_text"
            vo = cur / "vlm_output"
            lj = cur / "layout_json"

            img_main = fp / f"draft_macro_iter_{iter_idx:02d}.png"
            if img_main.exists():
                st.image(str(img_main), caption=img_main.name, width=520)
            else:
                st.info("No composed image for this iteration (likely no-change).")

            # One-click JSON viewing
            with st.expander("JSON artifacts"):
                tabs = st.tabs(["VLM layout JSON", "Final placements JSON", "Retry (if any)"])
                with tabs[0]:
                    jf = vo / f"layout_flex_iter_{iter_idx:02d}.json"
                    if jf.exists():
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            st.json(data, expanded=False)
                        except Exception:
                            st.code(jf.read_text(encoding="utf-8"), language="json")
                    else:
                        st.info("Not available.")
                with tabs[1]:
                    jf = lj / f"layout_macro_iter_{iter_idx:02d}.json"
                    if jf.exists():
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            st.json(data, expanded=False)
                        except Exception:
                            st.code(jf.read_text(encoding="utf-8"), language="json")
                    else:
                        st.info("Not available.")
                with tabs[2]:
                    jf = vo / f"layout_flex_iter_{iter_idx:02d}_retry.json"
                    if jf.exists():
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            st.json(data, expanded=False)
                        except Exception:
                            st.code(jf.read_text(encoding="utf-8"), language="json")
                    else:
                        st.info("No retry.")

            # Individually titled prompt/text sections
            with st.expander("Planner prompt" ):
                f = vi / "prompt_flex.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("Not available.")
            with st.expander("Refiner prompt"):
                f = vi / f"prompt_refine_iter_{iter_idx:02d}.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("Not available.")
            with st.expander("Refiner prompt (retry)"):
                f = vi / f"prompt_refine_iter_{iter_idx:02d}_retry.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("No retry.")
            with st.expander("Critic statement"):
                f = vo / f"critic_raw_iter_{iter_idx:02d}.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("Not available.")
            with st.expander("VLM raw output"):
                f = vo / f"vlm_raw_iter_{iter_idx:02d}.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("Not available.")
            with st.expander("Validation errors"):
                f = vi / f"flex_validation_error_iter_{iter_idx:02d}.txt"
                if f.exists():
                    st.code(f.read_text(encoding="utf-8"), language="text")
                else:
                    st.info("None.")

            tl = base_out / "time_log.txt"
            if tl.exists():
                with st.expander("Timing log"):
                    st.code(tl.read_text(encoding="utf-8"), language="text")
        else:
            st.info("No run artifacts yet. Configure parameters and click Run.")
    else:
        st.info("No run artifacts yet. Configure parameters and click Run.")