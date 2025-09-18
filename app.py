#This is just a placeholder for the app.py file.
import streamlit as st
from pathlib import Path
import os
import json
from typing import List
from PIL import Image as PILImage

from macro_placement_test import (
    run_macro_only,
    SCRIPT_DIR,
    _compute_nesting_conflicts,
    _ratio_family,
    _best_practices_text,
    _build_shared_prompt_context,
)
from layout_constraints import compute_canvas_size


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

def _collect_shared_context(bundle: Path, ratio: str, margin: float) -> tuple[str, str, list[str]]:
    results_json_path = bundle / "results.json"
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    # Use original background to compute canvas size for normalization context
    bg_path = bundle / "background.png"
    from PIL import Image as _PILImage
    ow, oh = _PILImage.open(bg_path).convert("RGBA").size
    canvas_size = compute_canvas_size((ow, oh), ratio)
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
        nx1 = round(x1 / max(1, ow), 4)
        ny1 = round(y1 / max(1, oh), 4)
        nx2 = round(x2 / max(1, ow), 4)
        ny2 = round(y2 / max(1, oh), 4)
        summary_lines.append(f"id={oid}, name='{label}', role={role}, bbox_norm=[{nx1},{ny1},{nx2},{ny2}]")
        role_lines.append(f"{oid}:{role}")
    summary_text = "\n".join(summary_lines)
    row_bad, col_bad = _compute_nesting_conflicts(str(results_json_path), canvas_size, margin)
    row_bad_str = ", ".join([f"({a}, {b})" for a, b in row_bad]) or "none"
    col_bad_str = ", ".join([f"({a}, {b})" for a, b in col_bad]) or "none"
    aspect_family = _ratio_family(ratio)
    best_practices = _best_practices_text(aspect_family)
    shared_context_block = _build_shared_prompt_context(best_practices, summary_text, role_lines, row_bad_str, col_bad_str)
    return shared_context_block, summary_text, role_lines

def _default_planner_prompt(shared_context_block: str) -> str:
    return (
        """### PERSONA

You are a pragmatic Layout Planner.
TASK

Your goal is to generate a valid first-draft layout in the Flex DSL JSON format. Analyze the original image to understand its visual intent and use the object data as your guide. Your layout must fit within the provided target canvas.

"""
        + shared_context_block
        + """
OUTPUT INSTRUCTIONS

    Your output must be ONLY the valid JSON object.

    Do not include any explanations, comments, or markdown code fences."""
    )

def _default_critic_prompt(shared_context_block: str) -> str:
    return (
        """### PERSONA

You are a professional Creative Director and a strict Design Critic.
TASK

Your goal is to evaluate the provided layout draft. Your primary focus is to determine how well the draft preserves the visual intent, balance, and core message of the original advertisement while adapting it to a new format. You must be specific, honest, and actionable. Do not generate a solution or JSON.

"""
        + shared_context_block
        + """
EVALUATION & OUTPUT INSTRUCTIONS

Analyze the draft and provide your critique structured into the following sections. Be concise but specific.

1. Overall Score (out of 10):

    A single number from 0 to 10. (10=Perfect, 7=Acceptable, <5=Major flaws).

2. Preservation of Original Intent:

    How well does the draft maintain the original's visual hierarchy?

    Is the focus on the correct elements (e.g., the product, the main message)?

    Does the new layout feel like a professional adaptation or a random assortment of parts?

3. Hard Constraint Violations:

    Did the layout violate any of the non-negotiable rules from the CORE CONTEXT?

    Name the specific objects and rules that were broken (e.g., "Violates Row Nesting Conflict: ('Logo', 'Main Image')").

4. Composition & Design Issues:

    Comment on balance, alignment, negative space, and visual flow.

    Is the logo placement appropriate? Is the Call-to-Action (CTA) prominent and logically placed?

    Are there any awkward gaps, crowded areas, or margin violations?

5. Actionable Improvement Plan:

    Provide a clear, imperative list of changes for the next agent.

    Examples: "Change the root container's direction to 'column'." or "Create a nested row container for the logo and the tagline to keep them together." or "Swap the positions of the 'Main Image' and the 'Product Details' text block.""" 
    )

def _default_refiner_prompt(shared_context_block: str) -> str:
    return (
        """### PERSONA

You are an expert Layout Engineer and a Problem-Solver.
TASK

Your goal is to fix a flawed design by writing a new, superior Flex DSL JSON. You have been provided with the previous attempt, the original design goal, and a detailed critique from a Creative Director. Your new JSON must directly address all issues raised in the critique.

"""
        + shared_context_block
        + """
CRITIC'S REVIEW (YOUR TO-DO LIST)
You must fix the following issues:
<Insert critic review here after baseline>
OUTPUT INSTRUCTIONS

    Your output must be ONLY the new, corrected, and valid JSON object.

    Address all points from the critic's review.

    You are authorized to make radical changes to the previous JSON structure to fix the reported problems.

    Do not include any explanations, comments, or markdown code fences."""
    )

# Pre-fill prompt boxes with defaults if empty and selection available
if selected_path is not None:
    bundle = (SCRIPT_DIR / "output") / selected_path.stem
    if bundle.exists():
        shared_block, _sum, _roles = _collect_shared_context(bundle, ratio, margin)
        if not st.session_state[planner_override_key]:
            st.session_state[planner_override_key] = _default_planner_prompt(shared_block)
        if not st.session_state[critic_override_key]:
            st.session_state[critic_override_key] = _default_critic_prompt(shared_block)
        if not st.session_state[refiner_override_key]:
            st.session_state[refiner_override_key] = _default_refiner_prompt(shared_block)

planner_prompt_override = st.text_area("Initial layout prompt (full)", value=st.session_state[planner_override_key], height=320)
critic_prompt_override = st.text_area("Critic prompt (full)", value=st.session_state[critic_override_key], height=240)
refiner_prompt_override = st.text_area("Refiner prompt (full)", value=st.session_state[refiner_override_key], height=240)

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
                st.session_state["has_run"] = True
                # reset slider to 0 for fresh viewing
                st.session_state["iter_idx"] = 0
                status_box.success("Run completed.")
            except Exception as e:
                status_box.error(f"Run failed: {e}")

# Only show artifacts after a run in this session
if selected_path and st.session_state.get("has_run", False):
    # Discover artifacts from the latest run (the pipeline currently purges and rewrites)
    base_out = SCRIPT_DIR / "output_macro_placement" / selected_path.stem
    if base_out.exists():
        iterations = sorted([p for p in base_out.iterdir() if p.is_dir() and p.name.startswith("iteration_")])
        if iterations:
            idx_max = len(iterations) - 1
            # Determine current iteration index from session (default 0)
            iter_idx = int(st.session_state.get("iter_idx", 0))
            if iter_idx < 0 or iter_idx > idx_max:
                iter_idx = 0
            cur = iterations[iter_idx]
            fp = cur / "final_product"
            vi = cur / "vlm_input_text"
            vo = cur / "vlm_output"
            lj = cur / "layout_json"

            img_main = fp / f"draft_macro_iter_{iter_idx:02d}.png"
            # Center the image above the slider using columns, cap max dimension to 1000px
            if img_main.exists():
                try:
                    iw, ih = PILImage.open(img_main).size
                except Exception:
                    iw, ih = 1000, 1000
                max_dim = max(1, max(iw, ih))
                # target so that neither width nor height exceeds 1000
                scale = min(1.0, 1000 / float(max_dim))
                display_width = max(1, int(iw * scale))
                _c1, _c2, _c3 = st.columns([1, 2, 1])
                with _c2:
                    st.image(str(img_main), caption=img_main.name, width=display_width)
            else:
                st.info("No composed image for this iteration (likely no-change).")

            # Slider placed BELOW the image, updates session state
            if "iter_idx" not in st.session_state:
                st.session_state["iter_idx"] = 0
            _ = st.slider("Iteration", min_value=0, max_value=idx_max, value=iter_idx, key="iter_idx")

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