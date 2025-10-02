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

# Layout: image selection on the left, folder controls on the right
left_col, right_col = st.columns([3, 1])

with right_col:
    st.subheader("Folders")
    default_images_folder = st.session_state.get("images_folder_name", "input")
    default_bundle_folder = st.session_state.get("bundle_folder_name", "output")
    images_folder = st.text_input(
        "Images folder name",
        value=default_images_folder,
        help="Folder (under this app directory) containing original images.",
    )
    bundle_folder = st.text_input(
        "Segmentation bundle folder name",
        value=default_bundle_folder,
        help="Folder (under this app directory) containing per-image bundles with objects and metadata.",
    )
    st.session_state["images_folder_name"] = images_folder.strip() or "input"
    st.session_state["bundle_folder_name"] = bundle_folder.strip() or "output"

# Resolve directories based on user inputs
input_dir = SCRIPT_DIR / st.session_state["images_folder_name"]
output_dir_base = SCRIPT_DIR / st.session_state["bundle_folder_name"]

with left_col:
    choices = _list_inputs(input_dir)
    if (
        "selected_input" not in st.session_state
        or st.session_state["selected_input"] not in [p.name for p in choices]
    ):
        st.session_state["selected_input"] = choices[0].name if choices else None

    # Thumbnails and selectors (2-column grid)
    thumb_cols = st.columns(2)
    for idx, p in enumerate(choices):
        with thumb_cols[idx % 2]:
            if p.exists():
                st.image(str(p), use_container_width=True)
                if st.button(f"Use {p.name}", key=f"select_{p.name}"):
                    st.session_state["selected_input"] = p.name

    selected_name = st.session_state.get("selected_input")
    selected_path = (input_dir / selected_name) if selected_name else None
    if selected_path:
        st.success(f"Selected: {selected_path.name}")

st.subheader("Design Rules (Custom Guiding Principles)")
st.caption("Customize the design rules below. Leave empty to use defaults. All other prompt components (object data, constraints, JSON schema) are automatically included.")
planner_rules_key = "planner_custom_design_rules"
critic_rules_key = "critic_custom_design_rules"
refiner_rules_key = "refiner_custom_design_rules"

if planner_rules_key not in st.session_state:
    st.session_state[planner_rules_key] = ""
if critic_rules_key not in st.session_state:
    st.session_state[critic_rules_key] = ""
if refiner_rules_key not in st.session_state:
    st.session_state[refiner_rules_key] = ""

def _get_default_design_rules(ratio: str) -> str:
    """Get the default design rules for the given aspect ratio."""
    aspect_family = _ratio_family(ratio)
    return _best_practices_text(aspect_family)

# Pre-fill design rules boxes with defaults if empty and selection available
if selected_path is not None:
    bundle = (SCRIPT_DIR / st.session_state.get("bundle_folder_name", "output")) / selected_path.stem
    if bundle.exists():
        default_rules = _get_default_design_rules(ratio)
        if not st.session_state[planner_rules_key]:
            st.session_state[planner_rules_key] = default_rules
        if not st.session_state[critic_rules_key]:
            st.session_state[critic_rules_key] = default_rules
        if not st.session_state[refiner_rules_key]:
            st.session_state[refiner_rules_key] = default_rules

planner_custom_design_rules = st.text_area(
    "Planner design rules", 
    value=st.session_state[planner_rules_key], 
    height=150,
    help="Custom design rules for the initial layout planner. Leave empty to use defaults based on aspect ratio."
)
critic_custom_design_rules = st.text_area(
    "Critic design rules", 
    value=st.session_state[critic_rules_key], 
    height=150,
    help="Custom design rules for the critic. Leave empty to use defaults based on aspect ratio."
)
refiner_custom_design_rules = st.text_area(
    "Refiner design rules", 
    value=st.session_state[refiner_rules_key], 
    height=150,
    help="Custom design rules for the layout refiner. Leave empty to use defaults based on aspect ratio."
)

st.session_state[planner_rules_key] = planner_custom_design_rules
st.session_state[critic_rules_key] = critic_custom_design_rules
st.session_state[refiner_rules_key] = refiner_custom_design_rules

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_clicked = st.button("Run macro placement")
with run_col2:
    st.write("")

status_box = st.empty()
gallery_box = st.container()

def _bundle_dir_for(image_path: Path) -> Path:
    return output_dir_base / image_path.stem


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
                    planner_custom_design_rules=planner_custom_design_rules.strip() or None,
                    critic_custom_design_rules=critic_custom_design_rules.strip() or None,
                    refiner_custom_design_rules=refiner_custom_design_rules.strip() or None,
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