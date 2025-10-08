from __future__ import annotations

import json
import sys
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from layout_constraints import compute_canvas_size
from agentic.models import create_chat_model
from agentic.workflow import compile_workflow, initialize_state


INPUT_DIR = WORKSPACE_ROOT / "input"
BUNDLE_ROOT = WORKSPACE_ROOT / "output"
OUTPUT_ROOT = WORKSPACE_ROOT / "agentic" / "results"


def _list_available_images() -> List[str]:
    images = []
    for path in INPUT_DIR.glob("*.png"):
        images.append(path.stem)
    for path in INPUT_DIR.glob("*.jpg"):
        images.append(path.stem)
    return sorted(set(images))


def _ensure_output_dir(image_name: str) -> Path:
    out_dir = OUTPUT_ROOT / image_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_original_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return im.size


def _run_workflow(
    image_name: str,
    ratio: str,
    temperature: float,
    max_iterations: int,
    api_key: Optional[str],
) -> Dict[str, str]:
    bundle_dir = BUNDLE_ROOT / image_name
    original_image_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = INPUT_DIR / f"{image_name}{ext}"
        if candidate.exists():
            original_image_path = candidate
            break
    if original_image_path is None:
        raise FileNotFoundError(f"Original image for '{image_name}' not found in input/")

    original_size = _load_original_size(original_image_path)
    canvas_size = compute_canvas_size(original_size, ratio)
    base_run_dir = _ensure_output_dir(image_name)

    state = initialize_state(
        image_name=image_name,
        ratio=ratio,
        bundle_dir=bundle_dir,
        base_run_dir=base_run_dir,
        api_type="nebius",
        api_key=api_key,
        temperature=temperature,
        max_iterations=max_iterations,
        canvas_size=canvas_size,
        original_image_path=original_image_path,
    )

    macro_model = create_chat_model(
        api_type=state.api_type,
        api_key=state.api_key,
        temperature=state.temperature,
    )
    micro_model = create_chat_model(
        api_type=state.api_type,
        api_key=state.api_key,
        temperature=max(0.0, state.temperature - 0.1),
    )
    critic_model = create_chat_model(
        api_type=state.api_type,
        api_key=state.api_key,
        temperature=0.2,
    )

    graph = compile_workflow(state, macro_model, micro_model, critic_model)
    app = graph.compile()

    result_state = app.invoke(state)

    return {
        "iteration_dir": str(result_state.get_iteration_dir()),
        "current_composite": str(result_state.current_composite_path or ""),
    }


def _display_iteration(iteration_dir: Path) -> None:
    st.subheader(f"Iteration {iteration_dir.name.split('_')[-1]}")
    final_dir = iteration_dir / "final_product"
    for img_path in sorted(final_dir.glob("*.png")):
        st.image(str(img_path), caption=img_path.name, use_column_width=True)

    layout_dir = iteration_dir / "layout_json"
    for json_path in sorted(layout_dir.glob("*.json")):
        with json_path.open(encoding="utf-8") as fh:
            st.markdown(f"**{json_path.name}**")
            st.json(json.load(fh))

    text_dir = iteration_dir / "vlm_output"
    for txt_path in sorted(text_dir.glob("*.txt")):
        st.markdown(f"**{txt_path.name}**")
        st.code(txt_path.read_text(encoding="utf-8"))


def _sidebar_controls() -> Dict[str, any]:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("API Key", type="password")
        col1, col2 = st.columns([1, 1])
        with col1:
            ratio_w = st.number_input("Aspect W", min_value=1, max_value=4000, value=9)
        with col2:
            ratio_h = st.number_input("Aspect H", min_value=1, max_value=4000, value=16)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.05)
        max_iterations = st.slider("Micro Iterations", min_value=0, max_value=10, value=3)
    return {
        "api_key": api_key or None,
        "ratio": f"{int(ratio_w)}:{int(ratio_h)}",
        "temperature": temperature,
        "max_iterations": max_iterations,
    }


def main() -> None:
    st.set_page_config(page_title="Agentic Macro Placement", layout="wide")
    st.title("Agentic Macro Placement")

    available_images = _list_available_images()
    if not available_images:
        st.error("No input images found in 'input/' directory.")
        return

    config = _sidebar_controls()

    selected_image = st.selectbox("Select Image", available_images)

    if st.button("Run Agentic Workflow"):
        with st.spinner("Running agentic workflow..."):
            try:
                result = _run_workflow(
                    image_name=selected_image,
                    ratio=config["ratio"],
                    temperature=config["temperature"],
                    max_iterations=config["max_iterations"],
                    api_key=config["api_key"],
                )
                st.success("Workflow completed")
                st.session_state["last_run_image"] = selected_image
            except Exception as exc:
                st.error(f"Agentic workflow failed: {exc}")

    last_image = st.session_state.get("last_run_image", selected_image)
    output_dir = OUTPUT_ROOT / last_image
    iteration_dirs = sorted(output_dir.glob("iteration_*"))

    if iteration_dirs:
        st.header(f"Run Artifacts: {last_image}")
        for iteration_dir in iteration_dirs:
            with st.expander(iteration_dir.name, expanded=False):
                _display_iteration(iteration_dir)
    else:
        st.info("Run the workflow to populate outputs.")


if __name__ == "__main__":
    main()


