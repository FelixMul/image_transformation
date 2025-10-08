# Agentic Macro Placement Workflow

> **Status:** Functional end-to-end pipeline with Nebius tool-calling
> agents, deterministic rendering, and Streamlit interface.

## Vision

This directory houses a reimplementation of the macro placement system
using the LangGraph agent framework. The goal is to separate concerns
more explicitly across agents:

- **Macro Layouter** – plans the container structure using the Flex DSL.
- **Micro Layouter** – adjusts object placements with pixel-true tools.
- **Critic** – reviews outputs and provides textual feedback.
- **Validator & Compositor** – enforce coverage and render deterministic
  outputs.

Each agent will operate with dedicated toolsets (see `tools/`).

## Directory Structure

```
agentic/
  app.py               # Streamlit entry point (fully wired)
  graph.py             # LangGraph graph assembly
  nodes/               # Agent node factories
  prompts/             # Prompt templates per agent
  state.py             # Workflow state dataclasses
  tools/               # Tool registries for agents
  utils/               # Shared helpers (prompt loading, JSON parsing)
```

## Workflow Overview

1. **State Initialisation** – `initialize_state()` loads the segmentation
   bundle, records object sizes, computes the target canvas (preserving
   pixel count), and prepares artifact directories.
2. **Macro Pass** – the Macro Layouter agent produces a Flex-DSL root
   container. The DSL is validated, converted into baseline placements,
   and coverage is enforced.
3. **Critique** – the critic agent reviews placements (deterministically
   rendered) and emits structured feedback with actionable suggestions.
4. **Micro Adjustments** – the micro layouter receives critic feedback
   and issues LangGraph tool calls (`adjust_x`, `adjust_y`) that nudge
   placements directly. All movements operate on original object sizes;
   no scaling is ever applied.
5. **Validation & Rendering** – each cycle runs coverage validation and
   composites the layout against a solid-fill background. Artifacts
   (images, JSON, prompts, responses) are saved under
   `output_macro_placement/<image>/iteration_xx/`.
6. **Iteration Control** – the graph repeats critic → micro → validator
   cycles until the requested iteration budget is exhausted or the micro
   agent issues no tool calls. The critic’s final message closes the run.

## Streamlit Usage

Run `streamlit run agentic/app.py`. The sidebar captures the Nebius API
key, aspect ratio, temperature, and iteration budget. Select an image
present in `input/` and click **Run Agentic Workflow**. Results and all
artifacts are displayed in expandable sections. Each iteration shows
rendered composites, layout JSON, VLM prompts, and raw responses.

## Prompts & Tools

- Prompts live in `prompts/`, one file per agent. They reference the
  state fields injected at runtime (canvas size, placements, feedback).
- Macro tools live in `tools/macro_layouter/` (currently a single
  `set_flex_json` entry point).
- Micro tools live in `tools/micro_layouter/`, exposing `adjust_x` and
  `adjust_y` with LangGraph-compatible function schemas.

## Extensibility Notes

- Add new micro tools by updating the registry in
  `tools/micro_layouter/__init__.py` and referencing them in the prompt.
- Additional critics or refinement phases can be added by extending
  `graph.py` with new nodes and conditional routes.
- Background generation is currently solid-fill but implemented via
  `background_resizing.fill_solid`, leaving room for future gradient or
  procedural alternatives without changing the node contract.

## Testing & Debugging

- Print statements persist in key nodes (e.g., micro tool execution) to
  assist with terminal debugging.
- If a tool call fails validation (missing objects, canvas overflow),
  the workflow raises an error and the Streamlit UI displays the message.
- Ensure the Nebius API key has access to `Qwen/Qwen2.5-VL-72B-Instruct`
  to support tool calling.

## Related Documentation

- `current_code_architecture.md` captures the legacy pipeline for
  reference.
- `README.md` at repository root summarises the overall project.



