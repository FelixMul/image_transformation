"""Model wrappers for calling Nebius chat models within LangGraph nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from api_client import get_api_client


class NebiusChatModel:
    """Thin wrapper exposing a LangChain-style invoke interface."""

    def __init__(
        self,
        api_type: str,
        api_key: str | None,
        *,
        temperature: float = 0.0,
        model_name: str | None = None,
    ) -> None:
        self.client = get_api_client(api_type, api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name

    def invoke(self, payload: Dict[str, Any]) -> SimpleNamespace:
        if isinstance(payload, dict):
            messages = payload.get("messages", payload)
            tools: Optional[List[Dict[str, Any]]] = payload.get("tools")
            tool_choice = payload.get("tool_choice")
        else:
            messages = payload
            tools = None
            tool_choice = None

        response = self.client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            model=self.model_name,
            tools=tools,
            tool_choice=tool_choice,
        )

        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        return SimpleNamespace(content=content, tool_calls=tool_calls)


def create_chat_model(
    api_type: str,
    api_key: str | None,
    *,
    temperature: float,
    model_name: str | None = None,
) -> NebiusChatModel:
    return NebiusChatModel(
        api_type=api_type,
        api_key=api_key,
        temperature=temperature,
        model_name=model_name,
    )


__all__ = ["NebiusChatModel", "create_chat_model"]

