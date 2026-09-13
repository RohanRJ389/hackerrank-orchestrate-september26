"""Claude Agent SDK orchestration for evidence review."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    ThinkingConfigDisabled,
)

from .models import NormalizationDirectives, NormalizationPacket, Route
from .prompts import SYSTEM_PROMPT, listed_workspace_files, task_prompt
from .trace import TraceWriter

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class AgentRun:
    directives: NormalizationDirectives
    model: str
    duration_ms: int
    turns: int
    cost_usd: float
    usage: dict[str, Any]
    model_usage: dict[str, Any]
    session_id: str


def _sandbox() -> dict[str, Any]:
    return {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "excludedCommands": [],
        "allowUnsandboxedCommands": False,
        "network": {
            "allowUnixSockets": [],
            "allowAllUnixSockets": False,
            "allowLocalBinding": False,
        },
        "ignoreViolations": {"file": [], "network": []},
        "enableWeakerNestedSandbox": False,
    }


def _event_type(message: Any) -> str:
    if isinstance(message, AssistantMessage):
        return "assistant_message"
    if isinstance(message, ResultMessage):
        return "result"
    return type(message).__name__.lower()


def parse_max_turns(value: str | None) -> int | None:
    """Return a turn cap, or None when the cap is disabled.

    Empty, 0, none, and unlimited all mean no SDK max_turns limit.
    """
    if value is None:
        return None
    stripped = value.strip().lower()
    if stripped in {"", "0", "none", "unlimited"}:
        return None
    return int(stripped)


class AgentRunner:
    def __init__(
        self,
        *,
        model: str = HAIKU_MODEL,
        max_turns: int | None = None,
        max_budget_usd: float = 1.0,
        trace_root: Path | None = None,
        client_factory: Callable[..., Any] = ClaudeSDKClient,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.trace_root = trace_root or Path("normalizer-traces")
        self.client_factory = client_factory

    def _hooks(self, trace: TraceWriter) -> dict:
        async def before(input_data, _tool_use_id, _context):
            trace.write(
                "tool_call",
                {
                    "tool_name": input_data["tool_name"],
                    "tool_input": input_data["tool_input"],
                    "tool_use_id": input_data["tool_use_id"],
                },
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                }
            }

        async def after(input_data, _tool_use_id, _context):
            trace.write(
                "tool_result",
                {
                    "tool_name": input_data["tool_name"],
                    "tool_use_id": input_data["tool_use_id"],
                    "content": input_data.get("tool_response"),
                },
            )
            return {}

        async def failure(input_data, _tool_use_id, _context):
            trace.write(
                "tool_failure",
                {
                    "tool_name": input_data["tool_name"],
                    "tool_use_id": input_data["tool_use_id"],
                    "error": input_data.get("error"),
                },
            )
            return {}

        return {
            "PreToolUse": [HookMatcher(matcher="Read", hooks=[before])],
            "PostToolUse": [HookMatcher(matcher="Read", hooks=[after])],
            "PostToolUseFailure": [HookMatcher(matcher="Read", hooks=[failure])],
        }

    def _options(
        self, workspace: Path, request_id: str, trace: TraceWriter
    ) -> ClaudeAgentOptions:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if not workspace_id:
            raise RuntimeError(
                "ANTHROPIC_WORKSPACE_ID is not set; this API key requires a workspace"
            )
        environment = {
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_WORKSPACE_ID": workspace_id,
            "ANTHROPIC_CUSTOM_HEADERS": (
                f"anthropic-workspace-id: {workspace_id}"
            ),
            "NORMALIZER_REQUEST_ID": request_id,
            "NORMALIZER_DATASET_ROOT": str(workspace / "data"),
            "NORMALIZER_WORKSPACE": str(workspace),
            "PYTHONPATH": str(workspace / "runtime"),
        }
        return ClaudeAgentOptions(
            model=self.model,
            tools=["Read"],
            allowed_tools=["Read"],
            system_prompt=SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            effort="low",
            thinking=ThinkingConfigDisabled(type="disabled"),
            cwd=workspace,
            cli_path=os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude"),
            env=environment,
            sandbox=_sandbox(),
            hooks=self._hooks(trace),
            output_format={
                "type": "json_schema",
                "schema": NormalizationDirectives.model_json_schema(),
            },
            setting_sources=[],
            max_buffer_size=5_000_000,
        )

    async def run(
        self,
        workspace: Path,
        request_id: str,
        route: Route,
        packet: NormalizationPacket | None = None,
    ) -> AgentRun:
        run_id = str(uuid.uuid4())
        trace = TraceWriter(
            self.trace_root / run_id / f"{request_id}.jsonl", run_id, request_id
        )
        files = listed_workspace_files(workspace, packet)
        prompt = task_prompt(request_id=request_id, route=route, files=files)
        trace.write(
            "agent_start",
            {
                "model": self.model,
                "route": route.value,
                "workspace": workspace.name,
                "listed_files": files,
            },
        )
        result: ResultMessage | None = None
        async with self.client_factory(
            options=self._options(workspace, request_id, trace)
        ) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                trace.write(_event_type(message), message)
                if isinstance(message, ResultMessage):
                    result = message
        if result is None:
            raise RuntimeError("Claude Agent SDK returned no ResultMessage")
        if result.is_error or result.subtype != "success":
            raise RuntimeError(
                f"agent failed: subtype={result.subtype}, result={result.result}"
            )
        if result.structured_output is None:
            raise RuntimeError("agent returned no directives")
        directives = NormalizationDirectives.model_validate(result.structured_output)
        if directives.request_id != request_id:
            raise ValueError("agent returned directives for a different request")
        return AgentRun(
            directives=directives,
            model=self.model,
            duration_ms=result.duration_ms,
            turns=result.num_turns,
            cost_usd=result.total_cost_usd or 0,
            usage=result.usage or {},
            model_usage=getattr(result, "model_usage", None) or {},
            session_id=result.session_id,
        )
