import json
import os
import re
from datetime import datetime, timezone
from typing import Iterable, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


class ProxySanitizer(CustomLogger):
    _tier_boundaries = {
        "simple_medium": 0.15,
        "medium_complex": 0.35,
        "complex_reasoning": 0.7,
    }
    _tier_models = {
        "SIMPLE": "gpt-5-nano",
        "MEDIUM": "gpt-5-mini",
        "COMPLEX": "gpt-5",
        "REASONING": "gpt-5.1",
    }
    _dimension_weights = {
        "tokenCount": 0.12,
        "codePresence": 0.30,
        "reasoningMarkers": 0.20,
        "technicalTerms": 0.25,
        "simpleIndicators": 0.08,
        "multiStepPatterns": 0.03,
        "questionComplexity": 0.02,
    }

    _reasoning_markers = re.compile(
        r"\b(step by step|reason|derive|prove|think through|explain your reasoning)\b",
        re.IGNORECASE,
    )
    _simple_markers = re.compile(
        r"\b(hi|hello|hey|what is|define|meaning of)\b",
        re.IGNORECASE,
    )
    _tech_markers = re.compile(
        r"\b(kubernetes|microservice|distributed|architecture|latency|throughput|"
        r"consistency|vector|embedding|kafka|grpc|oauth|jwt|postgres|mysql|redis|"
        r"cloud|observability|tracing|service mesh|istio|linkerd|saga|eventual)\b",
        re.IGNORECASE,
    )
    _code_markers = re.compile(
        r"```|^\s*(def|class|function)\b|\{|\};|\b(makefile|dockerfile|readme\.md)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    _multi_step_markers = re.compile(r"\b(1\.|2\.|first|second|then|next)\b", re.IGNORECASE)

    _repo_edit_markers = re.compile(
        r"\b(makefile|readme\.md|docker(-|\s*)compose|git\s+(diff|status|commit)|patch|diffstat|"
        r"edit|modify|update|insert|str_replace|file_editor|terminal)\b",
        re.IGNORECASE,
    )

    _ticket_id_re = re.compile(r"\bTicket:\s*(T-\d+)\b", re.IGNORECASE)
    _execution_id_re = re.compile(r"\bExecution:\s*([A-Za-z0-9_.:@/-]+)\b", re.IGNORECASE)
    _mode_re = re.compile(r"\bMode:\s*(smart-router|baseline)\b", re.IGNORECASE)

    _tool_schema_hint = {
        "role": "system",
        "content": (
            "Tool-calling requirement (critical): when you call ANY tool (e.g. terminal, file_editor, "
            "str_replace_editor), ALWAYS include the required argument security_risk with value \"low\" "
            "unless explicitly higher.\n"
            "- Prefer minimal, targeted edits.\n"
        ),
    }

    def _normalize_router_model_name(self, model: Optional[str]) -> Optional[str]:
        if not isinstance(model, str) or not model:
            return model
        # Some clients may send prefixed names like "litellm_proxy/smart-router".
        if model.endswith("/smart-router"):
            return "smart-router"
        if model.endswith("/complexity_router"):
            return "complexity_router"
        return model

    def _extract_ticket_id(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text:
            return None
        m = self._ticket_id_re.search(text)
        if not m:
            return None
        return m.group(1).upper()

    def _extract_execution_id(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text:
            return None
        m = self._execution_id_re.search(text)
        if not m:
            return None
        return m.group(1)

    def _extract_execution_mode(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text:
            return None
        m = self._mode_re.search(text)
        if not m:
            return None
        return m.group(1).lower()

    def _extract_user_text(self, messages: Iterable[dict]) -> str:
        parts: list[str] = []
        for msg in messages or []:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
            else:
                parts.append(str(content))
        return "\n".join(p for p in parts if p)

    def _normalize_responses_input_text_types(self, data: dict) -> None:
        """
        OpenAI Responses API expects different content types by role:
        - user/system/developer -> input_text
        - assistant -> output_text

        Some clients send generic type='text' blocks in `input`, which causes
        invalid value errors on follow-up turns.
        """
        payload = data.get("input")
        if not isinstance(payload, list):
            return

        for item in payload:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and "text" in block:
                    if role == "assistant":
                        block["type"] = "output_text"
                    else:
                        block["type"] = "input_text"

    def _normalize_responses_tools(self, data: dict) -> None:
        """
        Normalize Chat Completions style function tools to Responses API style.

        Chat style:
          {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
        Responses style:
          {"type":"function","name":"...","description":"...","parameters":{...}}
        """
        tools = data.get("tools")
        if not isinstance(tools, list):
            return

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            if tool.get("name"):
                continue

            fn = tool.get("function")
            if not isinstance(fn, dict):
                continue

            if fn.get("name"):
                tool["name"] = fn.get("name")
            if "description" not in tool and "description" in fn:
                tool["description"] = fn.get("description")
            if "parameters" not in tool and "parameters" in fn:
                tool["parameters"] = fn.get("parameters")
            if "strict" not in tool and "strict" in fn:
                tool["strict"] = fn.get("strict")

    def _normalize_stream_options(self, data: dict) -> None:
        """
        Some clients send stream_options.include_usage, which certain OpenAI paths reject.
        Remove only the unsupported key and keep the rest of stream_options.
        """
        stream_options = data.get("stream_options")
        if not isinstance(stream_options, dict):
            return

        if "include_usage" in stream_options:
            stream_options.pop("include_usage", None)
        if not stream_options:
            data.pop("stream_options", None)

    def _normalize_agent_history_items(self, data: dict) -> None:
        """
        Normalize Chat-style tool history in Responses API `input`.

        Cursor Agent can send:
        - assistant messages with `tool_calls`
        - tool role messages with `tool_call_id`

        OpenAI Responses rejects `input[*].tool_calls`, so we:
        1) strip `tool_calls` / `function_call` from role messages
        2) emit explicit `function_call` / `function_call_output` items
        """
        payload = data.get("input")
        if not isinstance(payload, list):
            return

        normalized_input = []
        for item in payload:
            if not isinstance(item, dict):
                normalized_input.append(item)
                continue

            role = str(item.get("role", "")).lower()

            # Convert chat "tool" messages to responses function_call_output items.
            if role == "tool":
                call_id = item.get("tool_call_id") or item.get("id") or "call_unknown"
                raw_output = item.get("content", "")
                if isinstance(raw_output, (dict, list)):
                    output_text = json.dumps(raw_output, ensure_ascii=False)
                else:
                    output_text = str(raw_output)
                normalized_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call_id),
                        "output": output_text,
                    }
                )
                continue

            # Keep a cleaned role-message version.
            cleaned_item = dict(item)
            tool_calls = cleaned_item.pop("tool_calls", None)
            legacy_function_call = cleaned_item.pop("function_call", None)
            normalized_input.append(cleaned_item)

            # Convert assistant tool_calls to Responses function_call items.
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function") or {}
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name")
                    arguments = fn.get("arguments")
                    call_id = tool_call.get("id") or tool_call.get("call_id")
                    if not name:
                        continue
                    if arguments is None:
                        arguments = "{}"
                    elif not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    normalized_input.append(
                        {
                            "type": "function_call",
                            "call_id": str(call_id or f"call_{name}"),
                            "name": str(name),
                            "arguments": arguments,
                        }
                    )

            # Convert legacy single function_call shape.
            if isinstance(legacy_function_call, dict):
                name = legacy_function_call.get("name")
                arguments = legacy_function_call.get("arguments")
                if name:
                    if arguments is None:
                        arguments = "{}"
                    elif not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    normalized_input.append(
                        {
                            "type": "function_call",
                            "call_id": f"call_{name}",
                            "name": str(name),
                            "arguments": arguments,
                        }
                    )

        data["input"] = normalized_input

    def _extract_tag_text(self, messages: Iterable[dict]) -> str:
        parts: list[str] = []
        for msg in messages or []:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
            else:
                parts.append(str(content))
        return "\n".join(p for p in parts if p)

    def _complexity_score(self, text: str) -> float:
        tokens = len(re.findall(r"\w+", text))
        token_count = min(tokens / 400.0, 1.0)
        code_presence = 1.0 if self._code_markers.search(text) else 0.0
        reasoning_markers = 1.0 if self._reasoning_markers.search(text) else 0.0
        tech_terms = len(self._tech_markers.findall(text))
        technical_terms = min(tech_terms / 5.0, 1.0)
        simple_indicators = 1.0 if (tokens <= 12 and self._simple_markers.search(text)) else 0.0
        multi_step = 1.0 if self._multi_step_markers.search(text) else 0.0
        question_complexity = 1.0 if tokens >= 40 or text.count("?") > 1 else 0.0

        score = (
            token_count * self._dimension_weights["tokenCount"]
            + code_presence * self._dimension_weights["codePresence"]
            + reasoning_markers * self._dimension_weights["reasoningMarkers"]
            + technical_terms * self._dimension_weights["technicalTerms"]
            + multi_step * self._dimension_weights["multiStepPatterns"]
            + question_complexity * self._dimension_weights["questionComplexity"]
            - simple_indicators * self._dimension_weights["simpleIndicators"]
        )
        return max(0.0, min(score, 1.0))

    def _pick_tier(self, score: float) -> str:
        if score >= self._tier_boundaries["complex_reasoning"]:
            return "REASONING"
        if score >= self._tier_boundaries["medium_complex"]:
            return "COMPLEX"
        if score >= self._tier_boundaries["simple_medium"]:
            return "MEDIUM"
        return "SIMPLE"

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ):
        # Normalize known client payload mismatches for OpenAI Responses API.
        self._normalize_agent_history_items(data)
        self._normalize_responses_input_text_types(data)
        self._normalize_responses_tools(data)
        self._normalize_stream_options(data)

        original_model = data.get("model")
        if "custom_llm_provider" in data:
            data.pop("custom_llm_provider", None)

        data["model"] = self._normalize_router_model_name(data.get("model"))

        messages = data.get("messages", [])
        tag_text = self._extract_tag_text(messages if isinstance(messages, list) else [])
        metadata = data.get("metadata") or {}
        if original_model:
            metadata["requested_model"] = str(original_model)
        ticket_id = metadata.get("ticket_id") or self._extract_ticket_id(tag_text)
        if ticket_id:
            metadata["ticket_id"] = ticket_id
        execution_id = metadata.get("execution_id") or self._extract_execution_id(tag_text)
        if execution_id:
            metadata["execution_id"] = execution_id
        execution_mode = metadata.get("execution_mode") or self._extract_execution_mode(tag_text)
        if not execution_mode and isinstance(original_model, str):
            execution_mode = "smart-router" if original_model.endswith("smart-router") else "baseline"
        if execution_mode:
            metadata["execution_mode"] = execution_mode
        data["metadata"] = metadata

        if data.get("model") in {"complexity_router", "auto_router/complexity_router"}:
            data["model"] = "smart-router"

        if data.get("model") == "smart-router":
            messages = data.get("messages")
            if isinstance(messages, list) and (
                not messages or messages[0].get("content") != self._tool_schema_hint["content"]
            ):
                data["messages"] = [self._tool_schema_hint, *messages]

            user_text = self._extract_user_text(data.get("messages", []))
            score = self._complexity_score(user_text)
            tier = self._pick_tier(score)
            selected_model = self._tier_models[tier]

            repo_edit_match = bool(self._repo_edit_markers.search(user_text))
            if repo_edit_match and tier in {"SIMPLE", "MEDIUM"}:
                selected_model = "gpt-5.1"

            data["model"] = selected_model
            metadata = data.get("metadata") or {}
            metadata["router_tier"] = tier
            metadata["router_score"] = round(score, 3)
            metadata["router_model"] = selected_model
            data["metadata"] = metadata

        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: N802
        try:
            metadata = (kwargs or {}).get("metadata") or {}
            messages = (kwargs or {}).get("messages")
            if not messages:
                messages = ((kwargs or {}).get("litellm_params") or {}).get("messages")
            if not messages:
                messages = []

            tag_text = self._extract_tag_text(messages if isinstance(messages, list) else [])
            for extra_key in ("input", "prompt", "user_message"):
                extra_val = (kwargs or {}).get(extra_key)
                if extra_val:
                    tag_text = f"{tag_text}\n{extra_val}"

            ticket_id = metadata.get("ticket_id") or self._extract_ticket_id(tag_text)
            execution_id = metadata.get("execution_id") or self._extract_execution_id(tag_text)
            execution_mode = metadata.get("execution_mode") or self._extract_execution_mode(tag_text)

            requested_model = metadata.get("requested_model") or (kwargs or {}).get("model")
            routed_tier = metadata.get("router_tier")
            routed_score = metadata.get("router_score")
            routed_model = metadata.get("router_model")

            usage = response_obj.get("usage") if isinstance(response_obj, dict) else getattr(response_obj, "usage", None)
            usage = usage or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            cost_usd = None
            hidden = response_obj.get("_hidden_params") if isinstance(response_obj, dict) else getattr(response_obj, "_hidden_params", None)
            if isinstance(hidden, dict) and isinstance(hidden.get("response_cost"), (int, float)):
                cost_usd = float(hidden["response_cost"])
            else:
                try:
                    cost_usd = float(litellm.completion_cost(completion_response=response_obj))
                except Exception:
                    cost_usd = None

            response_model = response_obj.get("model") if isinstance(response_obj, dict) else getattr(response_obj, "model", None)

            duration_s = None
            try:
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    duration_s = float(end_time - start_time)
            except Exception:
                duration_s = None

            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ticket_id": ticket_id,
                "execution_id": execution_id,
                "execution_mode": execution_mode,
                "requested_model": requested_model,
                "routed_tier": routed_tier,
                "routed_score": routed_score,
                "routed_model": routed_model,
                "response_model": response_model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "duration_s": duration_s,
            }

            path = os.environ.get("ROUTER_METRICS_PATH") or "/app/artifacts/router_metrics.jsonl"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            return

    async def async_post_call_response_headers_hook(  # noqa: N802
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_headers=None,
    ):
        metadata = (data or {}).get("metadata") or {}
        tier = metadata.get("router_tier")
        score = metadata.get("router_score")
        model = metadata.get("router_model")

        if tier is None and score is None and model is None:
            return None

        headers = {}
        if tier is not None:
            headers["x-router-tier"] = str(tier)
        if score is not None:
            headers["x-router-score"] = str(score)
        if model is not None:
            headers["x-router-model"] = str(model)

        resp_model = getattr(response, "model", None)
        if resp_model:
            headers["x-response-model"] = str(resp_model)

        return headers


proxy_handler_instance = ProxySanitizer()

