"""aidial_adapter_openai.utils.vllm_tokenizer

vLLM prompt token counting
--------------------------

For vLLM/open-source models we can't rely on tiktoken. Instead of local
  tokenization, we ask the upstream server to report `usage.prompt_tokens`.

Implementation strategy:
- Use the already-configured OpenAI client (AsyncOpenAI/AsyncAzureOpenAI).
- Send a **non-stream** chat completion request with `max_tokens=1`.
- Extract `usage.prompt_tokens` from the upstream response.

Notes:
- No modality-specific tokenization: we send the fully constructed payload
  (messages after Unified->OpenAI transformations, including embedded base64).
- VllmTokenizer does not tokenize the response.
"""

from typing import Any, Dict, List, Set

from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI

from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
)


class VllmTokenizer:
    """Tokenizer backed by a remote vLLM chat-completions endpoint."""

    model: str
    _client: AsyncAzureOpenAI | AsyncOpenAI

    def __init__(
        self,
        *,
        model: str,
        client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = client

    async def tokenize_request(
        self, original_request: dict, messages: List[MultiModalMessage]
    ) -> int:
        """Count prompt tokens for the full request via upstream usage."""

        raw_messages = [m.raw_message for m in messages]
        payload = self._build_usage_payload(original_request, raw_messages)
        return await self._call_completion_for_prompt_tokens(payload)

    async def truncate_prompt(
        self,
        original_request: dict,
        messages: List[MultiModalMessage],
        max_prompt_tokens: int,
    ) -> tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
        """Truncate messages to fit within *max_prompt_tokens*.

        Linear strategy:
        - Tokenize full payload; if it fits: return.
        - Otherwise, remove the oldest non-system message one-by-one.
          If a removed message is an assistant with tool_calls, also remove
          subsequent tool replies and the next assistant message that follows
          the tool chain.
        - If even system+last non-system doesn't fit: raise.
        """

        all_indices: Set[int] = set(range(len(messages)))

        def _collect(indices: Set[int]) -> List[MultiModalMessage]:
            return [messages[i] for i in sorted(indices)]

        # Fast path: everything fits
        prompt_tokens = await self.tokenize_request(
            original_request, _collect(all_indices)
        )
        if prompt_tokens <= max_prompt_tokens:
            return _collect(all_indices), [], prompt_tokens

        system_indices: list[int] = []
        non_system_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.raw_message.get("role") == "system":
                system_indices.append(idx)
            else:
                non_system_indices.append(idx)

        system_set: Set[int] = set(system_indices)

        if not non_system_indices:
            system_tokens = await self.tokenize_request(
                original_request, _collect(system_set)
            )
            raise TruncatePromptSystemError(max_prompt_tokens, system_tokens)

        kept: Set[int] = set(all_indices)

        def _cascade_remove_tool_replies(start_idx: int) -> None:
            """Remove tool replies following a tool-calling assistant.

            When we drop an assistant containing tool_calls, we must also drop:
            - consecutive 'tool' messages that follow it
            - and the next 'assistant' message (the agent follow-up that used
              tool results).
            """
            j = start_idx + 1
            while j < len(messages):
                if j not in kept:
                    j += 1
                    continue

                role = messages[j].raw_message.get("role")

                if role == "tool":
                    kept.discard(j)
                    j += 1
                    continue

                if role == "assistant":
                    # Remove the assistant that follows the tool chain and stop.
                    kept.discard(j)
                    break

                # Any other role stops the cascade.
                break

        # Remove the oldest non-system messages but keep the last non-system.
        for idx in non_system_indices[:-1]:
            if idx not in kept:
                continue

            raw = messages[idx].raw_message
            kept.discard(idx)

            if raw.get("role") == "assistant" and raw.get("tool_calls"):
                _cascade_remove_tool_replies(idx)

            prompt_tokens = await self.tokenize_request(
                original_request, _collect(kept)
            )
            if prompt_tokens <= max_prompt_tokens:
                discarded = sorted(all_indices - kept)
                return _collect(kept), discarded, prompt_tokens

        # Minimal viable prompt = system + last non-system
        last_non_system = non_system_indices[-1]
        last_kept = set(system_indices) | {last_non_system}

        last_tokens = await self.tokenize_request(
            original_request, _collect(last_kept)
        )
        if last_tokens <= max_prompt_tokens:
            discarded = sorted(all_indices - last_kept)
            return _collect(last_kept), discarded, last_tokens

        system_tokens = await self.tokenize_request(
            original_request, _collect(system_set)
        )
        if system_tokens > max_prompt_tokens:
            raise TruncatePromptSystemError(max_prompt_tokens, system_tokens)

        raise TruncatePromptSystemAndLastUserError(
            max_prompt_tokens, last_tokens
        )

    def _build_usage_payload(
        self, original_request: dict, messages: List[dict]
    ) -> Dict[str, Any]:
        """Build a request payload for prompt-token counting.

        To keep counting consistent with the *real* upstream request, we:
        - start from a shallow copy of the original request dict;
        - override messages with already-transformed MultiModalMessage.raw_message;
        - force a tiny non-stream completion (max_tokens=1, n=1).

        And we drop fields that are specific to streaming or otherwise not
        applicable to this internal counting call.
        """

        payload: Dict[str, Any] = dict(original_request)

        # Ensure the model matches the tokenizer model (caller usually sets it already).
        payload["model"] = self.model
        payload["messages"] = messages

        # Force a minimal non-stream completion.
        payload["stream"] = False
        payload["max_tokens"] = 1
        payload["n"] = 1

        # Ensure tools/functions are included so vLLM can account for their tokens.
        # (Some callers may move them into extra_body; we intentionally keep the
        # standard OpenAI fields.)
        if "tools" in original_request:
            payload["tools"] = original_request["tools"]
        if "functions" in original_request:
            payload["functions"] = original_request["functions"]

        # Remove fields that do not make sense for this internal call.
        payload.pop("stream_options", None)

        # extra_body is used in the adapter to pass through unsupported options.
        # For token counting we want to avoid accidental side effects.
        payload.pop("extra_body", None)

        return payload

    async def _call_completion_for_prompt_tokens(
        self, payload: Dict[str, Any]
    ) -> int:
        logger.debug(
            f"vLLM usage-token request via OpenAI client, "
            f"model={payload.get('model')}, messages_count={len(payload.get('messages', []))}"
        )

        try:
            response = await call_with_extra_body(
                self._client.chat.completions.create,
                payload,
            )
        except Exception as exc:
            # We intentionally keep error mapping simple here.
            raise InternalServerError(f"vLLM usage-token request failed: {exc}")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None

        if not isinstance(prompt_tokens, int):
            raise InternalServerError(
                "vLLM response does not contain 'usage.prompt_tokens'. "
                "Ensure vLLM returns usage for non-stream requests."
            )

        return prompt_tokens
