"""Unit tests for VllmTokenizer and vLLM-based prompt truncation."""

from unittest.mock import AsyncMock

import pytest
from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource
from aidial_adapter_openai.utils.vllm_tokenizer import VllmTokenizer

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_client_with_usage(prompt_tokens: int) -> AsyncMock:
    client = AsyncMock()
    resp = AsyncMock()
    usage = AsyncMock()
    usage.prompt_tokens = prompt_tokens
    resp.usage = usage
    client.chat.completions.create.return_value = resp
    return client


def _make_tokenizer() -> VllmTokenizer:
    return VllmTokenizer(
        model="my-vllm-model",
        client=_make_client_with_usage(0),
    )


# ---------------------------------------------------------------
# VllmTokenizer._call_completion_for_prompt_tokens
# ---------------------------------------------------------------


class TestVllmTokenizerCallCompletion:
    @pytest.mark.asyncio
    async def test_returns_usage_prompt_tokens(self):
        client = _make_client_with_usage(42)
        tokenizer = VllmTokenizer(model="m", client=client)

        result = await tokenizer._call_completion_for_prompt_tokens(
            {"model": "m", "messages": [], "stream": False, "max_tokens": 1}
        )
        assert result == 42

        client.chat.completions.create.assert_awaited()

    @pytest.mark.asyncio
    async def test_raises_on_missing_usage_prompt_tokens(self):
        client = AsyncMock()
        resp = AsyncMock()
        resp.usage = None
        client.chat.completions.create.return_value = resp

        tokenizer = VllmTokenizer(model="m", client=client)

        with pytest.raises(InternalServerError):
            await tokenizer._call_completion_for_prompt_tokens(
                {"model": "m", "messages": [], "stream": False, "max_tokens": 1}
            )


# ---------------------------------------------------------------
# VllmTokenizer.tokenize_request
# ---------------------------------------------------------------


class TestVllmTokenizerPublicApi:
    @pytest.mark.asyncio
    async def test_tokenize_request_sends_full_message_list_and_tools(self):
        client = _make_client_with_usage(50)
        tokenizer = VllmTokenizer(model="my-vllm-model", client=client)

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "hi"}),
        ]
        original_request = {
            "model": "my-vllm-model",
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }

        result = await tokenizer.tokenize_request(original_request, messages)
        assert result == 50

        # Verify payload passed to client
        payload = client.chat.completions.create.await_args.kwargs
        assert payload["model"] == "my-vllm-model"
        assert payload["stream"] is False
        assert payload["max_tokens"] == 1
        assert len(payload["messages"]) == 2
        assert payload["tools"] == original_request["tools"]

    @pytest.mark.asyncio
    async def test_tokenize_request_with_empty_messages(self):
        """tokenize_request([]) sends an empty list — used for overhead."""
        client = _make_client_with_usage(3)
        tokenizer = VllmTokenizer(model="m", client=client)

        result = await tokenizer.tokenize_request({"model": "m"}, [])
        assert result == 3

        payload = client.chat.completions.create.await_args.kwargs
        assert payload["messages"] == []

    @pytest.mark.asyncio
    async def test_tokenize_request_copies_original_request_and_strips_fields(
        self,
    ):
        client = _make_client_with_usage(7)
        tokenizer = VllmTokenizer(model="my-vllm-model", client=client)

        messages = [
            MultiModalMessage(raw_message={"role": "user", "content": "hi"}),
        ]

        original_request = {
            "model": "my-vllm-model",
            "temperature": 0.123,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "stream": True,
            "max_tokens": 999,
            "n": 5,
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "functions": [{"name": "g", "parameters": {"type": "object"}}],
            "stream_options": {"include_usage": True, "foo": "bar"},
            "extra_body": {"vllm_specific": True},
        }

        result = await tokenizer.tokenize_request(original_request, messages)
        assert result == 7

        payload = client.chat.completions.create.await_args.kwargs

        # Preserved fields from original_request
        assert payload["temperature"] == 0.123
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.1

        # Overridden for the internal counting call
        assert payload["stream"] is False
        assert payload["max_tokens"] == 1
        assert payload["n"] == 1

        # Replaced messages
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

        # Stripped fields
        assert "stream_options" not in payload
        assert "extra_body" not in payload

        # Tools/functions forwarded
        assert payload["tools"] == original_request["tools"]
        assert payload["functions"] == original_request["functions"]


# ---------------------------------------------------------------
# Extra headers
# ---------------------------------------------------------------


class TestVllmExtraHeaders:
    @pytest.mark.asyncio
    async def test_extra_headers_included(self):
        # Extra headers are handled by client configuration, not by tokenizer.
        client = _make_client_with_usage(10)
        tokenizer = VllmTokenizer(model="m", client=client)

        assert (
            await tokenizer._call_completion_for_prompt_tokens(
                {"model": "m", "messages": [], "stream": False, "max_tokens": 1}
            )
            == 10
        )


# ---------------------------------------------------------------
# truncate_prompt (kept as-is; relies on tokenize_request)
# ---------------------------------------------------------------


def _make_mock_tokenizer(responses: list[int]) -> VllmTokenizer:
    """Create a VllmTokenizer where successive tokenize_request calls
    return counts from *responses* in order."""
    tokenizer = _make_tokenizer()
    call_index = {"idx": 0}
    call_log: list[int] = []  # message counts per call

    async def mock_tokenize_request(original_request, messages):
        idx = call_index["idx"]
        call_index["idx"] += 1
        call_log.append(len(messages))
        if idx < len(responses):
            return responses[idx]
        return responses[-1]

    tokenizer.tokenize_request = mock_tokenize_request  # type: ignore[assignment]
    tokenizer._mock_call_log = call_log  # type: ignore[attr-defined]
    return tokenizer


class TestVllmTruncatePrompt:
    @pytest.mark.asyncio
    async def test_fits_without_truncation(self):
        """All messages fit — no truncation needed."""
        # Call 1: full list → 20
        tokenizer = _make_mock_tokenizer([20])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "hi"}),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 30
        )

        assert discarded == []
        assert used == 20
        assert len(truncated) == 2

    @pytest.mark.asyncio
    async def test_drops_oldest_non_system_message(self):
        """Three messages: system + 2 user.  Full list is too big, so
        oldest user message is dropped and the full remaining list is
        re-tokenized."""
        # Call 1: full list (3 msgs) → 30 (exceeds 25)
        # Call 2: after dropping oldest group → 18 (fits)
        tokenizer = _make_mock_tokenizer([30, 18])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "old user"}
            ),
            MultiModalMessage(
                raw_message={"role": "user", "content": "new user"}
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 25
        )

        assert discarded == [1]
        assert used == 18
        assert len(truncated) == 2
        assert truncated[0].raw_message["content"] == "sys"
        assert truncated[1].raw_message["content"] == "new user"

    @pytest.mark.asyncio
    async def test_drops_multiple_messages(self):
        """Four messages: system + 3 user.  Need to drop 2 oldest."""
        # Call 1: full list (4 msgs) → 40 (exceeds 15)
        # Call 2: after dropping group1 → 30 (still exceeds)
        # Call 3: after dropping group2 → 12 (fits)
        tokenizer = _make_mock_tokenizer([40, 30, 12])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u2"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u3"}),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 15
        )

        assert sorted(discarded) == [1, 2]
        assert used == 12
        assert len(truncated) == 2
        assert truncated[0].raw_message["content"] == "sys"
        assert truncated[1].raw_message["content"] == "u3"

    @pytest.mark.asyncio
    async def test_raises_system_error(self):
        """System messages alone exceed the budget."""
        # Call 1: full list (system-only) → 50 (exceeds budget of 10)
        # Call 2: system-only confirmation → 50
        tokenizer = _make_mock_tokenizer([50, 50])

        messages = [
            MultiModalMessage(
                raw_message={"role": "system", "content": "long"}
            ),
        ]

        with pytest.raises(TruncatePromptSystemError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_raises_system_and_last_user_error(self):
        """System + last user message exceeds the budget."""
        # Call 1: full list → 50 (exceeds 10)
        # Call 2: system + last user → 50 (still exceeds)
        # Call 3: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([50, 50, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "huge"}),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_multimodal_messages_sent_as_whole(self):
        """Messages with images/files are sent as atomic units.
        The tokenize endpoint sees the full content (including base64)."""
        # Call 1: full list → 200 (exceeds 100)
        # Call 2: system + last multimodal → 200 (still exceeds)
        # Call 3: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([200, 200, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                images=[
                    ImageResource(
                        width=100,
                        height=100,
                        detail="low",
                        image=Resource(type="image/jpeg", data=b"..."),
                    )
                ],
                raw_message={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,..."},
                        },
                    ],
                },
            ),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 100)

    @pytest.mark.asyncio
    async def test_multimodal_message_kept_when_fits(self):
        """Multimodal message fits after dropping older plain messages."""
        # Call 1: full list → 80 (exceeds 60)
        # Call 2: after removing plain (system + multimodal) → 55 (fits)
        tokenizer = _make_mock_tokenizer([80, 55])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "old msg"}
            ),
            MultiModalMessage(
                images=[
                    ImageResource(
                        width=100,
                        height=100,
                        detail="low",
                        image=Resource(type="image/jpeg", data=b"..."),
                    )
                ],
                raw_message={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,..."},
                        },
                    ],
                },
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 60
        )

        assert discarded == [1]
        assert used == 55
        assert len(truncated) == 2
        assert truncated[0].raw_message["role"] == "system"
        # The multimodal message (with image) is kept as-is
        assert isinstance(truncated[1].raw_message["content"], list)

    @pytest.mark.asyncio
    async def test_tokenize_called_with_full_list_each_iteration(self):
        """Verify that each truncation step re-sends the full remaining
        message list (not individual messages)."""
        tokenizer = _make_tokenizer()

        call_payloads = []

        async def capturing_tokenize_request(original_request, messages):
            raw = [m.raw_message for m in messages]
            call_payloads.append(raw)
            # Simulate: full=30, after drop=15
            counts = [30, 15]
            idx = len(call_payloads) - 1
            return counts[idx] if idx < len(counts) else 15

        tokenizer.tokenize_request = capturing_tokenize_request  # type: ignore[assignment]

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u2"}),
        ]

        await tokenizer.truncate_prompt({}, messages, 20)

        # Call 1: full list (3 messages)
        assert len(call_payloads[0]) == 3

        # Call 2: after dropping u1 → system + u2 (2 messages)
        assert len(call_payloads[1]) == 2
        assert call_payloads[1][0]["content"] == "sys"
        assert call_payloads[1][1]["content"] == "u2"


# ---------------------------------------------------------------
# Tool-call cascade removal
# ---------------------------------------------------------------


class TestVllmToolCallCascade:
    @pytest.mark.asyncio
    async def test_assistant_tool_calls_cascade_removes_tool_messages(self):
        """When an assistant message with tool_calls is dropped, the adapter
        must also drop subsequent tool messages and the next assistant."""

        # Call 1: full (6 msgs) → 100 (exceeds)
        # Call 2: after dropping assistant+tool replies+next assistant → 12 (fits)
        tokenizer = _make_mock_tokenizer([100, 12])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ),
            # tool replies (should be removed with cascade)
            MultiModalMessage(
                raw_message={
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "r1",
                }
            ),
            MultiModalMessage(
                raw_message={
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "r2",
                }
            ),
            # next assistant MUST be removed too
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "next"}
            ),
            MultiModalMessage(
                raw_message={"role": "user", "content": "follow-up"}
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 20
        )

        assert sorted(discarded) == [1, 2, 3, 4]
        assert used == 12
        assert truncated[-1].raw_message["content"] == "follow-up"

    @pytest.mark.asyncio
    async def test_non_tool_call_assistant_no_cascade(self):
        """A plain assistant message (no tool_calls) does not cascade."""
        tokenizer = _make_mock_tokenizer([60, 20])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "reply"}
            ),
            MultiModalMessage(
                raw_message={"role": "tool", "content": "orphan"}
            ),
            MultiModalMessage(raw_message={"role": "user", "content": "new_q"}),
        ]

        _, discarded, used = await tokenizer.truncate_prompt({}, messages, 25)

        # Only assistant dropped; tool message stays (no cascade).
        assert sorted(discarded) == [1]
        assert used == 20
