"""
Strata (https://github.com/Niko1221/Strata) provides the thought process
of reasoning models in the `message.reasoning_content` field of the
Chat Completions response (both streaming deltas and the final message).

The module adds the reasoning content into a dedicated DIAL Stage,
so that the thought process becomes visible to the DIAL Chat UI.
"""

from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from aidial_adapter_openai.utils.streaming import map_stream


class _ResponseTransformer(BaseModel):
    opened_reasoning_stages: set[int] = set()
    """Indices of choices where a reasoning stage was open.
    """

    streaming: bool

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices") or []
        for choice in choices:
            choice_index = choice.get("index")
            message = choice.get(self.message_key, {})
            reasoning_content = message.get("reasoning_content")

            is_ongoing = reasoning_content is not None
            is_opening = (
                choice_index not in self.opened_reasoning_stages and is_ongoing
            )
            is_closing = (
                choice_index in self.opened_reasoning_stages
                and choice.get("finish_reason") is not None
            )

            if is_opening:
                self.opened_reasoning_stages.add(choice_index)

            if is_opening or is_ongoing or is_closing:
                # Limitation: any existing "stages" are ignored,
                # whereas their indices should be offset by 1.
                # However, we don't expect any stages from the upstream.
                cc = message.setdefault("custom_content", {})
                stages = cc.setdefault("stages", [])

                opening_fields = {"name": "Reasoning"} if is_opening else {}
                closing_fields = {"status": "completed"} if is_closing else {}
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = (
                    {"content": reasoning_content} if is_ongoing else {}
                )

                stages.append(
                    {
                        **content_fields,
                        **streaming_fields,
                        **opening_fields,
                        **closing_fields,
                    }
                )

        return chunk


_T = TypeVar("_T", bound=AsyncIterator[dict] | dict)


def extract_reasoning(response: _T) -> _T:
    """Extract Strata `reasoning_content` into DIAL Stages.

    Handles both streaming (``delta.reasoning_content``) and
    non-streaming (``message.reasoning_content``) response formats.
    """
    if isinstance(response, dict):
        return _ResponseTransformer(streaming=False)(response)
    else:
        return map_stream(_ResponseTransformer(streaming=True), response)
