import dataclasses

import pytest

from attrx.config import get_settings
from attrx.llm.client import LLMClient, LLMUnavailable


def test_connection_error_is_reported_without_retry(tmp_path):
    s = dataclasses.replace(
        get_settings(), llm_enabled=True, llm_base_url="http://127.0.0.1:9/v1", llm_timeout=2, cache_path=tmp_path / "c.sqlite"
    )
    client = LLMClient(s)
    with pytest.raises(LLMUnavailable, match="нет соединения"):
        client.complete_json("sys", "user", {"type": "object"})
    assert client.calls == 1
