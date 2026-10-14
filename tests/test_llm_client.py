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


def test_timeout_is_reported_without_retry(tmp_path):
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    s = dataclasses.replace(
        get_settings(),
        llm_enabled=True,
        llm_base_url=f"http://127.0.0.1:{srv.getsockname()[1]}/v1",
        llm_timeout=0.5,
        llm_max_retries=0,
        cache_path=tmp_path / "c.sqlite",
    )
    client = LLMClient(s)
    try:
        with pytest.raises(LLMUnavailable, match="не ответила за 0.5 с"):
            client.complete_json("sys", "user", {"type": "object"})
    finally:
        srv.close()
    assert client.calls == 1
