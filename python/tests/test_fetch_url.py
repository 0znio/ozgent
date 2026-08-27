"""fetch_url tests, all offline.

The network is stubbed at `fetch_page`, so these check the parts that decide
what the model is handed: content-type dispatch, the focus filter, and whether
a refusal explains itself.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ozgent_tools import http
from ozgent_tools.base import ToolError
from ozgent_tools.builtin import fetch_url as mod


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def allow_network(monkeypatch):
    """Permissions are tested in test_permissions.py; here they are granted."""
    monkeypatch.setattr(mod, "check_host", lambda url: url.split("/")[2])


def serve(body: bytes, content_type: str, *, url: str = "https://example.com/x", status: int = 200):
    """Install a stub that answers every fetch with this one response."""
    page = http.Page(url=url, status=status, content_type=content_type, body=body, encoding="utf-8")

    async def stub(u, **kw):
        return page

    return stub


def test_html_comes_back_as_the_article(monkeypatch):
    body = b"""<html><head><title>A story</title></head><body>
      <nav><a href=/1>One</a><a href=/2>Two</a><a href=/3>Three</a></nav>
      <div class="article-body"><p>""" + b"The substance of the story. " * 12 + b"""</p></div>
    </body></html>"""
    monkeypatch.setattr(mod, "fetch_page", serve(body, "text/html; charset=utf-8"))

    got = run(mod.fetch_url("https://example.com/x"))

    assert got["title"] == "A story"
    assert "substance of the story" in got["text"]
    assert got["kind"] == "text/html"


def test_json_is_pretty_printed_not_stripped_as_markup(monkeypatch):
    payload = {"name": "llama.cpp", "stars": 12345, "nested": {"a": [1, 2, 3]}}
    monkeypatch.setattr(
        mod, "fetch_page", serve(json.dumps(payload).encode(), "application/json")
    )

    got = run(mod.fetch_url("https://api.example.com/repo"))

    assert got["strategy"] == "json"
    assert json.loads(got["text"]) == payload, "must round-trip, not be mangled"


def test_a_json_content_type_that_is_not_json_falls_back_to_text(monkeypatch):
    monkeypatch.setattr(mod, "fetch_page", serve(b"<!doctype html>oops", "application/json"))
    got = run(mod.fetch_url("https://example.com/x"))
    assert "oops" in got["text"]


def test_plain_text_is_left_exactly_alone(monkeypatch):
    monkeypatch.setattr(mod, "fetch_page", serve(b"# Title\n\n- a\n- b\n", "text/markdown"))
    got = run(mod.fetch_url("https://example.com/readme.md"))
    assert got["text"] == "# Title\n\n- a\n- b\n"
    assert got["strategy"] == "text"


def test_xml_feeds_give_up_their_entries(monkeypatch):
    feed = b"""<?xml version="1.0"?><rss><channel>
      <item><title>First headline</title><description>What happened</description></item>
      <item><title>Second headline</title><description>What else</description></item>
    </channel></rss>"""
    monkeypatch.setattr(mod, "fetch_page", serve(feed, "application/rss+xml"))

    got = run(mod.fetch_url("https://example.com/feed"))
    assert "First headline" in got["text"]
    assert "Second headline" in got["text"]


def test_a_binary_type_is_refused_with_its_type_named(monkeypatch):
    monkeypatch.setattr(mod, "fetch_page", serve(b"\x00\x01\x02", "image/png"))
    with pytest.raises(ToolError) as exc:
        run(mod.fetch_url("https://example.com/x.png"))
    assert "image/png" in str(exc.value)


def test_a_pdf_without_a_reader_says_what_to_install(monkeypatch):
    monkeypatch.setattr(mod, "fetch_page", serve(b"%PDF-1.4 binary junk", "application/pdf"))
    pytest.importorskip
    try:
        import pypdf  # noqa: F401
        pytest.skip("a PDF reader is installed, so this path is not taken")
    except ImportError:
        pass

    with pytest.raises(ToolError) as exc:
        run(mod.fetch_url("https://example.com/paper.pdf"))
    assert "pypdf" in str(exc.value)


def test_an_empty_page_is_reported_as_probably_javascript(monkeypatch):
    monkeypatch.setattr(
        mod, "fetch_page", serve(b"<html><body><div id=root></div></body></html>", "text/html")
    )
    with pytest.raises(ToolError) as exc:
        run(mod.fetch_url("https://example.com/app"))
    assert "JavaScript" in str(exc.value)


@pytest.mark.parametrize(
    "status,expect",
    [
        (401, "blocks automated readers"),
        (403, "blocks automated readers"),
        (404, "no page at that address"),
        (429, "rate limiting"),
        (503, "having trouble"),
    ],
)
def test_every_refusal_explains_itself(monkeypatch, status, expect):
    # The old message was "could not fetch <url>" for all of these, which told
    # the model nothing it could act on.
    async def stub(u, **kw):
        raise http.HttpError(status, f"HTTP {status}")

    monkeypatch.setattr(mod, "fetch_page", stub)
    with pytest.raises(ToolError) as exc:
        run(mod.fetch_url("https://blocked.example.com/x"))

    message = str(exc.value)
    assert expect in message, message
    assert "blocked.example.com" in message, "the host must be named"


def test_a_rate_limit_is_retryable_and_a_404_is_not(monkeypatch):
    def stub_for(status):
        async def stub(u, **kw):
            raise http.HttpError(status, f"HTTP {status}")
        return stub

    monkeypatch.setattr(mod, "fetch_page", stub_for(429))
    with pytest.raises(ToolError) as limited:
        run(mod.fetch_url("https://example.com/x"))
    assert limited.value.retryable

    monkeypatch.setattr(mod, "fetch_page", stub_for(404))
    with pytest.raises(ToolError) as missing:
        run(mod.fetch_url("https://example.com/x"))
    assert not missing.value.retryable, "retrying a 404 only wastes a round"


def test_query_keeps_the_matching_paragraphs(monkeypatch):
    # Long enough that trimming is actually called for; a page that already
    # fits is left alone, which the next test covers.
    paragraphs = [
        "Filler about the weather in an unrelated region. " * 120,
        "The quarterly dividend was raised to twelve rupees per share. " * 120,
        "More filler concerning transport schedules and roadworks. " * 120,
    ]
    body = ("<html><body><div class=content>"
            + "".join(f"<p>{p}</p>" for p in paragraphs)
            + "</div></body></html>").encode()
    monkeypatch.setattr(mod, "fetch_page", serve(body, "text/html"))

    got = run(mod.fetch_url("https://example.com/x", query="dividend per share"))

    assert "dividend" in got["text"]
    assert got["focused_on"] == "dividend per share"
    assert "roadworks" not in got["text"], "unmatched paragraphs are dropped"


def test_focus_works_when_blocks_are_separated_by_single_newlines(monkeypatch):
    # The shape the extractor actually produces: one newline between blocks,
    # not two. Splitting only on blank lines saw one paragraph and trimmed
    # nothing, which silently defeated `query` on every real page.
    from ozgent_tools.builtin.fetch_url import _focus

    text = "\n".join(
        [
            "Weather and roadworks and unrelated filler. " * 40,
            "The dividend was raised this quarter. " * 40,
        ]
    )
    got = _focus(text, "dividend")
    assert "dividend" in got.lower()
    assert "roadworks" not in got


def test_query_on_a_short_page_changes_nothing(monkeypatch):
    body = b"<html><body><div class=content><p>" + b"Short enough. " * 20 + b"</p></div></body></html>"
    monkeypatch.setattr(mod, "fetch_page", serve(body, "text/html"))

    got = run(mod.fetch_url("https://example.com/x", query="anything"))
    assert "focused_on" not in got, "trimming a page that already fits helps nobody"


def test_a_query_matching_nothing_returns_the_page_rather_than_nothing(monkeypatch):
    text = "\n\n".join("Paragraph about steel production. " * 30 for _ in range(4))
    body = f"<html><body><div class=content><p>{text}</p></div></body></html>".encode()
    monkeypatch.setattr(mod, "fetch_page", serve(body, "text/html"))

    got = run(mod.fetch_url("https://example.com/x", query="zzz qqq xxx"))
    assert "steel production" in got["text"]


def test_the_final_url_after_redirects_is_the_one_reported(monkeypatch):
    body = b"<html><body><div class=content><p>" + b"Landed here. " * 30 + b"</p></div></body></html>"
    monkeypatch.setattr(
        mod, "fetch_page", serve(body, "text/html", url="https://example.com/final")
    )
    got = run(mod.fetch_url("https://example.com/start"))
    assert got["url"] == "https://example.com/final"


def test_output_is_capped(monkeypatch):
    body = ("<html><body><div class=content><p>"
            + "word " * 200_000 + "</p></div></body></html>").encode()
    monkeypatch.setattr(mod, "fetch_page", serve(body, "text/html"))

    got = run(mod.fetch_url("https://example.com/x"))
    assert len(got["text"]) <= mod.MAX_PAGE
    assert got["truncated"] is True
