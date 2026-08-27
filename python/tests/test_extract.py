"""Extraction tests, all offline.

The failure these guard against is not a crash. It is a page that extracts
"successfully" into its own navigation menu, which reads to the model as the
answer and to the user as the tool being broken.
"""

from __future__ import annotations

from ozgent_tools.extract import extract, strip_tags

ARTICLE = """
<html><head><title>Steel prices rise</title>
<meta property="og:description" content="A short summary.">
</head><body>
<nav><a href="/a">Home</a><a href="/b">Markets</a><a href="/c">Sport</a>
     <a href="/d">Weather</a><a href="/e">More</a><a href="/f">Even more</a></nav>
<header><a href="/">Masthead</a></header>
<div class="content-body">
  <p>Steel prices rose sharply this quarter, driven by demand from the
     construction sector and a shortage of coking coal across the region.</p>
  <p>Analysts expect the trend to continue into next year, though several
     cautioned that the underlying demand picture remains uneven.</p>
  <p>The mills themselves have been reluctant to commit to new capacity,
     citing the cost of energy and an uncertain regulatory environment.</p>
</div>
<aside class="related"><a href="/x">Related story one</a><a href="/y">Related two</a></aside>
<footer><a href="/tos">Terms</a><a href="/privacy">Privacy</a></footer>
</body></html>
"""


def test_the_article_is_found_and_the_chrome_is_not():
    got = extract(ARTICLE)
    text = got["text"]

    assert "Steel prices rose sharply" in text
    assert "coking coal" in text
    for chrome in ("Weather", "Masthead", "Related story one", "Privacy"):
        assert chrome not in text, f"{chrome!r} is navigation, not content"


def test_the_title_and_description_come_back():
    got = extract(ARTICLE)
    assert got["title"] == "Steel prices rise"
    assert got["description"] == "A short summary."


def test_json_ld_wins_because_it_is_the_article_verbatim():
    page = """
    <html><head><title>T</title>
    <script type="application/ld+json">
    {"@type":"NewsArticle","articleBody":"%s"}
    </script></head>
    <body><nav>menu menu menu</nav><div><p>truncated preview</p></div></body></html>
    """ % ("The full article body as the publisher wrote it. " * 8)

    got = extract(page)
    assert got["strategy"] == "json-ld"
    assert "as the publisher wrote it" in got["text"]
    assert "menu" not in got["text"]


def test_json_ld_inside_a_graph_is_found():
    body = "Body text that is comfortably longer than the two hundred character floor. " * 4
    page = """
    <html><head><script type="application/ld+json">
    {"@context":"https://schema.org","@graph":[{"@type":"WebPage"},
     {"@type":"Article","articleBody":"%s"}]}
    </script></head><body><p>x</p></body></html>
    """ % body

    assert "Body text" in extract(page)["text"]


def test_broken_json_ld_does_not_stop_extraction():
    page = """
    <html><head><title>T</title>
    <script type="application/ld+json">{ this is not json </script></head>
    <body><div class="article"><p>%s</p></div></body></html>
    """ % ("Real prose that should still be found regardless. " * 6)

    got = extract(page)
    assert "Real prose" in got["text"]
    assert got["strategy"] != "json-ld"


def test_a_link_heavy_block_never_beats_prose():
    # The exact shape of a sidebar: plenty of characters, nearly all of them
    # anchor text.
    links = "".join(f'<a href="/{i}">Some reasonably long link title {i}</a>' for i in range(40))
    page = f"""
    <html><body>
      <div class="sidebar-links">{links}</div>
      <div class="story"><p>{"Actual prose in a paragraph. " * 12}</p></div>
    </body></html>
    """
    text = extract(page)["text"]
    assert "Actual prose" in text
    assert "Some reasonably long link title 7" not in text


def test_unclosed_tags_do_not_lose_the_content():
    # Real pages close tags they never opened, and vice versa.
    page = (
        "<html><body><div class='content'><p>First paragraph of the story here."
        "<p>Second paragraph continues without closing the first properly."
        "</div></body>"
    )
    text = extract(page)["text"]
    assert "First paragraph" in text
    assert "Second paragraph" in text


def test_a_page_with_no_markup_survives():
    got = extract("Just some plain text, no tags at all, nothing to parse.")
    assert "plain text" in got["text"]


def test_an_empty_page_is_empty_not_an_error():
    assert extract("")["text"] == ""
    assert extract("<html><body></body></html>")["text"] == ""


def test_scripts_and_styles_never_reach_the_output():
    page = """
    <html><body>
    <script>var tracking = "should not appear"; window.x = 1;</script>
    <style>.cls { color: red; content: "invisible"; }</style>
    <div class="content"><p>%s</p></div>
    </body></html>
    """ % ("The visible words of the article. " * 10)

    text = extract(page)["text"]
    assert "tracking" not in text
    assert "invisible" not in text
    assert "visible words" in text


def test_nested_dropped_elements_stay_dropped():
    # <nav> containing a <div> containing a <nav>: the outer one must not be
    # un-muted by the inner one closing.
    page = """
    <html><body>
      <nav>outer nav <div>middle <nav>inner nav</nav> still nav</div> end nav</nav>
      <div class="content"><p>%s</p></div>
    </body></html>
    """ % ("The story itself goes here and is long enough to win. " * 8)

    text = extract(page)["text"]
    assert "nav" not in text.lower()
    assert "The story itself" in text


def test_entities_are_decoded():
    page = (
        "<html><body><div class='content'><p>"
        "Tom &amp; Jerry &mdash; &quot;quoted&quot; &lt;tag&gt; " * 12
        + "</p></div></body></html>"
    )
    text = extract(page)["text"]
    assert "Tom & Jerry" in text
    assert "&amp;" not in text


def test_strip_tags_is_the_floor_not_the_plan():
    assert "hello" in strip_tags("<p>hello</p>")
    assert "<p>" not in strip_tags("<p>hello</p>")
    assert strip_tags("no tags here") == "no tags here"
