"""Parser tests for the keyless DuckDuckGo fallback.

The lite endpoint is scraped, so its markup is not a contract. It changed
under us once already: attributes moved to single quotes and href before
class, which the old patterns did not match, and every search quietly returned
nothing. These fixtures pin both spellings so the same silence cannot recur.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools.builtin.web_search import _parse_ddg_lite, _unwrap_ddg_redirect

# What DuckDuckGo serves today: single quotes, href before class.
CURRENT = """
<table border="0">
<tr><td valign="top">1.&nbsp;</td>
<td><a rel="nofollow" href="https://rust-lang.org/" class='result-link'>Rust Programming Language</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>A <b>language</b> empowering everyone.</td></tr>
<tr><td valign="top">2.&nbsp;</td>
<td><a rel="nofollow" href="https://doc.rust-lang.org/book/" class='result-link'>The Rust Book</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>The official book.</td></tr>
</table>
"""

# What it served before: double quotes, class before href.
LEGACY = """
<a class="result-link" href="https://example.com/one">One</a>
<td class="result-snippet">First snippet.</td>
"""


class DdgLiteParsing(unittest.TestCase):
    def test_parses_the_current_markup(self):
        out = _parse_ddg_lite(CURRENT)
        assert len(out) == 2, out
        title, url, snippet = out[0]
        assert title == "Rust Programming Language"
        assert url == "https://rust-lang.org/"
        assert snippet == "A language empowering everyone."


    def test_still_parses_the_older_markup(self):
        # Accepting both costs nothing and means a rollback on their side does not
        # become an outage on ours.
        out = _parse_ddg_lite(LEGACY)
        assert out == [("One", "https://example.com/one", "First snippet.")], out


    def test_non_http_links_are_dropped(self):
        assert _parse_ddg_lite("<a href='/settings' class='result-link'>Settings</a>") == []


    def test_redirect_wrapper_is_unwrapped(self):
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
        assert _unwrap_ddg_redirect(wrapped) == "https://example.com/page"


    def test_a_page_with_no_results_parses_to_nothing_rather_than_raising(self):
        assert _parse_ddg_lite("<html><body>no results</body></html>") == []


if __name__ == "__main__":
    unittest.main()
