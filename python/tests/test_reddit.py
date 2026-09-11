"""Parsing tests for the reddit tool.

The feed route is the one every install without credentials uses, and a feed
is not a contract, so its shape is pinned here from a real response.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools.base import ToolError
from ozgent_tools.builtin.reddit import (
    _subreddit, feed_posts, parse_feed, post_id, shape_api_comments, shape_api_post,
)

FEED = b"""<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><author><name>/u/Patriot_tech</name></author><category term="stocks" label="r/stocks"/>
<content type="html">&lt;div class=&quot;md&quot;&gt;&lt;p&gt;Guidance around data center growth caught my attention.&lt;/p&gt;&lt;/div&gt; submitted by /u/Patriot_tech [link] [comments]</content>
<id>t3_1wd09hn</id><link href="https://www.reddit.com/r/stocks/comments/1wd09hn/nvda_post/"/>
<published>2026-09-11T00:04:11+00:00</published><title>NVDA post-earnings</title></entry>
<entry><author><name>/u/someone</name></author><category term="stocks" label="r/stocks"/>
<content type="html">&lt;p&gt;Priced in.&lt;/p&gt;</content><id>t1_abc123</id>
<link href="https://www.reddit.com/r/stocks/comments/1wd09hn/nvda_post/abc123/"/>
<updated>2026-09-11T01:00:00+00:00</updated><title>/u/someone on NVDA post-earnings</title></entry>
</feed>"""


class Feeds(unittest.TestCase):
    def test_posts_and_comments_are_told_apart(self):
        entries = parse_feed(FEED)
        assert [e["kind"] for e in entries] == ["post", "comment"]
        assert entries[1]["text"] == "Priced in."
        assert entries[0]["author"] == "Patriot_tech" and entries[0]["subreddit"] == "stocks"

    def test_the_feeds_own_chrome_is_not_part_of_the_post(self):
        post = feed_posts(parse_feed(FEED))[0]
        assert post["snippet"] == "Guidance around data center growth caught my attention."
        assert post["id"] == "1wd09hn"

    def test_a_page_instead_of_a_feed_is_explained(self):
        with self.assertRaises(ToolError) as caught:
            parse_feed(b"<!DOCTYPE html><html><body>blocked</body>")
        assert "blocking anonymous access" in str(caught.exception)


class Api(unittest.TestCase):
    def test_a_post_carries_its_score_and_permalink(self):
        p = shape_api_post({"id": "x1", "title": "t", "score": 42, "num_comments": 7,
                            "permalink": "/r/stocks/comments/x1/t/", "is_self": True,
                            "selftext": "body", "created_utc": 1757534400})
        assert p["score"] == 42 and p["comments"] == 7 and p["link"] is None
        assert p["url"] == "https://www.reddit.com/r/stocks/comments/x1/t/"

    def test_comments_skip_placeholders_and_removed_ones(self):
        listing = [
            {"data": {"children": [{"data": {"id": "x1", "title": "t", "permalink": "/p/"}}]}},
            {"data": {"children": [
                {"kind": "t1", "data": {"body": "[removed]"}},
                {"kind": "more", "data": {}},
                {"kind": "t1", "data": {"body": "good point", "score": 3, "permalink": "/c/"}},
            ]}},
        ]
        out = shape_api_comments(listing, 10)
        assert [c["text"] for c in out["comments"]] == ["good point"]


class References(unittest.TestCase):
    def test_a_post_can_be_named_by_url_fullname_or_id(self):
        assert post_id("https://www.reddit.com/r/stocks/comments/1wd09hn/title/") == "1wd09hn"
        assert post_id("t3_1wd09hn") == "1wd09hn"
        assert post_id("1WD09HN") == "1wd09hn"
        with self.assertRaises(ToolError):
            post_id("not a post!")

    def test_subreddit_names_are_normalised_and_checked(self):
        assert _subreddit("r/wallstreetbets") == "wallstreetbets"
        assert _subreddit("/r/stocks/") == "stocks"
        with self.assertRaises(ToolError):
            _subreddit("a b")


if __name__ == "__main__":
    unittest.main()
