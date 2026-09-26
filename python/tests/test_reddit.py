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
    _subreddit, feed_posts, parse_comments, parse_feed, parse_listing, parse_search, post_id,
    shape_api_comments, shape_api_post,
)

# Trimmed from real fragments, 2026-09-26: the attributes and nesting that
# the parsers read, and nothing else.
LISTING = """<shreddit-post permalink="/r/LocalLLaMA/comments/1wqcrly/ling_tiny/"
 content-href="https://www.reddit.com/r/LocalLLaMA/comments/1wqcrly/ling_tiny/" comment-count="38"
 created-timestamp="2026-09-26T00:40:52.000000+0000" id="t3_1wqcrly" post-title="Ling Tiny &amp; friends"
 score="120" upvote-ratio="0.9545" author="netherreddit" subreddit-name="LocalLLaMA">
 <div slot="text-body"><div class="md"><p>An 8B model</p><p>that <b>surprised</b> me.</p><img src="x"></div></div>
 <shreddit-post-overflow-menu></shreddit-post-overflow-menu>
</shreddit-post>
<shreddit-post id="t3_abc" post-title="A link" permalink="/r/x/comments/abc/a_link/"
 content-href="https://example.com/article" score="5" comment-count="1" subreddit-name="x"></shreddit-post>"""

COMMENTS = """<shreddit-comment author="alice" score="7" depth="0" created="2026-09-26T01:57:40.879000+0000"
 permalink="/r/LocalLLaMA/comments/1wqcrly/comment/c1/">
 <div slot="comment"><div class="md"><p>First point.</p><p>Second.</p></div></div>
 <shreddit-comment author="bob" score="2" depth="1" permalink="/r/LocalLLaMA/comments/1wqcrly/comment/c2/">
  <div slot="comment"><p>A reply<br>on two lines</p></div>
 </shreddit-comment>
</shreddit-comment>
<shreddit-comment author="[deleted]" score="1" depth="0"><div slot="comment"><p>[deleted]</p></div></shreddit-comment>"""

SEARCH = """<search-telemetry-tracker data-faceplate-tracking-context="{&quot;action_info&quot;:{&quot;type&quot;:&quot;post&quot;},
&quot;post&quot;:{&quot;id&quot;:&quot;t3_1wmucw0&quot;,&quot;title&quot;:&quot;Qwen at 128K&quot;},&quot;profile&quot;:{&quot;name&quot;:&quot;tm&quot;},
&quot;search&quot;:{&quot;snippet&quot;:&quot;Running it on one card.&quot;},&quot;subreddit&quot;:{&quot;name&quot;:&quot;LocalLLM&quot;}}">
<div data-testid="sdui-post-unit"><faceplate-timeago ts="2026-09-22T00:24:46.805000+0000"></faceplate-timeago>
<div data-testid="search-counter-row"><faceplate-number number="95"></faceplate-number> votes ·
<faceplate-number number="71"></faceplate-number> comments</div></div></search-telemetry-tracker>
<search-telemetry-tracker data-faceplate-tracking-context="{&quot;action_info&quot;:{&quot;type&quot;:&quot;snippet&quot;},&quot;post&quot;:{&quot;id&quot;:&quot;t3_1wmucw0&quot;}}"></search-telemetry-tracker>
<faceplate-partial loading="lazy" src="/svc/shreddit/search/?q=qwen&amp;type=posts&amp;cursor=abc"></faceplate-partial>"""

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


class WebFragmentsTest(unittest.TestCase):
    def test_a_listing_gives_posts_with_scores_and_bodies(self):
        posts = parse_listing(LISTING)
        self.assertEqual([p["id"] for p in posts], ["1wqcrly", "abc"])
        first = posts[0]
        self.assertEqual(first["title"], "Ling Tiny & friends")
        self.assertEqual((first["score"], first["comments"], first["upvote_ratio"]), (120, 38, 0.95))
        self.assertEqual(first["created"], "2026-09-26T00:40:52Z")
        self.assertEqual(first["snippet"], "An 8B model that surprised me.")
        self.assertIsNone(first["link"], "a text post links to nothing else")
        self.assertEqual(posts[1]["link"], "https://example.com/article")

    def test_comments_keep_their_nesting_and_drop_the_deleted(self):
        comments = parse_comments(COMMENTS)
        self.assertEqual([(c["author"], c["depth"]) for c in comments], [("alice", 0), ("bob", 1)])
        self.assertEqual(comments[0]["text"], "First point. Second.")
        self.assertEqual(comments[1]["text"], "A reply on two lines")
        self.assertEqual(comments[0]["score"], 7)

    def test_search_results_carry_counts_and_the_next_page(self):
        posts, more = parse_search(SEARCH)
        self.assertEqual(len(posts), 1, "the snippet's own tracker is not a second result")
        p = posts[0]
        self.assertEqual((p["id"], p["title"], p["subreddit"], p["author"]), ("1wmucw0", "Qwen at 128K", "LocalLLM", "tm"))
        self.assertEqual((p["score"], p["comments"]), (95, 71))
        self.assertEqual(p["created"], "2026-09-22T00:24:46Z")
        self.assertEqual(p["url"], "https://www.reddit.com/r/LocalLLM/comments/1wmucw0/")
        self.assertEqual(more, "/svc/shreddit/search/?q=qwen&type=posts&cursor=abc")


if __name__ == "__main__":
    unittest.main()
