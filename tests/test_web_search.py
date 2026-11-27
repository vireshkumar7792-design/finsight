"""Tavily wrapper with a fake client: caching, params, graceful failures (offline)."""

from datetime import date

from finsight.tools.web_search import WebSearch


class FakeTavily:
    def __init__(self, fail: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.fail:
            raise self.fail
        return {"results": [
            {"title": "Tesla recalls cars", "url": "https://news.example/tsla", "content": "Tesla  recalled\nvehicles.",
             "score": 0.9, "published_date": "2026-11-20"},
            {"title": "", "url": "https://blog.example/x", "content": "No title here", "score": 0.5},
            {"title": "missing url", "content": "dropped"},
        ]}


def make(tmp_path, fake, key="k"):
    return WebSearch(key, tmp_path, client=fake, today=lambda: date(2026, 11, 26))


def test_results_are_normalized(tmp_path):
    fake = FakeTavily()
    r = make(tmp_path, fake).search("tesla news", max_results=3)
    assert r.error is None and not r.from_cache
    assert [x.url for x in r.results] == ["https://news.example/tsla", "https://blog.example/x"]
    assert r.results[0].snippet == "Tesla recalled vehicles."  # whitespace collapsed
    assert r.results[0].published_date == "2026-11-20"
    assert r.results[1].title == "https://blog.example/x"  # empty title falls back to url


def test_second_identical_search_is_served_from_cache(tmp_path):
    fake = FakeTavily()
    ws = make(tmp_path, fake)
    ws.search("Tesla news", recent_days=30)
    again = ws.search("  tesla NEWS ", recent_days=30)  # same query after normalization
    assert again.from_cache and len(again.results) == 1  # undated result dropped for news
    assert len(fake.calls) == 1 and ws.api_calls == 1 and ws.cache_hits == 1
    # Cache survives a new instance (new process).
    assert make(tmp_path, FakeTavily()).search("tesla news", recent_days=30).from_cache


def test_params_basic_depth_and_news_window(tmp_path):
    fake = FakeTavily()
    ws = make(tmp_path, fake)
    ws.search("q1", max_results=2)
    ws.search("q2", recent_days=30)
    ex = {"exclude_domains": ["tavily.com"]}
    assert fake.calls[0][1] == {"max_results": 2, "search_depth": "basic", **ex}  # 1 credit
    assert fake.calls[1][1] == {"max_results": 3, "search_depth": "basic", **ex, "topic": "news",
                                "start_date": "2026-10-27"}


def test_different_params_miss_the_cache(tmp_path):
    fake = FakeTavily()
    ws = make(tmp_path, fake)
    ws.search("q", recent_days=30)
    ws.search("q", recent_days=7)
    assert len(fake.calls) == 2


def test_missing_key_is_a_graceful_gap(tmp_path):
    fake = FakeTavily()
    r = make(tmp_path, fake, key="  ").search("anything")
    assert r.results == [] and "TAVILY_API_KEY is not set" in r.error
    assert fake.calls == []


def test_quota_or_network_error_is_a_graceful_gap(tmp_path):
    class UsageLimitExceededError(Exception):
        pass

    r = make(tmp_path, FakeTavily(fail=UsageLimitExceededError("432 plan limit"))).search("anything")
    assert r.results == [] and "web search unavailable" in r.error and "432" in r.error
    assert not list(tmp_path.glob("*.json"))  # failures are not cached


def test_news_search_drops_undated_and_tavily_profile_pages(tmp_path):
    class Fake(FakeTavily):
        def search(self, query, **kwargs):
            return {"results": [
                {"title": "Real news", "url": "https://news.example/a", "content": "x", "published_date": "2026-11-20"},
                {"title": "Profile", "url": "https://tavily.com/knowledge/company/abc", "content": "x",
                 "published_date": "2026-11-20"},
                {"title": "TSLA - The Sales Leader Association", "url": "https://other.example", "content": "x"},
            ]}

    ws = make(tmp_path, Fake())
    assert [r.title for r in ws.search("tesla news", recent_days=30).results] == ["Real news"]
    assert [r.title for r in ws.search("tesla", recent_days=None).results] == [
        "Real news", "TSLA - The Sales Leader Association"]  # undated kept for non-news searches
