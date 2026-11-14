import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broll
import metadata
import timeline


def test_metadata_fallback_is_valid():
    def boom(*_):
        raise RuntimeError("no llm")
    m = metadata.generate({"title": "A long clip title", "description": "d", "hashtags": ["a"]}, "x", boom)
    assert m["source"] == "fallback" and len(m["youtube_titles"]) == 3
    assert all(len(t) < 50 for t in m["youtube_titles"])


def test_metadata_llm_titles_are_trimmed():
    m = metadata.generate({"title": "t"}, "x", lambda p, t: {"youtube_titles": ["x" * 80, "b", "c", "d"]})
    assert len(m["youtube_titles"]) == 3 and len(m["youtube_titles"][0]) < 50


def test_broll_moments_respect_spacing_and_edges():
    words = [{"word": "w", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(60)]
    llm = lambda p, t: {"items": [{"t": 0.5, "query": "x"}, {"t": 5, "query": "city"},
                                  {"t": 6, "query": "too close"}, {"t": 12, "query": "sea"}, {"t": 29, "query": "end"}]}
    items = broll.plan_moments(words, 30, llm)
    assert [i["query"] for i in items] == ["city", "sea"]
    assert all(2.5 <= i["d"] <= 3.5 for i in items)


def test_fcpxml_is_well_formed():
    xml = timeline.fcpxml("clip <1>", 20.0, "source_cut.mp4", 720, 1280,
                          [{"t": 3, "d": 3, "query": "a & b", "file": "broll_1.mp4"}])
    xml_dom = __import__("xml.dom.minidom", fromlist=["x"]).parseString(xml.encode())
    assert xml_dom.getElementsByTagName("asset-clip")
