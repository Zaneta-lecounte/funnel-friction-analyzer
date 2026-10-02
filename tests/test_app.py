"""Smoke tests: the Streamlit app renders without exceptions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_sample_funnel():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Funnel Friction Analyzer"
    summary = " ".join(m.value for m in at.markdown).replace("\\", "")
    assert "A 10% lift at Landing Page → Form Start yields ~250 more form starts" in summary
    assert "Highest priority given effort." in summary


def test_app_renders_segment_tab():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown).replace("\\", "")
    assert "Segments diverge most at Form Complete → Qualified Lead" in text


def test_app_renders_benchmark_tab():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown).replace("\\", "")
    assert "Biggest benchmark gap by impact:" in text
