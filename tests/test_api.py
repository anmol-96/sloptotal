"""HTTP contract tests. The lifespan (model preload, queue workers) is not
started, so only routes that do not run inference are exercised here; the
full pipeline is covered by the end-to-end run described in .github/CONTRIBUTING.md."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from app.database import create_report, init_database
from app.main import app
from uuid import uuid4


@pytest.fixture(scope="module")
def client():
    asyncio.run(init_database())
    return TestClient(app)


def test_engines_endpoint_lists_all_engines(client):
    r = client.get("/api/engines")
    assert r.status_code == 200
    body = r.json()
    engines = body["engines"] if isinstance(body, dict) else body
    assert len(engines) == 23


@pytest.mark.parametrize("path", ["/api/analyze", "/api/quick-score"])
def test_rejects_short_text(client, path):
    r = client.post(path, json={"text": "too short"})
    assert r.status_code == 400
    assert "50 characters" in r.json()["error"]


@pytest.mark.parametrize("path", ["/api/analyze", "/api/quick-score"])
def test_rejects_empty_request(client, path):
    r = client.post(path, json={})
    assert r.status_code == 400


def test_home_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SlopTotal" in r.text


def test_form_rejects_short_text(client):
    r = client.post("/analyze", data={"text": "short"})
    assert r.status_code == 200
    assert "at least 50 characters" in r.text


def test_unknown_report_is_404(client):
    assert client.get("/report/abc123").status_code == 404


def test_health(client):
    from app.engine_status import mark_failed, mark_loaded, reset_status

    reset_status()
    mark_loaded("perplexity")
    mark_loaded("classifier_tmr")
    mark_failed("classifier_superannotate", "test failure")

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["engines"]["total"] == 23
    assert r.json()["engines"]["loaded"] == 2
    assert r.json()["engines"]["failed"] == ["classifier_superannotate"]


def test_site_scan_requires_url(client):
    assert client.post("/api/scan/site", json={}).status_code == 400


def test_url_scans_refuse_private_addresses(client):
    r = client.post("/api/scan/site", json={"url": "http://127.0.0.1:8000/health"})
    assert r.status_code == 400
    assert "private" in r.json()["error"]


def test_extract_endpoint(client):
    from tests.samples import AI_TEXT

    r = client.post(
        "/api/extract", files={"file": ("essay.txt", AI_TEXT.encode(), "text/plain")}
    )
    assert r.status_code == 200
    assert r.json()["word_count"] == len(AI_TEXT.split())
    r = client.post(
        "/api/extract",
        files={"file": ("x.exe", b"MZ" * 50, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_a_report_says_which_language_it_scored(client):
    from datetime import datetime

    from app.schemas import AnalysisReport

    report = AnalysisReport(
        id="abc",
        source_type="text",
        source="x",
        text_excerpt="El ayuntamiento aprobó el plan después de varios meses de debate, y los vecinos dijeron que no se les había consultado.",
        word_count=20,
        engine_results=[],
        overall_score=10.0,
        overall_verdict="Clean",
        engines_flagged=0,
        engines_total=23,
        created_at=datetime(2026, 10, 1),
    )
    dumped = report.model_dump()
    assert dumped["language"] == "es"
    assert dumped["language_support"] in {"supported", "experimental", "unsupported"}


def test_report_shows_detected_language(client):
    report_id = uuid4().hex[:12]

    asyncio.run(
        create_report(
            report_id=report_id,
            text_hash="a" * 64,
            source_type="text",
            source="",
            text_excerpt=(
                "The quick brown fox jumps over the lazy dog. "
                "This is a simple English sentence with the words "
                "the, and, of, to, in, is, that, it, for, and was. "
            ),
            word_count=25,
            engines_total=23,
        )
    )

    response = client.get(f"/report/{report_id}")

    assert response.status_code == 200
    assert "Language:" in response.text
    assert "<b>en</b>" in response.text


def test_report_hides_unknown_language(client):
    report_id = uuid4().hex[:12]

    asyncio.run(
        create_report(
            report_id=report_id,
            text_hash="b" * 64,
            source_type="text",
            source="",
            text_excerpt="12345 !!!",
            word_count=0,
            engines_total=23,
        )
    )

    response = client.get(f"/report/{report_id}")

    assert response.status_code == 200
    assert "Language:" not in response.text
