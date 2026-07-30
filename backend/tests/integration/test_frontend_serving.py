"""Production frontend serving stays same-origin without swallowing API failures."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from shrimp_screening.main import create_app
from shrimp_screening.settings import Settings


def _app(frontend_dir: Path) -> object:
    return create_app(
        Settings(env="test", provider="unavailable"),
        frontend_dir=frontend_dir,
    )


def test_built_frontend_and_assets_are_served_from_the_api_origin(tmp_path: Path) -> None:
    frontend = tmp_path / "dist"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (frontend / "index.html").write_text(
        '<!doctype html><div id="root"></div><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("globalThis.__LOCAL_BUILD__ = true;", encoding="utf-8")

    with TestClient(_app(frontend)) as client:  # type: ignore[arg-type]
        root = client.get("/")
        browser_route = client.get("/screen/new")
        asset = client.get("/assets/app.js")

    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in root.text
    assert browser_route.status_code == 200
    assert browser_route.text == root.text
    assert asset.status_code == 200
    assert asset.text == "globalThis.__LOCAL_BUILD__ = true;"


def test_spa_fallback_never_masks_api_or_health_404s(tmp_path: Path) -> None:
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><p>SPA</p>", encoding="utf-8")

    with TestClient(_app(frontend), raise_server_exceptions=False) as client:  # type: ignore[arg-type]
        for path in ("/api/v1/not-real", "/api/not-real", "/livez/not-real", "/readyz/nope"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert response.headers["content-type"].startswith("application/problem+json"), path
            assert response.json()["code"] == "NOT_FOUND", path
            assert "SPA" not in response.text, path


def test_missing_frontend_build_leaves_root_as_problem_404(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path / "missing")) as client:  # type: ignore[arg-type]
        response = client.get("/")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "NOT_FOUND"
