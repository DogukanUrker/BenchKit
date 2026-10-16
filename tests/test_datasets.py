"""Downloaded datasets: checksummed fetches, the catalog and the CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from benchkit import catalog, cli
from benchkit.benchmarks import REGISTRY, downloads
from benchkit.benchmarks.downloads import DatasetUnavailable, RemoteFile
from benchkit.engine import task_count

PAYLOAD = b'{"task_id": "x"}\n' * 100
BUNDLED = Path(downloads.__file__).parent.parent / "datasets"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHKIT_DATASET_CACHE", str(tmp_path))
    monkeypatch.setattr(downloads.time, "sleep", lambda _s: None)
    return tmp_path


def _serve(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    monkeypatch.setattr(
        downloads.httpx, "stream", lambda method, url, **_kw: client.stream(method, url)
    )
    return seen


def _remote(payload: bytes = PAYLOAD) -> RemoteFile:
    return RemoteFile(
        name="demo.jsonl",
        url="https://example.test/demo.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def test_download_is_verified_and_reported(cache, monkeypatch):
    _serve(monkeypatch, lambda _r: httpx.Response(200, content=PAYLOAD))
    progress: list[tuple[int, int | None]] = []
    remote = _remote()

    assert not remote.installed()
    path = remote.ensure(lambda done, total: progress.append((done, total)))

    assert path == cache / "demo.jsonl"
    assert path.read_bytes() == PAYLOAD
    assert remote.installed()
    assert progress[-1] == (len(PAYLOAD), len(PAYLOAD))


def test_cached_file_is_not_fetched_again(cache, monkeypatch):
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, content=PAYLOAD))
    remote = _remote()
    remote.ensure()
    remote.ensure()
    assert len(seen) == 1


def test_checksum_mismatch_never_reaches_the_cache(cache, monkeypatch):
    _serve(monkeypatch, lambda _r: httpx.Response(200, content=PAYLOAD))
    remote = _remote(b"something else entirely")

    with pytest.raises(DatasetUnavailable, match="checksum mismatch"):
        remote.ensure()
    assert list(cache.iterdir()) == []


def test_client_errors_fail_fast_with_instructions(cache, monkeypatch):
    seen = _serve(monkeypatch, lambda _r: httpx.Response(404))

    with pytest.raises(DatasetUnavailable, match="benchkit datasets install"):
        _remote().ensure()
    assert len(seen) == 1


def test_transient_errors_are_retried(cache, monkeypatch):
    responses = iter([httpx.Response(503), httpx.Response(200, content=PAYLOAD)])
    seen = _serve(monkeypatch, lambda _r: next(responses))

    assert _remote().ensure().read_bytes() == PAYLOAD
    assert len(seen) == 2


def test_heavy_suites_count_tasks_without_downloading(cache, monkeypatch):
    _serve(monkeypatch, lambda _r: pytest.fail("counting must not download"))
    assert task_count("mmlu") == 14042
    assert task_count("mmlu-pro") == 12032


def test_catalog_separates_bundled_and_downloaded(cache):
    statuses = {entry.benchmark: entry for entry in catalog.statuses()}

    assert statuses.keys() == REGISTRY.keys()
    assert statuses["sanity"].kind == "bundled" and statuses["sanity"].installed
    assert statuses["mmlu"].kind == "download" and statuses["mmlu"].installable
    assert statuses["patcheval"].kind == "manual"
    assert "mmlu" in catalog.missing()


def test_suites_sharing_a_source_install_once(cache):
    missing = catalog.missing(["ruler", "ruler-full"])
    assert len(missing) <= 1


def test_heavy_files_stay_out_of_the_package():
    assert not (BUNDLED / "mmlu.jsonl").exists()
    assert not (BUNDLED / "mmlu_pro.jsonl").exists()
    total = sum(path.stat().st_size for path in BUNDLED.iterdir())
    assert total < 8 * 1024 * 1024, f"bundled datasets grew to {total:,} bytes"


def test_cli_installs_named_datasets(cache, monkeypatch):
    _serve(monkeypatch, lambda _r: httpx.Response(200, content=PAYLOAD))
    monkeypatch.setattr(REGISTRY["mmlu"], "data_source", _remote().source("demo"))

    cli.main(["datasets", "install", "mmlu"])
    assert (cache / "demo.jsonl").exists()


def test_cli_rejects_unknown_benchmarks(cache):
    with pytest.raises(SystemExit) as exc:
        cli.main(["datasets", "install", "nope"])
    assert exc.value.code == 1
