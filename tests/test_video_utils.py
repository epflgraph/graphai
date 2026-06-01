from urllib.error import HTTPError

from graphai.core.video.video_utils import (
    _extract_useful_ffmpeg_error_text,
    is_kaltura_manifest_url,
    retrieve_file_from_any_source,
    retrieve_video_file_from_generic_url,
)


def test_is_kaltura_manifest_url_detects_playlist_urls():
    assert is_kaltura_manifest_url("https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_08e19b90")
    assert is_kaltura_manifest_url("https://example.org/video/master.m3u8")


def test_is_kaltura_manifest_url_skips_direct_mp4_urls():
    assert not is_kaltura_manifest_url(
        "https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_08e19b90/v/2/ev/3/flavorId/0_yhre7ze7/fileName/video.mp4/forceproxy/true/name/a.mp4"
    )


def test_retrieve_file_from_any_source_uses_generic_for_direct_mp4(monkeypatch):
    calls = []

    def fake_generic(url, output_filename_with_path, output_token):
        calls.append((url, output_filename_with_path, output_token))
        return output_token, "fpid"

    def fail_kaltura(*_args, **_kwargs):
        raise AssertionError("kaltura handler should not be used for direct mp4")

    monkeypatch.setattr("graphai.core.video.video_utils.retrieve_video_file_from_generic_url", fake_generic)
    monkeypatch.setattr("graphai.core.video.video_utils.retrieve_file_from_kaltura", fail_kaltura)

    result = retrieve_file_from_any_source(
        "https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_08e19b90/v/2/ev/3/flavorId/0_yhre7ze7/fileName/video.mp4/forceproxy/true/name/a.mp4",
        "/tmp/out.mp4",
        "token.mp4",
        is_kaltura=False,
    )

    assert result[:2] == ("token.mp4", "fpid")
    assert len(calls) == 1


def test_retrieve_video_file_from_generic_url_retries_kaltura_on_404(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_x/v/2/ev/7/flavorId/0_y/fileName/video.mp4/forceproxy/true/name/a.mp4",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("graphai.core.video.video_utils.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "graphai.core.video.video_utils.retrieve_file_from_kaltura",
        lambda *_args, **_kwargs: ("token.mp4", "entry"),
    )

    result = retrieve_video_file_from_generic_url(
        "https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_x/v/2/ev/7/flavorId/0_y/fileName/video.mp4/forceproxy/true/name/a.mp4",
        "/tmp/out.mp4",
        "token.mp4",
    )

    assert result[:2] == ("token.mp4", "entry")


def test_extract_useful_ffmpeg_error_text_keeps_tail_lines():
    raw = """
ffmpeg version x
configuration: ...
libavutil ...
[http @ 0x123] HTTP error 404 Not Found
https://api.cast.switch.ch/...: Server returned 404 Not Found
"""
    useful = _extract_useful_ffmpeg_error_text(raw)
    assert "404" in useful
    assert "Server returned 404" in useful


def test_retrieve_video_file_from_generic_url_non_404_returns_failure_reason(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://example.org/not-found.mp4",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("graphai.core.video.video_utils.urlopen", fake_urlopen)

    token, fp_id, failure_reason = retrieve_video_file_from_generic_url(
        "https://example.org/not-found.mp4",
        "/tmp/out.mp4",
        "token.mp4",
    )

    assert token is None
    assert fp_id is None
    assert failure_reason == "http_403"


def test_retrieve_video_file_from_generic_url_404_chains_kaltura_failure_reason(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_x/v/2/ev/7/flavorId/0_y/fileName/video.mp4/forceproxy/true/name/a.mp4",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("graphai.core.video.video_utils.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "graphai.core.video.video_utils.retrieve_file_from_kaltura",
        lambda *_args, **_kwargs: (None, None, "kaltura_ffmpeg_failed"),
    )

    token, fp_id, failure_reason = retrieve_video_file_from_generic_url(
        "https://api.cast.switch.ch/p/113/sp/11300/serveFlavor/entryId/0_x/v/2/ev/7/flavorId/0_y/fileName/video.mp4/forceproxy/true/name/a.mp4",
        "/tmp/out.mp4",
        "token.mp4",
    )

    assert token is None
    assert fp_id is None
    assert failure_reason == "kaltura_ffmpeg_failed_after_http_404"
