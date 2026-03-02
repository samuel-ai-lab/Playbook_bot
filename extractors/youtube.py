import html
import os
import re
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable


LANGUAGE_PRIORITY = ["en", "en-US", "en-GB"]


def parse_youtube_id(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
        if video_id:
            return video_id

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
            if video_id:
                return video_id

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]

    raise ValueError(f"Could not parse YouTube video ID from URL: {url}")


def _format_segments(segments: list[dict]) -> str:
    return "\n".join(segment.get("text", "").strip() for segment in segments if segment.get("text"))


def _clean_caption_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return " ".join(text.replace("\n", " ").split()).strip()


def _parse_json3_captions(payload: dict) -> str:
    lines: list[str] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        raw = "".join(seg.get("utf8", "") for seg in segs if isinstance(seg, dict))
        cleaned = _clean_caption_text(raw)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _parse_vtt_captions(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        cleaned = _clean_caption_text(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _pick_language_keys(caption_map: dict) -> list[str]:
    keys = [key for key in caption_map.keys() if isinstance(key, str)]
    ordered: list[str] = []

    for lang in LANGUAGE_PRIORITY:
        ordered.extend([key for key in keys if key == lang])
        ordered.extend([key for key in keys if key.startswith(f"{lang}-")])
        ordered.extend([key for key in keys if key.startswith(f"a.{lang}")])

    ordered.extend([key for key in keys if key.startswith("en") and key not in ordered])
    ordered.extend([key for key in keys if key not in ordered])
    return ordered


def _pick_caption_url(track_items: list[dict]) -> tuple[str, str] | None:
    preferred_exts = ["json3", "vtt", "ttml", "srv3", "srv2", "srv1"]
    for ext in preferred_exts:
        for item in track_items:
            if not isinstance(item, dict):
                continue
            if item.get("ext") == ext and isinstance(item.get("url"), str):
                return item["url"], ext
    for item in track_items:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            return item["url"], str(item.get("ext", ""))
    return None


def _extract_caption_text_from_url(url: str, ext: str) -> str:
    response = requests.get(url, timeout=45)
    response.raise_for_status()

    if ext == "json3":
        payload = response.json()
        if isinstance(payload, dict):
            return _parse_json3_captions(payload)
        return ""
    return _parse_vtt_captions(response.text)


def extract_youtube_transcript_with_ytdlp(video_url: str) -> str:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for YouTube transcript fallback.") from exc

    ydl_opts: dict = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "extractor_args": {
            "youtube": {"player_client": ["android", "web", "ios", "tv"], "formats": ["missing_pot"]}
        },
    }
    cookies_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    caption_maps = [
        info.get("subtitles") if isinstance(info, dict) else None,
        info.get("automatic_captions") if isinstance(info, dict) else None,
    ]

    for caption_map in caption_maps:
        if not isinstance(caption_map, dict):
            continue
        for lang_key in _pick_language_keys(caption_map):
            tracks = caption_map.get(lang_key)
            if not isinstance(tracks, list):
                continue
            picked = _pick_caption_url(tracks)
            if not picked:
                continue
            url, ext = picked
            try:
                text = _extract_caption_text_from_url(url, ext).strip()
            except Exception:
                continue
            if text:
                return text

    raise RuntimeError("No usable subtitle/auto-caption tracks found via yt-dlp.")


def extract_youtube_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        for lang in LANGUAGE_PRIORITY:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                return _format_segments(transcript.fetch())
            except Exception:
                pass

        for lang in LANGUAGE_PRIORITY:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                return _format_segments(transcript.fetch())
            except Exception:
                pass

        for transcript in transcript_list:
            try:
                return _format_segments(transcript.fetch())
            except Exception:
                continue

        raise RuntimeError("No usable transcripts found for this video")
    except (VideoUnavailable, TranscriptsDisabled) as exc:
        raise RuntimeError(f"YouTube transcript unavailable: {exc}") from exc
