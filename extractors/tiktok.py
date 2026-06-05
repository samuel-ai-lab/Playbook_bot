import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

from .instagram import (
    _download_media_from_url,
    _pick_media_url_from_apify_item,
    _transcribe_with_groq,
    _transcribe_with_local_whisper,
)


def _env_truthy(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}


def _env_falsey(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"0", "false", "no"}


def _normalize_tiktok_url(tiktok_url: str) -> str:
    parsed = urlparse(tiktok_url)
    if "tiktok.com" in parsed.netloc:
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return tiktok_url


def _run_apify_actor(actor_id: str, run_input: dict, timeout_seconds: int) -> list[dict]:
    token = os.getenv("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing APIFY_TOKEN for TikTok extractor provider 'apify'.")
    if not actor_id:
        raise RuntimeError("Missing TikTok Apify actor ID.")

    endpoint = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    params = {"format": "json", "clean": "1"}
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(endpoint, params=params, headers=headers, json=run_input, timeout=timeout_seconds)
    if not response.ok:
        details = response.text[:500]
        raise RuntimeError(f"Apify TikTok extractor failed ({response.status_code}): {details}")

    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Apify returned no dataset items for TikTok URL.")

    items: list[dict] = [item for item in payload if isinstance(item, dict)]
    if not items:
        raise RuntimeError("Unexpected Apify TikTok response item format.")
    return items


def _clean_webvtt_text(text: str) -> str:
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "WEBVTT":
            continue
        if re.match(r"^\d+$", line):
            continue
        if "-->" in line:
            continue
        cleaned_lines.append(line)

    merged = " ".join(cleaned_lines)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def _extract_text_values(value: object) -> list[str]:
    results: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            results.append(stripped)
        return results

    if isinstance(value, list):
        for item in value:
            results.extend(_extract_text_values(item))
        return results

    if isinstance(value, dict):
        for item in value.values():
            results.extend(_extract_text_values(item))
        return results

    return results


def _extract_transcript_from_primary_item(item: dict) -> str:
    transcript = item.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        cleaned = _clean_webvtt_text(transcript)
        if cleaned:
            return cleaned

    if isinstance(transcript, dict):
        text = transcript.get("text")
        if isinstance(text, str) and text.strip():
            cleaned = _clean_webvtt_text(text)
            if cleaned:
                return cleaned

    for key in ("transcriptText",):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = _clean_webvtt_text(value)
            if cleaned:
                return cleaned

    return ""


def _extract_transcript_from_secondary_item(item: dict) -> str:
    transcription_link = item.get("transcriptionLink")
    if isinstance(transcription_link, str) and transcription_link.startswith(("http://", "https://")):
        try:
            response = requests.get(transcription_link, timeout=120)
            if response.ok and response.text.strip():
                cleaned = _clean_webvtt_text(response.text)
                if cleaned:
                    return cleaned
        except requests.RequestException:
            pass

    candidate_keys = (
        "transcript",
        "transcription",
        "transcriptText",
        "subtitleText",
        "captionsText",
        "subtitles",
    )
    for key in candidate_keys:
        value = item.get(key)
        texts = _extract_text_values(value)
        if texts:
            cleaned = _clean_webvtt_text("\n".join(texts))
            if cleaned:
                return cleaned

    return ""


def _extract_media_url_from_secondary_item(item: dict) -> str:
    try:
        return _pick_media_url_from_apify_item(item)
    except Exception:
        pass

    video_meta = item.get("videoMeta")
    if isinstance(video_meta, dict):
        for key in ("downloadAddr", "playAddr", "playUrl"):
            value = video_meta.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value

    media_urls = item.get("mediaUrls")
    if isinstance(media_urls, list):
        for value in media_urls:
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value

    raise RuntimeError("Apify TikTok response did not contain a downloadable media URL.")


def _download_tiktok_media_with_apify_item(item: dict) -> tuple[str, str]:
    media_url = _extract_media_url_from_secondary_item(item)
    return _download_media_from_url(media_url, prefix="playbook_media_tiktok_apify_")


def _build_primary_tiktok_input(tiktok_url: str) -> dict:
    input_payload: dict = {"videos": [_normalize_tiktok_url(tiktok_url)]}

    extra_input_raw = os.getenv("APIFY_TIKTOK_EXTRA_INPUT_JSON", "").strip()
    if extra_input_raw:
        try:
            extra_input = json.loads(extra_input_raw)
        except ValueError as exc:
            raise RuntimeError("APIFY_TIKTOK_EXTRA_INPUT_JSON must be valid JSON.") from exc
        if not isinstance(extra_input, dict):
            raise RuntimeError("APIFY_TIKTOK_EXTRA_INPUT_JSON must be a JSON object.")
        input_payload.update(extra_input)

    return input_payload


def _build_secondary_tiktok_input(tiktok_url: str) -> dict:
    input_payload: dict = {
        "postURLs": [_normalize_tiktok_url(tiktok_url)],
        "resultsPerPage": 1,
        "downloadSubtitlesOptions": os.getenv(
            "APIFY_TIKTOK_FALLBACK_DOWNLOAD_SUBTITLES_OPTIONS",
            "DOWNLOAD_AND_TRANSCRIBE_VIDEOS_WITHOUT_SUBTITLES",
        ).strip()
        or "DOWNLOAD_AND_TRANSCRIBE_VIDEOS_WITHOUT_SUBTITLES",
    }

    extra_input_raw = os.getenv("APIFY_TIKTOK_FALLBACK_EXTRA_INPUT_JSON", "").strip()
    if extra_input_raw:
        try:
            extra_input = json.loads(extra_input_raw)
        except ValueError as exc:
            raise RuntimeError("APIFY_TIKTOK_FALLBACK_EXTRA_INPUT_JSON must be valid JSON.") from exc
        if not isinstance(extra_input, dict):
            raise RuntimeError("APIFY_TIKTOK_FALLBACK_EXTRA_INPUT_JSON must be a JSON object.")
        input_payload.update(extra_input)

    return input_payload


def extract_tiktok_transcript(tiktok_url: str) -> str:
    provider = os.getenv("TIKTOK_EXTRACTOR_PROVIDER", "apify").strip().lower()
    use_groq_whisper = _env_truthy(os.getenv("USE_GROQ_WHISPER", "true"))

    transcript_only_values = [
        os.getenv("TIKTOK_TRANSCRIPT_ONLY"),
        os.getenv("APIFY_TIKTOK_TRANSCRIPT_ONLY"),
    ]
    if any(_env_falsey(value) for value in transcript_only_values):
        transcript_only_mode = False
    elif any(_env_truthy(value) for value in transcript_only_values):
        transcript_only_mode = True
    else:
        transcript_only_mode = False

    if provider != "apify":
        raise RuntimeError(f"Unsupported TIKTOK_EXTRACTOR_PROVIDER='{provider}'. Use 'apify'.")

    primary_actor_id = os.getenv(
        "APIFY_TIKTOK_TRANSCRIPT_ACTOR_ID",
        "scrape-creators~best-tiktok-transcripts-scraper",
    ).strip()
    secondary_actor_id = os.getenv(
        "APIFY_TIKTOK_FALLBACK_ACTOR_ID",
        "clockworks~tiktok-scraper",
    ).strip()
    timeout_seconds = int(os.getenv("APIFY_TIKTOK_TIMEOUT_SECONDS", "300"))

    primary_items = _run_apify_actor(primary_actor_id, _build_primary_tiktok_input(tiktok_url), timeout_seconds)
    primary_item = primary_items[0]
    primary_transcript = _extract_transcript_from_primary_item(primary_item)
    if primary_transcript:
        return primary_transcript

    secondary_items = _run_apify_actor(secondary_actor_id, _build_secondary_tiktok_input(tiktok_url), timeout_seconds)
    secondary_item = secondary_items[0]
    secondary_transcript = _extract_transcript_from_secondary_item(secondary_item)
    if secondary_transcript:
        return secondary_transcript

    if transcript_only_mode:
        raise RuntimeError(
            "TikTok transcript actor returned no transcript text and the secondary Apify fallback did not produce "
            "usable transcript data. Disable TIKTOK_TRANSCRIPT_ONLY to allow media transcription fallback."
        )

    if use_groq_whisper and not os.getenv("GROQ_API_KEY", "").strip():
        raise RuntimeError(
            "TikTok Apify transcript extraction returned no usable transcript text, and media fallback is configured "
            "to use Groq Whisper but GROQ_API_KEY is missing."
        )

    media_path = ""
    temp_dir = ""
    try:
        media_path, temp_dir = _download_tiktok_media_with_apify_item(secondary_item)
        if use_groq_whisper:
            return _transcribe_with_groq(media_path)
        return _transcribe_with_local_whisper(media_path)
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
