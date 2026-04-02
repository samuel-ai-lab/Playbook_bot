import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests


def _normalize_source_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    if "instagram.com" in parsed.netloc:
        # Remove tracking parameters for cleaner extraction.
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        # Drop tracking parameters on YouTube URLs.
        if "youtube.com" in parsed.netloc and parsed.path == "/watch":
            query = parse_qs(parsed.query)
            clean_query = urlencode({"v": query.get("v", [""])[0]}) if query.get("v") else ""
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, ""))
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return source_url


def _download_media_with_ytdlp(source_url: str) -> tuple[str, str]:
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for media URL extraction. Install dependencies and retry.") from exc

    normalized_url = _normalize_source_url(source_url)
    temp_dir = tempfile.mkdtemp(prefix="playbook_media_")
    outtmpl = str(Path(temp_dir) / "media.%(ext)s")

    ydl_base_opts: dict = {
        "noplaylist": True,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }
    cookies_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    if cookies_file:
        ydl_base_opts["cookiefile"] = cookies_file
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_from_browser:
        ydl_base_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    info = None
    downloaded_path = ""
    last_exc: Exception | None = None
    is_youtube = "youtube.com" in normalized_url or "youtu.be" in normalized_url

    format_candidates = [
        "worstaudio[abr<=64]/worstaudio/bestaudio[abr<=64]/bestaudio/best",
        "bestaudio/best",
        "worstaudio/worst",
        "best",
    ]
    option_sets: list[dict] = [dict(ydl_base_opts)]
    if is_youtube:
        retry_opts = dict(ydl_base_opts)
        retry_opts["extractor_args"] = {"youtube": {"player_client": ["android", "web"]}}
        option_sets.append(retry_opts)

        pot_opts = dict(ydl_base_opts)
        # See yt-dlp issue 13058: some videos expose only formats flagged as missing PO Token.
        pot_opts["extractor_args"] = {
            "youtube": {
                "player_client": ["android", "ios", "tv", "web"],
                "formats": ["missing_pot"],
            }
        }
        option_sets.append(pot_opts)

    for base_opts in option_sets:
        for fmt in format_candidates:
            opts = dict(base_opts)
            opts["format"] = fmt
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(normalized_url, download=True)
                    downloaded_path = ydl.prepare_filename(info)
                last_exc = None
                break
            except DownloadError as exc:
                last_exc = exc
                continue
        if info is not None:
            break

    # Final fallback: let yt-dlp choose default format without constraints.
    if info is None:
        for base_opts in option_sets:
            opts = dict(base_opts)
            opts.pop("format", None)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(normalized_url, download=True)
                    downloaded_path = ydl.prepare_filename(info)
                last_exc = None
                break
            except DownloadError as exc:
                last_exc = exc
                continue

    if info is None:
        # Fallback: pick concrete format IDs from metadata instead of relying on selectors.
        if is_youtube:
            try:
                meta_opts = dict(ydl_base_opts)
                meta_opts["skip_download"] = True
                meta_opts["ignore_no_formats_error"] = True
                meta_opts["extractor_args"] = {
                    "youtube": {
                        "player_client": ["android", "ios", "tv", "web"],
                        "formats": ["missing_pot"],
                    }
                }
                with yt_dlp.YoutubeDL(meta_opts) as ydl:
                    metadata = ydl.extract_info(normalized_url, download=False)

                formats = metadata.get("formats", []) if isinstance(metadata, dict) else []
                audio_only: list[tuple[str, float]] = []
                audio_any: list[tuple[str, float]] = []
                for fmt in formats:
                    if not isinstance(fmt, dict):
                        continue
                    format_id = fmt.get("format_id")
                    if not isinstance(format_id, str) or not format_id:
                        continue
                    acodec = str(fmt.get("acodec", "none"))
                    if acodec == "none":
                        continue
                    quality = float(fmt.get("abr") or fmt.get("tbr") or 999999)
                    if str(fmt.get("vcodec", "none")) == "none":
                        audio_only.append((format_id, quality))
                    else:
                        audio_any.append((format_id, quality))

                candidates = sorted(audio_only or audio_any, key=lambda item: item[1])
                tried: set[str] = set()
                for format_id, _ in candidates:
                    if format_id in tried:
                        continue
                    tried.add(format_id)
                    opts = dict(ydl_base_opts)
                    opts["extractor_args"] = {
                        "youtube": {
                            "player_client": ["android", "ios", "tv", "web"],
                            "formats": ["missing_pot"],
                        }
                    }
                    opts["format"] = format_id
                    try:
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            info = ydl.extract_info(normalized_url, download=True)
                            downloaded_path = ydl.prepare_filename(info)
                        last_exc = None
                        break
                    except DownloadError as exc:
                        last_exc = exc
                        continue
            except Exception as exc:
                last_exc = exc

        if last_exc:
            raise last_exc
        raise RuntimeError("yt-dlp could not download media from the provided URL.")

    candidate_paths = []
    if downloaded_path and os.path.exists(downloaded_path):
        candidate_paths.append(downloaded_path)

    requested = info.get("requested_downloads") if isinstance(info, dict) else None
    if isinstance(requested, list):
        for item in requested:
            if isinstance(item, dict):
                filepath = item.get("filepath")
                if isinstance(filepath, str) and os.path.exists(filepath):
                    candidate_paths.append(filepath)

    if not candidate_paths:
        for child in Path(temp_dir).iterdir():
            if child.is_file() and not child.name.endswith((".part", ".ytdl")):
                candidate_paths.append(str(child))

    if not candidate_paths:
        raise RuntimeError("yt-dlp could not download media from the provided URL.")

    # Prefer the largest file when multiple artifacts exist.
    candidate_paths.sort(key=lambda path: os.path.getsize(path), reverse=True)
    return candidate_paths[0], temp_dir


def _download_youtube_audio_with_pytubefix(source_url: str) -> tuple[str, str]:
    try:
        from pytubefix import YouTube
    except ImportError as exc:
        raise RuntimeError("pytubefix is required for YouTube downloader strategy 'pytubefix'.") from exc

    normalized_url = _normalize_source_url(source_url)
    temp_dir = tempfile.mkdtemp(prefix="playbook_media_yt_")

    yt = YouTube(normalized_url, use_oauth=False, allow_oauth_cache=False)
    stream = (
        yt.streams.filter(only_audio=True, file_extension="mp4").order_by("abr").desc().first()
        or yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        or yt.streams.filter(only_audio=True).first()
    )
    if not stream:
        raise RuntimeError("pytubefix could not find an audio stream for this YouTube video.")

    downloaded_path = stream.download(output_path=temp_dir, filename="media")
    if not downloaded_path or not os.path.exists(downloaded_path):
        raise RuntimeError("pytubefix did not produce a downloadable media file.")

    return downloaded_path, temp_dir


def _download_media(source_url: str) -> tuple[str, str]:
    normalized_url = _normalize_source_url(source_url)
    is_youtube = "youtube.com" in normalized_url or "youtu.be" in normalized_url

    if not is_youtube:
        return _download_media_with_ytdlp(source_url)

    youtube_downloader = os.getenv("YOUTUBE_DOWNLOADER", "pytubefix").strip().lower()
    last_exc: Exception | None = None

    if youtube_downloader in {"pytubefix", "auto"}:
        try:
            return _download_youtube_audio_with_pytubefix(source_url)
        except Exception as exc:
            last_exc = exc
            if youtube_downloader == "pytubefix":
                raise RuntimeError(f"YouTube download via pytubefix failed: {exc}") from exc

    if youtube_downloader in {"yt-dlp", "auto"}:
        try:
            return _download_media_with_ytdlp(source_url)
        except Exception as exc:
            last_exc = exc

    if last_exc:
        raise RuntimeError(f"YouTube media download failed: {last_exc}") from last_exc
    raise RuntimeError(f"Unsupported YOUTUBE_DOWNLOADER value: {youtube_downloader}")


def _looks_like_media_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    media_exts = (".mp4", ".m4a", ".mp3", ".webm", ".mkv", ".aac", ".wav", ".ogg", ".mov")
    return path.endswith(media_exts) or any(ext in path for ext in media_exts)


def _extract_url_candidates_from_payload(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            urls.append(value)
        return urls

    if isinstance(value, list):
        for item in value:
            urls.extend(_extract_url_candidates_from_payload(item))
        return urls

    if isinstance(value, dict):
        for item in value.values():
            urls.extend(_extract_url_candidates_from_payload(item))
        return urls

    return urls


def _pick_media_url_from_apify_item(item: dict) -> str:
    priority_keys = [
        "videoUrl",
        "video_url",
        "videoHdUrl",
        "video_hd_url",
        "videoPlayUrl",
        "video_play_url",
        "downloadUrl",
        "download_url",
        "downloadedVideo",
        "downloaded_video",
        "audioUrl",
        "audio_url",
        "fileUrl",
        "file_url",
    ]
    for key in priority_keys:
        candidate = item.get(key)
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate

    all_urls = _extract_url_candidates_from_payload(item)
    media_like = [url for url in all_urls if _looks_like_media_url(url)]
    if media_like:
        return media_like[0]
    if all_urls:
        return all_urls[0]

    raise RuntimeError("Apify response did not contain a downloadable media URL.")


def _build_apify_instagram_input(instagram_url: str, actor_id: str) -> dict:
    mode = os.getenv("APIFY_INSTAGRAM_INPUT_MODE", "auto").strip().lower()
    normalized_actor = actor_id.replace("/", "~").strip().lower()

    if mode == "auto":
        # apify/instagram-reel-scraper requires a "username" field and accepts reel URLs in it.
        if normalized_actor == "apify~instagram-reel-scraper":
            mode = "username"
        else:
            mode = "starturls"

    input_payload: dict
    if mode == "urls":
        input_payload = {"urls": [{"url": instagram_url}]}
    elif mode == "starturls":
        input_payload = {"startUrls": [{"url": instagram_url}]}
    elif mode == "directurls":
        input_payload = {"directUrls": [instagram_url]}
    elif mode == "username":
        input_payload = {"username": [instagram_url]}
    else:
        raise RuntimeError(
            "Unsupported APIFY_INSTAGRAM_INPUT_MODE="
            f"'{mode}'. Use one of: auto, username, urls, startUrls, directUrls."
        )

    # Official apify/instagram-reel-scraper input schema has includeTranscript=false by default.
    # Request transcript explicitly so pipeline does not unexpectedly fall back to media transcription.
    include_transcript_raw = os.getenv("APIFY_INSTAGRAM_INCLUDE_TRANSCRIPT", "true")
    input_payload.setdefault(
        "includeTranscript",
        include_transcript_raw.strip().lower() in {"1", "true", "yes"},
    )

    extra_input_raw = os.getenv("APIFY_INSTAGRAM_EXTRA_INPUT_JSON", "").strip()
    if extra_input_raw:
        try:
            extra_input = json.loads(extra_input_raw)
        except ValueError as exc:
            raise RuntimeError("APIFY_INSTAGRAM_EXTRA_INPUT_JSON must be valid JSON.") from exc
        if not isinstance(extra_input, dict):
            raise RuntimeError("APIFY_INSTAGRAM_EXTRA_INPUT_JSON must be a JSON object.")
        input_payload.update(extra_input)

    return input_payload


def _run_instagram_apify_actor(instagram_url: str) -> dict:
    token = os.getenv("APIFY_TOKEN", "").strip()
    actor_id = os.getenv("APIFY_INSTAGRAM_ACTOR_ID", "apify~instagram-reel-scraper").strip()
    timeout_seconds = int(os.getenv("APIFY_INSTAGRAM_TIMEOUT_SECONDS", "300"))

    if not token:
        raise RuntimeError("Missing APIFY_TOKEN for Instagram extractor provider 'apify'.")
    if not actor_id:
        raise RuntimeError("Missing APIFY_INSTAGRAM_ACTOR_ID for Instagram extractor provider 'apify'.")

    endpoint = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    params = {
        "format": "json",
        "clean": "1",
    }
    headers = {"Authorization": f"Bearer {token}"}
    run_input = _build_apify_instagram_input(instagram_url, actor_id=actor_id)

    response = requests.post(endpoint, params=params, headers=headers, json=run_input, timeout=timeout_seconds)
    if not response.ok:
        details = response.text[:500]
        if response.status_code == 400 and "input.username" in details:
            raise RuntimeError(
                "Selected Apify actor expects input.username. "
                "Use APIFY_INSTAGRAM_ACTOR_ID=apify~instagram-reel-scraper and "
                "APIFY_INSTAGRAM_INPUT_MODE=username (or auto)."
            )
        if response.status_code == 400 and "input.startUrls" in details:
            raise RuntimeError(
                "Selected Apify actor expects input.startUrls. "
                "Set APIFY_INSTAGRAM_INPUT_MODE=starturls."
            )
        raise RuntimeError(f"Apify Instagram extractor failed ({response.status_code}): {details}")

    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Apify returned no dataset items for URL: {instagram_url}")

    first = payload[0]
    if not isinstance(first, dict):
        raise RuntimeError("Unexpected Apify response item format.")
    return first


def _extract_transcript_from_apify_item(item: dict) -> str:
    transcript = item.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return transcript.strip()

    if isinstance(transcript, list):
        lines: list[str] = []
        for part in transcript:
            if isinstance(part, str) and part.strip():
                lines.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    lines.append(text.strip())
        merged = "\n".join(lines).strip()
        if merged:
            return merged

    return ""


def _download_media_from_url(media_url: str, prefix: str = "playbook_media_ig_") -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    parsed = urlparse(media_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    output_path = Path(temp_dir) / f"media{suffix}"

    with requests.get(media_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Downloaded media file is empty.")

    return str(output_path), temp_dir


def _download_instagram_media_with_apify(instagram_url: str, actor_item: dict | None = None) -> tuple[str, str]:
    item = actor_item if isinstance(actor_item, dict) else _run_instagram_apify_actor(_normalize_source_url(instagram_url))
    media_url = _pick_media_url_from_apify_item(item)
    return _download_media_from_url(media_url, prefix="playbook_media_ig_apify_")


class _GroqUploadTooLarge(RuntimeError):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details_source = stderr or stdout or "No ffmpeg output captured."
        lines = details_source.splitlines()
        tail = "\n".join(lines[-25:])
        raise RuntimeError(f"ffmpeg failed (exit {completed.returncode}): {tail[:2000]}")


def _compress_audio_for_groq(input_path: str) -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for large media fallback. Install ffmpeg or use local Whisper.")

    output_path = str(Path(input_path).with_suffix(".groq.mp3"))
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "24k",
            output_path,
        ]
    )
    return output_path


def _list_chunk_files(chunk_dir: Path) -> list[str]:
    return sorted(str(path) for path in chunk_dir.glob("chunk_*.mp3") if path.is_file() and path.stat().st_size > 0)


def _clear_chunk_files(chunk_dir: Path) -> None:
    for path in chunk_dir.glob("chunk_*.mp3"):
        try:
            path.unlink()
        except OSError:
            continue


def _split_audio_for_groq(input_path: str, chunk_seconds: int = 900) -> list[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for chunked transcription fallback. Install ffmpeg.")

    chunk_dir = Path(input_path).parent / "groq_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%03d.mp3")

    _clear_chunk_files(chunk_dir)
    try:
        # Fast path: split without re-encoding.
        _run_ffmpeg(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                input_path,
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-c",
                "copy",
                pattern,
            ]
        )
    except RuntimeError:
        pass

    chunks = _list_chunk_files(chunk_dir)
    if chunks:
        return chunks

    # Fallback: segment while re-encoding for sources that fail in copy mode.
    _clear_chunk_files(chunk_dir)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "24k",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            pattern,
        ]
    )

    chunks = _list_chunk_files(chunk_dir)
    if not chunks:
        raise RuntimeError("Failed to create non-empty audio chunks for large-file transcription.")
    return chunks


def _groq_transcribe_file(
    media_path: str, endpoint: str, headers: dict[str, str], model: str, timeout_seconds: int
) -> str:
    with open(media_path, "rb") as media_stream:
        files = {"file": (Path(media_path).name, media_stream, "application/octet-stream")}
        data = {"model": model, "response_format": "verbose_json"}
        response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=timeout_seconds)

    if response.status_code == 413:
        raise _GroqUploadTooLarge("Groq upload too large for single-request transcription.")
    if not response.ok:
        details = response.text[:500]
        raise RuntimeError(f"Groq transcription request failed ({response.status_code}): {details}")

    payload = response.json()
    text = payload.get("text", "").strip()
    if not text:
        raise RuntimeError(f"Groq Whisper returned empty transcript: {payload}")
    return text


def _transcribe_with_groq(media_path: str) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_whisper_model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    groq_max_upload_bytes = int(os.getenv("GROQ_MAX_UPLOAD_BYTES", "24000000"))
    groq_chunk_seconds = int(os.getenv("GROQ_AUDIO_CHUNK_SECONDS", "900"))

    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY for Groq Whisper transcription")

    endpoint = f"{groq_base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_api_key}"}
    working_path = media_path

    if os.path.getsize(working_path) > groq_max_upload_bytes:
        working_path = _compress_audio_for_groq(media_path)

    try:
        return _groq_transcribe_file(working_path, endpoint, headers, groq_whisper_model, timeout_seconds=300)
    except _GroqUploadTooLarge:
        chunk_input = working_path
        if os.path.getsize(chunk_input) > groq_max_upload_bytes:
            chunk_input = _compress_audio_for_groq(working_path)
        chunk_paths = _split_audio_for_groq(chunk_input, chunk_seconds=groq_chunk_seconds)
        transcripts: list[str] = []
        for chunk_path in chunk_paths:
            try:
                transcripts.append(
                    _groq_transcribe_file(
                        chunk_path,
                        endpoint,
                        headers,
                        groq_whisper_model,
                        timeout_seconds=180,
                    )
                )
            except _GroqUploadTooLarge as exc:
                raise RuntimeError(
                    "Chunk is still too large for Groq. Reduce GROQ_AUDIO_CHUNK_SECONDS (e.g. 600 or 300)."
                ) from exc

        merged = "\n".join(part for part in transcripts if part.strip()).strip()
        if not merged:
            raise RuntimeError("Chunked Groq transcription returned empty text.")
        return merged


def _transcribe_with_local_whisper(media_path: str) -> str:
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Local Whisper requested but package is not installed. Install openai-whisper or set USE_GROQ_WHISPER=true"
        ) from exc

    model_name = os.getenv("LOCAL_WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    result = model.transcribe(media_path)

    text = (result or {}).get("text", "").strip()
    if not text:
        raise RuntimeError("Local Whisper returned empty transcript")

    return text


def extract_media_transcript(source_url: str) -> str:
    use_groq_whisper = os.getenv("USE_GROQ_WHISPER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    media_path = ""
    temp_dir = ""
    try:
        media_path, temp_dir = _download_media(source_url)

        if use_groq_whisper:
            return _transcribe_with_groq(media_path)
        return _transcribe_with_local_whisper(media_path)
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_instagram_transcript(instagram_url: str) -> str:
    provider = os.getenv("INSTAGRAM_EXTRACTOR_PROVIDER", "apify").strip().lower()
    use_groq_whisper = os.getenv("USE_GROQ_WHISPER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    allow_ytdlp_fallback = os.getenv("INSTAGRAM_APIFY_FALLBACK_TO_YTDLP", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    transcript_only_raw = os.getenv(
        "APIFY_INSTAGRAM_TRANSCRIPT_ONLY",
        os.getenv("INSTAGRAM_TRANSCRIPT_ONLY", "false"),
    )
    transcript_only_mode = transcript_only_raw.strip().lower() in {
        "1",
        "true",
        "yes",
    }
    print(
        "Instagram transcript mode debug:",
        {
            "APIFY_INSTAGRAM_TRANSCRIPT_ONLY": os.getenv("APIFY_INSTAGRAM_TRANSCRIPT_ONLY"),
            "INSTAGRAM_TRANSCRIPT_ONLY": os.getenv("INSTAGRAM_TRANSCRIPT_ONLY"),
            "transcript_only_raw": transcript_only_raw,
            "transcript_only_mode": transcript_only_mode,
            "USE_GROQ_WHISPER": os.getenv("USE_GROQ_WHISPER"),
        },
    )

    if provider == "yt-dlp":
        return extract_media_transcript(instagram_url)

    if provider == "apify":
        media_path = ""
        temp_dir = ""
        try:
            actor_item = _run_instagram_apify_actor(instagram_url)
            actor_transcript = _extract_transcript_from_apify_item(actor_item)
            if actor_transcript:
                return actor_transcript
            if transcript_only_mode:
                raise RuntimeError(
                    "Apify actor returned no transcript text. "
                    "Set APIFY_INSTAGRAM_INCLUDE_TRANSCRIPT=true (or pass includeTranscript=true in "
                    "APIFY_INSTAGRAM_EXTRA_INPUT_JSON) "
                    "or disable INSTAGRAM_TRANSCRIPT_ONLY."
                )
            if use_groq_whisper and not os.getenv("GROQ_API_KEY", "").strip():
                raise RuntimeError(
                    "Apify actor returned no transcript text, and media fallback is configured to use Groq Whisper "
                    "but GROQ_API_KEY is missing. Set INSTAGRAM_TRANSCRIPT_ONLY=true for no-fallback mode, or set "
                    "GROQ_API_KEY to allow media transcription."
                )

            media_path, temp_dir = _download_instagram_media_with_apify(instagram_url, actor_item=actor_item)
            if use_groq_whisper:
                return _transcribe_with_groq(media_path)
            return _transcribe_with_local_whisper(media_path)
        except Exception:
            if allow_ytdlp_fallback:
                return extract_media_transcript(instagram_url)
            raise
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    raise RuntimeError(
        f"Unsupported INSTAGRAM_EXTRACTOR_PROVIDER='{provider}'. Use 'yt-dlp' or 'apify'."
    )
