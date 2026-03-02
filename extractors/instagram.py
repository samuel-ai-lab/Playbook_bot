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


class _GroqUploadTooLarge(RuntimeError):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[:500]
        raise RuntimeError(f"ffmpeg failed: {stderr}")


def _compress_audio_for_groq(input_path: str) -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for large media fallback. Install ffmpeg or use local Whisper.")

    output_path = str(Path(input_path).with_suffix(".groq.mp3"))
    _run_ffmpeg(
        [
            ffmpeg_bin,
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


def _split_audio_for_groq(input_path: str, chunk_seconds: int = 900) -> list[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for chunked transcription fallback. Install ffmpeg.")

    chunk_dir = Path(input_path).parent / "groq_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%03d.mp3")

    _run_ffmpeg(
        [
            ffmpeg_bin,
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

    chunks = sorted(str(path) for path in chunk_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("Failed to create audio chunks for large-file transcription.")
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
        media_path, temp_dir = _download_media_with_ytdlp(source_url)

        if use_groq_whisper:
            return _transcribe_with_groq(media_path)
        return _transcribe_with_local_whisper(media_path)
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_instagram_transcript(instagram_url: str) -> str:
    return extract_media_transcript(instagram_url)
