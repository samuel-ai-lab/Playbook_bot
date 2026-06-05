import os

from .instagram import extract_instagram_transcript, extract_media_transcript
from .tiktok import extract_tiktok_transcript
from .youtube import (
    extract_youtube_duration_seconds,
    extract_youtube_transcript,
    extract_youtube_transcript_with_transcriptapi,
    extract_youtube_transcript_with_ytdlp,
    parse_youtube_id,
)


def extract_transcript(url: str) -> tuple[str, dict]:
    lowered = (url or "").lower()
    youtube_audio_fallback_enabled = os.getenv("YOUTUBE_AUDIO_FALLBACK_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    youtube_caption_fallback_enabled = os.getenv("YOUTUBE_CAPTION_FALLBACK_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    youtube_fetch_duration = os.getenv("YOUTUBE_FETCH_DURATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if "youtube.com" in lowered or "youtu.be" in lowered:
        video_id = parse_youtube_id(url)
        duration_seconds = extract_youtube_duration_seconds(url) if youtube_fetch_duration else None
        extraction_errors: list[str] = []

        if os.getenv("TRANSCRIPT_API_KEY", "").strip():
            try:
                transcript = extract_youtube_transcript_with_transcriptapi(url)
                return transcript, {
                    "source": "youtube_transcriptapi",
                    "video_id": video_id,
                    "duration_seconds": duration_seconds,
                }
            except Exception as transcript_api_exc:
                extraction_errors.append(f"transcriptapi: {transcript_api_exc}")

        try:
            transcript = extract_youtube_transcript(video_id)
            return transcript, {"source": "youtube", "video_id": video_id, "duration_seconds": duration_seconds}
        except Exception as primary_exc:
            extraction_errors.append(f"youtube_transcript_api: {primary_exc}")
            if youtube_caption_fallback_enabled:
                try:
                    transcript = extract_youtube_transcript_with_ytdlp(url)
                    return transcript, {
                        "source": "youtube_caption_fallback",
                        "video_id": video_id,
                        "duration_seconds": duration_seconds,
                        "reason": "; ".join(extraction_errors),
                    }
                except Exception as secondary_exc:
                    extraction_errors.append(f"yt_dlp_captions: {secondary_exc}")

            if youtube_audio_fallback_enabled:
                # Audio fallback can hit anti-bot checks on CI runners.
                transcript = extract_media_transcript(url)
                return transcript, {
                    "source": "youtube_audio_fallback",
                    "video_id": video_id,
                    "duration_seconds": duration_seconds,
                    "reason": "; ".join(extraction_errors),
                }

            raise RuntimeError(
                "YouTube transcript extraction failed without yt-dlp fallbacks. "
                f"Details: {'; '.join(extraction_errors)}"
            ) from primary_exc

    if "instagram.com" in lowered:
        transcript = extract_instagram_transcript(url)
        return transcript, {"source": "instagram"}

    if "tiktok.com" in lowered:
        transcript = extract_tiktok_transcript(url)
        return transcript, {"source": "tiktok"}

    raise ValueError(f"Unsupported URL: {url}")
