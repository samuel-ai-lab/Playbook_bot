from .instagram import extract_instagram_transcript, extract_media_transcript
from .youtube import extract_youtube_transcript, parse_youtube_id


def extract_transcript(url: str) -> tuple[str, dict]:
    lowered = (url or "").lower()

    if "youtube.com" in lowered or "youtu.be" in lowered:
        video_id = parse_youtube_id(url)
        try:
            transcript = extract_youtube_transcript(video_id)
            return transcript, {"source": "youtube", "video_id": video_id}
        except Exception as exc:
            # Some videos have no caption tracks; fall back to audio transcription.
            transcript = extract_media_transcript(url)
            return transcript, {"source": "youtube_audio_fallback", "video_id": video_id, "reason": str(exc)}

    if "instagram.com" in lowered:
        transcript = extract_instagram_transcript(url)
        return transcript, {"source": "instagram"}

    raise ValueError(f"Unsupported URL: {url}")
