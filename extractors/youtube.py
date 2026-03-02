from urllib.parse import parse_qs, urlparse

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
