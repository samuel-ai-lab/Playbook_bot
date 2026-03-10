import json
import os
import random
import re
import time
from typing import Any

import requests

LONGFORM_SCHEMA = """
{
  "title": string,
  "introduction": string,
  "sections": [
    {
      "heading": string,
      "content": string,
      "notes_pack": string[]
    }
  ],
  "conclusion": string,
  "implementation_checklist": string[],
  "tags": string[]
}
""".strip()

SYSTEM_PROMPT = f"""
Situation:
You are an expert content strategist and educational writer specializing in transforming conversational podcast content
into polished, actionable written resources.

Task:
Transform the transcript into a clean, structured playbook formatted as an insight digest / educational blog post.

Objective:
Create a condensed 4-5 page longform teaching resource in an operator/playbook style that is personal, friendly,
and publication-ready.

Output constraints:
- Return ONLY a JSON object with this exact schema:
{LONGFORM_SCHEMA}
- Remove podcast artifacts (speaker labels, timestamps, fillers, back-and-forth chatter).
- Preserve key ideas, frameworks, examples, and storytelling.
- Use descriptive section headings and smooth transitions.
- End every major section with a short notes-pack bullet list in `notes_pack`.
- Keep text practical, actionable, and structured for immediate execution.
- Do not include markdown fences or extra keys.
""".strip()

CHUNK_SYSTEM_PROMPT = f"""
You are transforming one chunk of a larger transcript into longform educational playbook material.
Return ONLY a JSON object with this exact schema:
{LONGFORM_SCHEMA}

Chunk rules:
- Focus only on this chunk's ideas while writing in polished written form.
- Preserve frameworks/examples from the chunk.
- Use 2-4 section objects when possible.
- Include notes-pack bullets in each section.
- Do not include markdown fences or extra keys.
""".strip()

MERGE_SYSTEM_PROMPT = f"""
You are combining multiple chunk-level longform drafts into one coherent final playbook.
Return ONLY a JSON object with this exact schema:
{LONGFORM_SCHEMA}

Merge rules:
- Produce a single coherent narrative that reads like it was written as an article.
- Remove repetition and merge overlapping ideas.
- Preserve all high-value frameworks and examples.
- Keep transitions natural across sections.
- Keep notes-pack bullets concise and actionable.
- Do not include markdown fences or extra keys.
""".strip()

LONG_VIDEO_MERGE_SYSTEM_PROMPT = f"""
You are combining chunk drafts from a long transcript (over 1 hour) into one final longform playbook.
Return ONLY a JSON object with this exact schema:
{LONGFORM_SCHEMA}

Long-video merge rules:
- Preserve context coverage from beginning, middle, and end.
- Prioritize durable themes while retaining critical examples.
- Keep narrative flow strong and section sequencing intentional.
- Ensure each section ends with a practical notes-pack bullet list.
- Do not include markdown fences or extra keys.
""".strip()


class _PayloadTooLargeError(RuntimeError):
    pass


_LAST_LLM_CALL_TS = 0.0


def _ensure_string(value: Any, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) else fallback


def _ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _ensure_sections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    sections: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        heading = _ensure_string(item.get("heading"))
        content = _ensure_string(item.get("content"))
        notes_pack = _ensure_string_list(item.get("notes_pack"))

        if not heading and content:
            heading = "Key Idea"
        if not heading and not content and not notes_pack:
            continue

        sections.append(
            {
                "heading": heading[:180] if heading else "Key Idea",
                "content": content,
                "notes_pack": notes_pack,
            }
        )

    return sections


def _normalize_payload(payload: dict) -> dict:
    normalized = {
        "title": _ensure_string(payload.get("title"), "Playbook"),
        "introduction": _ensure_string(payload.get("introduction")),
        "sections": _ensure_sections(payload.get("sections")),
        "conclusion": _ensure_string(payload.get("conclusion")),
        "implementation_checklist": _ensure_string_list(payload.get("implementation_checklist")),
        "tags": _ensure_string_list(payload.get("tags")),
    }

    if not normalized["introduction"]:
        normalized["introduction"] = (
            "This playbook distills the transcript into a structured guide you can apply immediately."
        )

    if not normalized["sections"]:
        normalized["sections"] = [
            {
                "heading": "Core Insights",
                "content": _ensure_string(payload.get("summary")) or _ensure_string(payload.get("introduction")),
                "notes_pack": _ensure_string_list(payload.get("action_steps"))[:6],
            }
        ]

    if not normalized["conclusion"]:
        normalized["conclusion"] = "Use this playbook as an operating reference and implement one section at a time."

    if not normalized["implementation_checklist"]:
        normalized["implementation_checklist"] = [
            "Choose one section from this playbook to apply this week.",
            "Turn the notes-pack bullets into scheduled tasks.",
            "Review outcomes and refine your operating playbook."
        ]

    normalized["sections"] = normalized["sections"][:16]
    for section in normalized["sections"]:
        section["content"] = _ensure_string(section.get("content"))[:12000]
        section["notes_pack"] = _ensure_string_list(section.get("notes_pack"))[:10]

    normalized["implementation_checklist"] = normalized["implementation_checklist"][:20]
    normalized["tags"] = normalized["tags"][:20]

    return normalized


def _split_text(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    hard_min = int(chunk_chars * 0.6)
    while start < len(cleaned):
        end = min(start + chunk_chars, len(cleaned))
        if end < len(cleaned):
            split_at = cleaned.rfind("\n", start + hard_min, end)
            if split_at > start:
                end = split_at
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end, start + 1)
        if len(chunks) >= max_chunks and start < len(cleaned):
            chunks[-1] = f"{chunks[-1]}\n{cleaned[start:]}".strip()
            break
    return chunks


def _chat_json(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_seconds: int,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    max_retries = int(os.getenv("BRAIN_LLM_MAX_RETRIES", "6"))
    retry_base_sec = float(os.getenv("BRAIN_LLM_RETRY_BASE_SEC", "1.5"))
    max_backoff_sec = float(os.getenv("BRAIN_LLM_MAX_BACKOFF_SEC", "45"))
    min_interval_sec = float(os.getenv("BRAIN_LLM_MIN_INTERVAL_SEC", "0.35"))

    response = None
    for attempt in range(max_retries + 1):
        global _LAST_LLM_CALL_TS
        elapsed = time.time() - _LAST_LLM_CALL_TS
        if min_interval_sec > 0 and elapsed < min_interval_sec:
            time.sleep(min_interval_sec - elapsed)

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
            sleep_for = min(max_backoff_sec, retry_base_sec * (2**attempt)) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            continue
        finally:
            _LAST_LLM_CALL_TS = time.time()

        if response.status_code == 413:
            raise _PayloadTooLargeError("LLM request payload too large.")

        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= max_retries:
                response.raise_for_status()

            retry_after_hdr = response.headers.get("Retry-After", "").strip()
            retry_after_sec = 0.0
            if retry_after_hdr:
                try:
                    retry_after_sec = float(retry_after_hdr)
                except ValueError:
                    retry_after_sec = 0.0

            backoff_sec = min(max_backoff_sec, retry_base_sec * (2**attempt))
            sleep_for = max(retry_after_sec, backoff_sec) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            continue

        response.raise_for_status()
        break

    if response is None:
        raise RuntimeError("LLM request failed before receiving a response.")

    body = response.json()

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"LLM returned empty content: {body}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return valid JSON: {content[:500]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object, got: {type(parsed)}")
    return parsed


def _compact_analysis(item: dict) -> dict:
    sections = _ensure_sections(item.get("sections"))[:8]
    compact_sections: list[dict[str, Any]] = []
    for section in sections:
        compact_sections.append(
            {
                "heading": _ensure_string(section.get("heading"))[:140],
                "content": _ensure_string(section.get("content"))[:700],
                "notes_pack": [_ensure_string(note)[:180] for note in _ensure_string_list(section.get("notes_pack"))[:6]],
            }
        )

    return {
        "title": _ensure_string(item.get("title"), "")[:140],
        "introduction": _ensure_string(item.get("introduction"), "")[:700],
        "sections": compact_sections,
        "conclusion": _ensure_string(item.get("conclusion"), "")[:600],
        "implementation_checklist": [
            _ensure_string(step)[:180] for step in _ensure_string_list(item.get("implementation_checklist"))[:8]
        ],
        "tags": [_ensure_string(tag)[:40] for tag in _ensure_string_list(item.get("tags"))[:10]],
    }


def _estimate_duration_seconds_from_transcript(transcript_text: str) -> int:
    words = re.findall(r"\b\w+\b", transcript_text)
    # Rough speech density estimate for long-form business content.
    words_per_minute = 150
    return int((len(words) / max(1, words_per_minute)) * 60)


def _merge_batch(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    source_url: str,
    analyses_batch: list[dict],
    merge_system_prompt: str,
) -> dict:
    merge_prompt = (
        f"Source URL: {source_url}\n\n"
        "Combine these chunk analyses into one final playbook JSON:\n"
        f"{json.dumps(analyses_batch, ensure_ascii=True)}\n\n"
        "Produce the final merged JSON now."
    )
    merged = _chat_json(
        endpoint=endpoint,
        headers=headers,
        model=model,
        messages=[
            {"role": "system", "content": merge_system_prompt},
            {"role": "user", "content": merge_prompt},
        ],
        temperature=0.15,
        timeout_seconds=240,
    )
    return _normalize_payload(merged)


def _merge_analyses_hierarchical(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    source_url: str,
    analyses: list[dict],
    initial_batch_size: int,
    merge_system_prompt: str,
) -> dict:
    if not analyses:
        raise RuntimeError("No chunk analyses to merge.")

    current = analyses
    batch_size = max(2, initial_batch_size)

    while len(current) > 1:
        next_round: list[dict] = []
        restart_level = False

        for i in range(0, len(current), batch_size):
            batch = current[i : i + batch_size]
            if len(batch) == 1:
                next_round.append(batch[0])
                continue
            try:
                next_round.append(
                    _merge_batch(endpoint, headers, model, source_url, batch, merge_system_prompt=merge_system_prompt)
                )
            except _PayloadTooLargeError:
                if batch_size > 2:
                    batch_size = max(2, batch_size // 2)
                    restart_level = True
                    break
                compact = [_compact_analysis(item) for item in batch]
                next_round.append(
                    _merge_batch(
                        endpoint,
                        headers,
                        model,
                        source_url,
                        compact,
                        merge_system_prompt=merge_system_prompt,
                    )
                )

        if restart_level:
            continue
        current = next_round

    return _normalize_payload(current[0])


def _split_chunk_for_retry(text: str) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return [cleaned]

    mid = len(cleaned) // 2
    left_break = cleaned.rfind("\n", int(len(cleaned) * 0.3), mid)
    if left_break == -1:
        right_break = cleaned.find("\n", mid, int(len(cleaned) * 0.7))
        split_at = right_break if right_break != -1 else mid
    else:
        split_at = left_break

    first = cleaned[:split_at].strip()
    second = cleaned[split_at:].strip()
    if not first or not second:
        split_at = mid
        first = cleaned[:split_at].strip()
        second = cleaned[split_at:].strip()

    return [part for part in [first, second] if part]


def _analyze_chunk_resilient(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    source_url: str,
    chunk: str,
    chunk_label: str,
    min_chunk_chars: int,
) -> list[dict]:
    user_prompt = (
        f"Source URL: {source_url}\n"
        f"Chunk: {chunk_label}\n\n"
        f"Transcript chunk:\n{chunk}\n\n"
        "Produce the JSON now."
    )

    try:
        parsed = _chat_json(
            endpoint=endpoint,
            headers=headers,
            model=model,
            messages=[
                {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout_seconds=180,
        )
        return [_normalize_payload(parsed)]
    except _PayloadTooLargeError:
        if len(chunk) <= min_chunk_chars:
            raise RuntimeError(
                "Chunk request exceeded payload size at minimum split. "
                "Reduce BRAIN_MIN_CHUNK_CHARS (e.g. 4000) or shorten transcript input."
            )

        sub_chunks = _split_chunk_for_retry(chunk)
        if len(sub_chunks) < 2:
            raise RuntimeError("Chunk request exceeded payload size and could not split further.")

        analyses: list[dict] = []
        for index, sub_chunk in enumerate(sub_chunks, start=1):
            analyses.extend(
                _analyze_chunk_resilient(
                    endpoint=endpoint,
                    headers=headers,
                    model=model,
                    source_url=source_url,
                    chunk=sub_chunk,
                    chunk_label=f"{chunk_label}.{index}",
                    min_chunk_chars=min_chunk_chars,
                )
            )
        return analyses


def _single_pass_playbook(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    transcript_text: str,
    source_url: str,
) -> dict:
    user_prompt = (
        f"Source URL: {source_url}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Produce the JSON now."
    )
    parsed = _chat_json(
        endpoint=endpoint,
        headers=headers,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        timeout_seconds=240,
    )
    return _normalize_payload(parsed)


def _chunked_playbook(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    transcript_text: str,
    source_url: str,
    chunk_chars: int,
    max_chunks: int,
    merge_batch_size: int,
    min_chunk_chars: int,
    long_video_mode: bool,
) -> dict:
    chunks = _split_text(transcript_text, chunk_chars=chunk_chars, max_chunks=max_chunks)
    if not chunks:
        raise RuntimeError("Transcript is empty")

    analyses: list[dict] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        analyses.extend(
            _analyze_chunk_resilient(
                endpoint=endpoint,
                headers=headers,
                model=model,
                source_url=source_url,
                chunk=chunk,
                chunk_label=f"{index}/{total}",
                min_chunk_chars=min_chunk_chars,
            )
        )

    return _merge_analyses_hierarchical(
        endpoint=endpoint,
        headers=headers,
        model=model,
        source_url=source_url,
        analyses=analyses,
        initial_batch_size=merge_batch_size,
        merge_system_prompt=LONG_VIDEO_MERGE_SYSTEM_PROMPT if long_video_mode else MERGE_SYSTEM_PROMPT,
    )


def generate_playbook(transcript_text: str, source_url: str = "", duration_seconds: int | None = None) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    chunk_chars = int(os.getenv("BRAIN_TRANSCRIPT_CHUNK_CHARS", "18000"))
    max_chunks = int(os.getenv("BRAIN_MAX_CHUNKS", "200"))
    merge_batch_size = int(os.getenv("BRAIN_MERGE_BATCH_SIZE", "4"))
    min_chunk_chars = int(os.getenv("BRAIN_MIN_CHUNK_CHARS", "4000"))
    long_video_seconds = int(os.getenv("BRAIN_LONG_VIDEO_SECONDS", "3600"))

    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    if not transcript_text.strip():
        raise RuntimeError("Transcript is empty")

    endpoint = f"{groq_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    effective_duration = duration_seconds if isinstance(duration_seconds, int) and duration_seconds > 0 else None
    if effective_duration is None:
        effective_duration = _estimate_duration_seconds_from_transcript(transcript_text)
    long_video_mode = effective_duration > long_video_seconds

    if len(transcript_text) <= chunk_chars:
        try:
            return _single_pass_playbook(endpoint, headers, groq_llm_model, transcript_text, source_url)
        except _PayloadTooLargeError:
            pass
    return _chunked_playbook(
        endpoint=endpoint,
        headers=headers,
        model=groq_llm_model,
        transcript_text=transcript_text,
        source_url=source_url,
        chunk_chars=chunk_chars,
        max_chunks=max_chunks,
        merge_batch_size=merge_batch_size,
        min_chunk_chars=min_chunk_chars,
        long_video_mode=long_video_mode,
    )
