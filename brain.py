import json
import os
from typing import Any

import requests

SYSTEM_PROMPT = """
You are a Senior Business Consultant.
Read the transcript and extract high-leverage tactics.
Return ONLY a JSON object with this exact schema:
{
  "title": string,
  "summary": string,
  "action_steps": string[],
  "insights": string[],
  "tags": string[]
}
Rules:
- title must be action-oriented.
- summary must be concise and practical.
- action_steps must be imperative and specific.
- insights should contain strategic observations.
- tags should be short categorical labels.
- Do not include markdown, prose, or extra keys.
""".strip()

CHUNK_SYSTEM_PROMPT = """
You are a Senior Business Consultant analyzing one chunk of a longer transcript.
Extract the highest-leverage tactics from this chunk only.
Return ONLY a JSON object with this exact schema:
{
  "title": string,
  "summary": string,
  "action_steps": string[],
  "insights": string[],
  "tags": string[]
}
Rules:
- Keep output concise and specific.
- action_steps must be imperative.
- tags must be short labels.
- Do not include markdown, prose, or extra keys.
""".strip()

MERGE_SYSTEM_PROMPT = """
You are a Senior Business Consultant combining multiple chunk analyses into one final playbook.
Return ONLY a JSON object with this exact schema:
{
  "title": string,
  "summary": string,
  "action_steps": string[],
  "insights": string[],
  "tags": string[]
}
Rules:
- Synthesize across all chunk analyses without losing important tactics.
- Remove duplicates and merge overlapping steps.
- Keep action_steps specific and executable.
- Keep tags compact and de-duplicated.
- Do not include markdown, prose, or extra keys.
""".strip()


class _PayloadTooLargeError(RuntimeError):
    pass


def _ensure_string(value: Any, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) else fallback


def _ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_payload(payload: dict) -> dict:
    normalized = {
        "title": _ensure_string(payload.get("title"), "Build an actionable playbook"),
        "summary": _ensure_string(payload.get("summary")),
        "action_steps": _ensure_string_list(payload.get("action_steps")),
        "insights": _ensure_string_list(payload.get("insights")),
        "tags": _ensure_string_list(payload.get("tags")),
    }

    if not normalized["action_steps"]:
        normalized["action_steps"] = ["Review the source and extract 3 specific actions to execute this week."]

    if not normalized["summary"]:
        normalized["summary"] = "No summary was generated from the transcript."

    normalized["action_steps"] = normalized["action_steps"][:40]
    normalized["insights"] = normalized["insights"][:40]
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
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
    if response.status_code == 413:
        raise _PayloadTooLargeError("LLM request payload too large.")
    response.raise_for_status()
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
    return {
        "title": _ensure_string(item.get("title"), "")[:140],
        "summary": _ensure_string(item.get("summary"), "")[:500],
        "action_steps": [_ensure_string(step)[:220] for step in _ensure_string_list(item.get("action_steps"))[:10]],
        "insights": [_ensure_string(step)[:220] for step in _ensure_string_list(item.get("insights"))[:10]],
        "tags": [_ensure_string(tag)[:40] for tag in _ensure_string_list(item.get("tags"))[:10]],
    }


def _merge_batch(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    source_url: str,
    analyses_batch: list[dict],
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
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
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
                next_round.append(_merge_batch(endpoint, headers, model, source_url, batch))
            except _PayloadTooLargeError:
                if batch_size > 2:
                    batch_size = max(2, batch_size // 2)
                    restart_level = True
                    break
                compact = [_compact_analysis(item) for item in batch]
                next_round.append(_merge_batch(endpoint, headers, model, source_url, compact))

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
    )


def generate_playbook(transcript_text: str, source_url: str = "") -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    chunk_chars = int(os.getenv("BRAIN_TRANSCRIPT_CHUNK_CHARS", "40000"))
    max_chunks = int(os.getenv("BRAIN_MAX_CHUNKS", "200"))
    merge_batch_size = int(os.getenv("BRAIN_MERGE_BATCH_SIZE", "8"))
    min_chunk_chars = int(os.getenv("BRAIN_MIN_CHUNK_CHARS", "8000"))

    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    if not transcript_text.strip():
        raise RuntimeError("Transcript is empty")

    endpoint = f"{groq_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if len(transcript_text) <= chunk_chars:
        return _single_pass_playbook(endpoint, headers, groq_llm_model, transcript_text, source_url)
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
    )
