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
) -> dict:
    chunks = _split_text(transcript_text, chunk_chars=chunk_chars, max_chunks=max_chunks)
    if not chunks:
        raise RuntimeError("Transcript is empty")

    analyses: list[dict] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        user_prompt = (
            f"Source URL: {source_url}\n"
            f"Chunk: {index}/{total}\n\n"
            f"Transcript chunk:\n{chunk}\n\n"
            "Produce the JSON now."
        )
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
        analyses.append(_normalize_payload(parsed))

    merge_prompt = (
        f"Source URL: {source_url}\n\n"
        "Combine these chunk analyses into one final playbook JSON:\n"
        f"{json.dumps(analyses, ensure_ascii=True)}\n\n"
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


def generate_playbook(transcript_text: str, source_url: str = "") -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    chunk_chars = int(os.getenv("BRAIN_TRANSCRIPT_CHUNK_CHARS", "40000"))
    max_chunks = int(os.getenv("BRAIN_MAX_CHUNKS", "200"))

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
    )
