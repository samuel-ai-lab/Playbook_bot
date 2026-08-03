import os
import re
from typing import Any

from notion_client import Client

MAX_BLOCKS_PER_REQUEST = 100


def _truncate_text(text: str, max_len: int = 1900) -> str:
    text = (text or "").strip()
    return text[:max_len]


def _paragraph_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": _truncate_text(text)},
                }
            ]
        },
    }


def _normalize_inline_lists(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    patterns = [
        (r"(?<=:)\s+([A-Z][^:;\n]{0,120}?)(?:,\s+|\s+and\s+)", "\n- "),
    ]

    normalized = cleaned
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized)

    return normalized


def _split_long_paragraph(part: str, target_sentences: int = 3) -> list[str]:
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
    sentences = [item.strip() for item in re.split(sentence_pattern, part) if item.strip()]
    if len(sentences) <= target_sentences:
        return [part.strip()]

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= target_sentences:
            chunks.append(" ".join(current).strip())
            current = []

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _split_paragraph_chunks(text: str, max_len: int = 1700) -> list[str]:
    cleaned = _normalize_inline_lists(text)
    if not cleaned:
        return []

    parts = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not parts:
        parts = [cleaned]

    chunks: list[str] = []
    for part in parts:
        subparts = [item.strip() for item in part.split("\n") if item.strip()]
        if not subparts:
            subparts = [part]

        expanded: list[str] = []
        for subpart in subparts:
            expanded.extend(_split_long_paragraph(subpart))

        for expanded_part in expanded:
            remaining = expanded_part
            while len(remaining) > max_len:
                split_at = remaining.rfind(". ", 0, max_len)
                if split_at < int(max_len * 0.5):
                    split_at = remaining.rfind(" ", 0, max_len)
                if split_at <= 0:
                    split_at = max_len
                chunk = remaining[:split_at].strip()
                if chunk:
                    chunks.append(chunk)
                remaining = remaining[split_at:].strip()
            if remaining:
                chunks.append(remaining)
    return chunks


def _content_blocks(text: str) -> list[dict[str, Any]]:
    cleaned = _normalize_inline_lists(text)
    if not cleaned:
        return []

    blocks: list[dict[str, Any]] = []
    parts = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not parts:
        parts = [cleaned]

    for part in parts:
        lines = [line.strip() for line in part.split("\n") if line.strip()]
        if not lines:
            continue

        paragraph_buffer: list[str] = []
        for line in lines:
            if line.startswith("- "):
                if paragraph_buffer:
                    blocks.extend(_paragraph_blocks("\n".join(paragraph_buffer)))
                    paragraph_buffer = []
                blocks.append(_bulleted_block(line[2:].strip()))
            else:
                paragraph_buffer.append(line)

        if paragraph_buffer:
            blocks.extend(_paragraph_blocks("\n".join(paragraph_buffer)))

    return blocks


def _paragraph_blocks(text: str) -> list[dict[str, Any]]:
    return [_paragraph_block(chunk) for chunk in _split_paragraph_chunks(text)]


def _bulleted_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": _truncate_text(text)},
                }
            ]
        },
    }


def _heading_block(text: str, level: int = 2) -> dict[str, Any]:
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": _truncate_text(text, 200)},
                }
            ]
        },
    }


def _build_blocks(playbook: dict, source_url: str = "") -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    introduction = str(playbook.get("introduction", "")).strip()
    if not introduction:
        introduction = str(playbook.get("summary", "")).strip()

    if introduction:
        blocks.append(_heading_block("Overview", level=2))
        blocks.extend(_content_blocks(introduction))

    sections = playbook.get("sections", [])
    if not isinstance(sections, list):
        sections = []

    if not sections:
        legacy_content = str(playbook.get("summary", "")).strip()
        legacy_notes = playbook.get("action_steps", [])
        if legacy_content or legacy_notes:
            sections = [
                {
                    "heading": "Core Playbook",
                    "content": legacy_content,
                    "notes_pack": legacy_notes if isinstance(legacy_notes, list) else [],
                }
            ]

    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading", "")).strip() or "Key Section"
        content = str(section.get("content", "")).strip()
        notes_pack = section.get("notes_pack", [])

        blocks.append(_heading_block(heading, level=2))
        if content:
            blocks.extend(_content_blocks(content))

        if isinstance(notes_pack, list):
            notes = [str(note).strip() for note in notes_pack if str(note).strip()]
            if notes:
                for note in notes:
                    blocks.append(_bulleted_block(note))

    checklist = playbook.get("implementation_checklist", [])
    if isinstance(checklist, list):
        checklist_items = [str(item).strip() for item in checklist if str(item).strip()]
        if checklist_items:
            blocks.append(_heading_block("Implementation Checklist", level=2))
            for item in checklist_items:
                blocks.append(_bulleted_block(item))

    conclusion = str(playbook.get("conclusion", "")).strip()
    if conclusion:
        blocks.append(_heading_block("Conclusion", level=2))
        blocks.extend(_content_blocks(conclusion))

    tags_raw = playbook.get("tags", [])
    tags = [str(tag).strip() for tag in tags_raw] if isinstance(tags_raw, list) else []
    tags = [tag for tag in tags if tag]
    if tags:
        blocks.append(_heading_block("Tags", level=3))
        blocks.append(_paragraph_block(", ".join(tags)))

    if source_url:
        blocks.append(_heading_block("Source", level=3))
        blocks.append(
            {
                "object": "block",
                "type": "bookmark",
                "bookmark": {"url": source_url},
            }
        )

    return blocks


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _title_property_name(db_schema: dict[str, Any]) -> str:
    for name, metadata in db_schema.items():
        if metadata.get("type") == "title":
            return name
    return "Name"


def _find_tags_property(db_schema: dict[str, Any]) -> tuple[str, str] | None:
    preferred_name = os.getenv("NOTION_TAGS_PROP", "Tags").strip() or "Tags"

    for name, metadata in db_schema.items():
        if name.strip().lower() != preferred_name.lower():
            continue
        property_type = metadata.get("type")
        if property_type in {"multi_select", "select"}:
            return name, property_type

    for name, metadata in db_schema.items():
        property_type = metadata.get("type")
        if property_type not in {"multi_select", "select"}:
            continue
        if "tag" in name.strip().lower():
            return name, property_type

    for name, metadata in db_schema.items():
        property_type = metadata.get("type")
        if property_type in {"multi_select", "select"}:
            return name, property_type

    return None


def _ensure_multi_select_options(
    notion: Client,
    db_id: str,
    db_schema: dict[str, Any],
    property_name: str,
    tags: list[str],
) -> dict[str, Any]:
    property_meta = db_schema.get(property_name, {})
    multi_select_meta = property_meta.get("multi_select", {}) if isinstance(property_meta, dict) else {}
    options = multi_select_meta.get("options", []) if isinstance(multi_select_meta, dict) else []

    existing_by_name = {
        str(option.get("name", "")).strip().lower(): option
        for option in options
        if isinstance(option, dict) and str(option.get("name", "")).strip()
    }
    missing_tags = [tag for tag in tags if tag.lower() not in existing_by_name]
    if not missing_tags:
        return db_schema

    preserved_options: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("id", "")).strip()
        option_name = str(option.get("name", "")).strip()
        if option_id:
            preserved_options.append({"id": option_id, "name": option_name} if option_name else {"id": option_id})
        elif option_name:
            preserved_options.append({"name": option_name})

    new_options = [{"name": _truncate_text(tag, 100)} for tag in missing_tags]
    updated_db = notion.databases.update(
        database_id=db_id,
        properties={
            property_name: {
                "multi_select": {
                    "options": preserved_options + new_options,
                }
            }
        },
    )
    return updated_db.get("properties", {}) if isinstance(updated_db, dict) else db_schema


def _page_properties_for_database(
    notion: Client,
    db_id: str,
    playbook: dict,
    notion_tags: list[str] | None = None,
) -> dict[str, Any]:
    db = notion.databases.retrieve(database_id=db_id)
    db_schema = db.get("properties", {})

    title_prop = _title_property_name(db_schema)
    properties: dict[str, Any] = {
        title_prop: {
            "title": [
                {
                    "type": "text",
                    "text": {"content": _truncate_text(playbook.get("title", "Untitled Playbook"), 200)},
                }
            ]
        }
    }

    tags_prop = _find_tags_property(db_schema)
    tags_source = notion_tags if notion_tags is not None else []
    tags = [str(tag).strip() for tag in tags_source] if isinstance(tags_source, list) else []
    tags = [tag for tag in tags if tag]
    if tags_prop and tags:
        tags_prop_name, tags_prop_type = tags_prop
        if tags_prop_type == "multi_select":
            db_schema = _ensure_multi_select_options(notion, db_id, db_schema, tags_prop_name, tags)
            properties[tags_prop_name] = {"multi_select": [{"name": _truncate_text(tag, 100)} for tag in tags]}
        elif tags_prop_type == "select":
            properties[tags_prop_name] = {"select": {"name": _truncate_text(tags[0], 100)}}

    return properties


def publish_playbook(playbook: dict, source_url: str = "", notion_tags: list[str] | None = None) -> str:
    notion_token = os.getenv("NOTION_TOKEN")
    notion_db_id = os.getenv("NOTION_DB_ID")

    if not notion_token:
        raise RuntimeError("Missing NOTION_TOKEN")
    if not notion_db_id:
        raise RuntimeError("Missing NOTION_DB_ID")

    notion = Client(auth=notion_token)

    page = notion.pages.create(
        parent={"database_id": notion_db_id},
        properties=_page_properties_for_database(notion, notion_db_id, playbook, notion_tags=notion_tags),
    )
    page_id = page["id"]

    blocks = _build_blocks(playbook, source_url=source_url)
    for batch in _chunk(blocks, MAX_BLOCKS_PER_REQUEST):
        notion.blocks.children.append(block_id=page_id, children=batch)

    return page.get("url", "")
