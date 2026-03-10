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


def _split_paragraph_chunks(text: str, max_len: int = 1700) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    parts = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not parts:
        parts = [cleaned]

    chunks: list[str] = []
    for part in parts:
        remaining = part
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
        blocks.extend(_paragraph_blocks(introduction))

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
            blocks.extend(_paragraph_blocks(content))

        if isinstance(notes_pack, list):
            notes = [str(note).strip() for note in notes_pack if str(note).strip()]
            if notes:
                blocks.append(_heading_block("Notes Pack", level=3))
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
        blocks.extend(_paragraph_blocks(conclusion))

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


def _find_multi_select_name(db_schema: dict[str, Any]) -> str | None:
    for name, metadata in db_schema.items():
        if metadata.get("type") == "multi_select":
            return name
    return None


def _page_properties_for_database(notion: Client, db_id: str, playbook: dict) -> dict[str, Any]:
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

    tags_prop = _find_multi_select_name(db_schema)
    tags_raw = playbook.get("tags", [])
    tags = [str(tag).strip() for tag in tags_raw] if isinstance(tags_raw, list) else []
    tags = [tag for tag in tags if tag]
    if tags_prop and tags:
        properties[tags_prop] = {"multi_select": [{"name": _truncate_text(tag, 100)} for tag in tags]}

    return properties


def publish_playbook(playbook: dict, source_url: str = "") -> str:
    notion_token = os.getenv("NOTION_TOKEN")
    notion_db_id = os.getenv("NOTION_DB_ID")

    if not notion_token:
        raise RuntimeError("Missing NOTION_TOKEN")
    if not notion_db_id:
        raise RuntimeError("Missing NOTION_DB_ID")

    notion = Client(auth=notion_token)

    page = notion.pages.create(
        parent={"database_id": notion_db_id},
        properties=_page_properties_for_database(notion, notion_db_id, playbook),
    )
    page_id = page["id"]

    blocks = _build_blocks(playbook, source_url=source_url)
    for batch in _chunk(blocks, MAX_BLOCKS_PER_REQUEST):
        notion.blocks.children.append(block_id=page_id, children=batch)

    return page.get("url", "")
