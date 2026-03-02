import os
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

    blocks.append(_heading_block("Summary", level=2))
    blocks.append(_paragraph_block(playbook.get("summary", "")))

    blocks.append(_heading_block("Action Steps", level=2))
    for step in playbook.get("action_steps", []):
        blocks.append(_bulleted_block(step))

    insights = playbook.get("insights", [])
    if insights:
        blocks.append(_heading_block("Insights", level=2))
        for insight in insights:
            blocks.append(_bulleted_block(insight))

    tags = playbook.get("tags", [])
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
    tags = playbook.get("tags", [])
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
