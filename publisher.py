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


def _page_title_property(playbook: dict) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": {"content": _truncate_text(playbook.get("title", "Untitled Playbook"), 200)},
        }
    ]


def publish_playbook(playbook: dict, source_url: str = "") -> str:
    notion_token = os.getenv("NOTION_TOKEN")
    notion_parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID")

    if not notion_token:
        raise RuntimeError("Missing NOTION_TOKEN")
    if not notion_parent_page_id:
        raise RuntimeError("Missing NOTION_PARENT_PAGE_ID")

    notion = Client(auth=notion_token)

    page = notion.pages.create(
        parent={"page_id": notion_parent_page_id},
        properties={"title": _page_title_property(playbook)},
    )
    page_id = page["id"]

    blocks = _build_blocks(playbook, source_url=source_url)
    for batch in _chunk(blocks, MAX_BLOCKS_PER_REQUEST):
        notion.blocks.children.append(block_id=page_id, children=batch)

    return page.get("url", "")
