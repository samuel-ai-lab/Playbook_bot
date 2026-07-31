import base64
import json
import os
import traceback
from typing import Any

import gspread
import config  # noqa: F401

from brain import generate_playbook
from extractors import extract_transcript
from publisher import publish_playbook

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "NOTION_TOKEN",
    "NOTION_DB_ID",
    "SHEET_ID",
]

STATUS_NEW = "New"
STATUS_DONE = "Done"
STATUS_ERROR = "Error"
STATUS_IN_PROGRESS = "In Progress"

STATUS_COL = os.getenv("SHEET_STATUS_COL", "Status")
URL_COL = os.getenv("SHEET_URL_COL", "URL")
NOTION_URL_COL = os.getenv("SHEET_NOTION_URL_COL", "Notion URL")
TITLE_COL = os.getenv("SHEET_TITLE_COL", "Notion Title")
ERROR_COL = os.getenv("SHEET_ERROR_COL", "Error")
TAGS_COL = os.getenv("SHEET_TAGS_COL", "Tags")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "")


def _require_env() -> None:
    missing = [key for key in REQUIRED_ENV_VARS if not os.getenv(key)]
    if not os.getenv("G_SHEETS_JSON") and not os.getenv("G_SHEETS_JSON_B64"):
        missing.append("G_SHEETS_JSON or G_SHEETS_JSON_B64")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def _parse_sheets_creds() -> dict[str, Any]:
    raw = os.getenv("G_SHEETS_JSON", "").strip()
    raw_b64 = os.getenv("G_SHEETS_JSON_B64", "").strip()

    candidates: list[str] = []

    if raw_b64:
        try:
            decoded = base64.b64decode(raw_b64).decode("utf-8")
            candidates.append(decoded)
        except Exception:
            pass

    if raw:
        candidates.append(raw)
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            candidates.append(raw[1:-1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "Google Sheets credentials are malformed. Set G_SHEETS_JSON as valid JSON with double quotes, "
        "or set G_SHEETS_JSON_B64 as base64-encoded JSON."
    )


def _sheet_client() -> gspread.Worksheet:
    creds = _parse_sheets_creds()

    client = gspread.service_account_from_dict(creds)
    spreadsheet = client.open_by_key(os.getenv("SHEET_ID", ""))

    if WORKSHEET_NAME:
        return spreadsheet.worksheet(WORKSHEET_NAME)
    return spreadsheet.sheet1


def _headers(worksheet: gspread.Worksheet) -> dict[str, int]:
    header_row = worksheet.row_values(1)
    return {name.strip(): index + 1 for index, name in enumerate(header_row)}


def _cell_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    return str(value).strip() if value is not None else ""


def _update_if_present(
    worksheet: gspread.Worksheet,
    header_map: dict[str, int],
    row_number: int,
    column_name: str,
    value: str,
) -> None:
    col_index = header_map.get(column_name)
    if col_index:
        worksheet.update_cell(row_number, col_index, value)


def _parse_sheet_tags(raw_value: str) -> list[str]:
    raw = (raw_value or "").strip()
    if not raw:
        return []

    normalized = raw.replace("\n", ",").replace(";", ",").replace("|", ",")
    tags: list[str] = []
    seen: set[str] = set()

    for part in normalized.split(","):
        tag = part.strip()
        if not tag:
            continue
        dedupe_key = tag.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        tags.append(tag)

    return tags


def _new_rows(worksheet: gspread.Worksheet, header_map: dict[str, int]) -> list[tuple[int, dict[str, Any]]]:
    records = worksheet.get_all_records(default_blank="")
    rows: list[tuple[int, dict[str, Any]]] = []

    for index, row in enumerate(records, start=2):
        if _cell_value(row, STATUS_COL).lower() == STATUS_NEW.lower():
            rows.append((index, row))

    if STATUS_COL not in header_map:
        raise RuntimeError(f"Missing required column in sheet: {STATUS_COL}")
    if URL_COL not in header_map:
        raise RuntimeError(f"Missing required column in sheet: {URL_COL}")

    return rows


def _process_row(
    worksheet: gspread.Worksheet,
    header_map: dict[str, int],
    row_number: int,
    row: dict[str, Any],
) -> None:
    source_url = _cell_value(row, URL_COL)
    if not source_url:
        raise RuntimeError(f"Row {row_number} is missing URL")

    print(f"Processing row {row_number}: {source_url}")
    _update_if_present(worksheet, header_map, row_number, STATUS_COL, STATUS_IN_PROGRESS)

    transcript, metadata = extract_transcript(source_url)
    playbook = generate_playbook(
        transcript,
        source_url=source_url,
        duration_seconds=metadata.get("duration_seconds"),
    )

    sheet_tags = _parse_sheet_tags(_cell_value(row, TAGS_COL))
    if sheet_tags:
        playbook["tags"] = sheet_tags

    title = playbook.get("title", "").strip()
    if not title:
        default_title = f"Playbook - {metadata.get('source', 'source').capitalize()}"
        playbook["title"] = default_title

    notion_url = publish_playbook(playbook, source_url=source_url)

    _update_if_present(worksheet, header_map, row_number, STATUS_COL, STATUS_DONE)
    _update_if_present(worksheet, header_map, row_number, NOTION_URL_COL, notion_url)
    _update_if_present(worksheet, header_map, row_number, TITLE_COL, playbook.get("title", ""))
    _update_if_present(worksheet, header_map, row_number, ERROR_COL, "")


def run_once() -> None:
    _require_env()

    worksheet = _sheet_client()
    header_map = _headers(worksheet)
    pending_rows = _new_rows(worksheet, header_map)

    print(f"Found {len(pending_rows)} rows with status '{STATUS_NEW}'.")
    for row_number, row in pending_rows:
        try:
            _process_row(worksheet, header_map, row_number, row)
        except Exception as exc:
            print(f"Failed row {row_number}: {exc}")
            print(traceback.format_exc())
            _update_if_present(worksheet, header_map, row_number, STATUS_COL, STATUS_ERROR)
            _update_if_present(worksheet, header_map, row_number, ERROR_COL, str(exc)[:500])


if __name__ == "__main__":
    run_once()
