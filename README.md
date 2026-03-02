# Playbook Factory

Automates extraction of transcripts from YouTube/Instagram and publishes structured playbooks to Notion.

## Configuration (.env)

Create a `.env` file in the project root (you can copy `.env.example`).

### Required variables

- `GROQ_API_KEY`
- `NOTION_TOKEN`
- `NOTION_DB_ID` (database where each generated playbook is created as a page item)
- `G_SHEETS_JSON` (stringified Google service account JSON)
  or `G_SHEETS_JSON_B64` (base64-encoded service account JSON)
- `SHEET_ID`

### Optional variables

- `WORKSHEET_NAME` (default: first sheet)
- `SHEET_STATUS_COL` (default: `Status`)
- `SHEET_URL_COL` (default: `URL`)
- `SHEET_NOTION_URL_COL` (default: `Notion URL`)
- `SHEET_TITLE_COL` (default: `Notion Title`)
- `SHEET_ERROR_COL` (default: `Error`)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `GROQ_LLM_MODEL` (default: `openai/gpt-oss-120b`)
- `BRAIN_TRANSCRIPT_CHUNK_CHARS` (default: `40000`; LLM chunk size for full transcript processing)
- `BRAIN_MAX_CHUNKS` (default: `200`; safety cap for extreme transcript lengths)
- `BRAIN_MERGE_BATCH_SIZE` (default: `8`; chunk-analysis merge batch size to avoid payload limits)
- `BRAIN_MIN_CHUNK_CHARS` (default: `8000`; smallest auto-split chunk size before failing)
- `GROQ_WHISPER_MODEL` (default: `whisper-large-v3`)
- `USE_GROQ_WHISPER` (default: `true`)
- `GROQ_MAX_UPLOAD_BYTES` (default: `24000000`)
- `GROQ_AUDIO_CHUNK_SECONDS` (default: `900`)
- `YTDLP_COOKIES_FILE` (optional; path to exported Netscape cookies file)
- `YTDLP_COOKIES_FROM_BROWSER` (optional; e.g. `chrome`, `brave`, `firefox`, `safari`)

## Extraction notes

- Cobalt is not used.
- URL media extraction is handled by `yt-dlp`.
- If a YouTube video has no caption track, the pipeline falls back to `yt-dlp` + Whisper transcription.
- If YouTube fallback returns `403 Forbidden`, set `YTDLP_COOKIES_FROM_BROWSER` (or `YTDLP_COOKIES_FILE`) and retry.
- If Groq returns `413 Request Entity Too Large`, the pipeline auto-compresses/chunks audio (requires `ffmpeg`).
- Transcript intelligence is fully end-to-end: long transcripts are processed in chunks and merged into one final playbook.

## Local run

```bash
cp .env.example .env
# edit .env with your values
python3 -m pip install -r requirements.txt
python3 main.py
```

## GitHub Actions run

Set one GitHub Actions secret:
- `PLAYBOOK_FACTORY_ENV`: entire `.env` content as a multiline secret value.

Optional for YouTube bot-check protected videos:
- `YTDLP_COOKIES_TXT`: full exported `cookies.txt` content from a signed-in browser session.

## Google Sheet columns

Minimum columns:
- `URL`
- `Status`

Recommended extra columns:
- `Notion URL`
- `Notion Title`
- `Error`

Rows with `Status = New` will be processed.
