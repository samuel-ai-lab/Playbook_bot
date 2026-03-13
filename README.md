# Playbook Factory

Automates extraction of transcripts from YouTube/Instagram and publishes longform playbooks to Notion.

## Configuration (.env)

Create a `.env` file in the project root (you can copy `.env.example`).

### Required variables

- `OPENAI_API_KEY`
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
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (default: `gpt-5`)
- `OPENAI_MAX_OUTPUT_TOKENS` (default: `32000`)
- `OPENAI_TEXT_VERBOSITY` (default: `high`; options: `low`, `medium`, `high`)
- `OPENAI_REASONING_EFFORT` (optional; `minimal`, `low`, `medium`, `high`)
- `BRAIN_LLM_MAX_RETRIES` (default: `6`; retries on `429`/`5xx` or transient network errors)
- `BRAIN_LLM_RETRY_BASE_SEC` (default: `1.5`; exponential backoff base)
- `BRAIN_LLM_MAX_BACKOFF_SEC` (default: `45`; max wait between retries)
- `BRAIN_LLM_MIN_INTERVAL_SEC` (default: `0.35`; pacing between LLM requests)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `TRANSCRIPT_API_KEY` (optional; if set, YouTube transcript fetch uses transcriptapi.com first)
- `TRANSCRIPT_API_BASE_URL` (default: `https://transcriptapi.com`)
- `TRANSCRIPT_API_TIMEOUT_SECONDS` (default: `45`)
- `TRANSCRIPT_API_MAX_RETRIES` (default: `3`)
- `TRANSCRIPT_API_RETRY_BASE_SEC` (default: `1.5`)
- `GROQ_API_KEY` (optional; only needed when `USE_GROQ_WHISPER=true` and media transcription uses Groq Whisper)
- `GROQ_WHISPER_MODEL` (default: `whisper-large-v3`)
- `USE_GROQ_WHISPER` (default: `true`)
- `GROQ_MAX_UPLOAD_BYTES` (default: `24000000`)
- `GROQ_AUDIO_CHUNK_SECONDS` (default: `900`)
- `INSTAGRAM_EXTRACTOR_PROVIDER` (default: `apify`; options: `apify`, `yt-dlp`)
- `APIFY_TOKEN` (required when `INSTAGRAM_EXTRACTOR_PROVIDER=apify`)
- `APIFY_INSTAGRAM_ACTOR_ID` (default: `apify~instagram-reel-scraper`)
- `APIFY_INSTAGRAM_INPUT_MODE` (default: `auto`; options: `auto`, `username`, `urls`, `startUrls`, `directUrls`)
- `APIFY_INSTAGRAM_EXTRA_INPUT_JSON` (optional JSON object merged into the actor input)
- `APIFY_INSTAGRAM_TIMEOUT_SECONDS` (default: `300`)
- `INSTAGRAM_APIFY_FALLBACK_TO_YTDLP` (default: `false`)
- `YTDLP_COOKIES_FILE` (optional; path to exported Netscape cookies file)
- `YTDLP_COOKIES_FROM_BROWSER` (optional; e.g. `chrome`, `brave`, `firefox`, `safari`)
- `YOUTUBE_CAPTION_FALLBACK_ENABLED` (default: `false`; enables `yt-dlp` caption fallback for YouTube)
- `YOUTUBE_AUDIO_FALLBACK_ENABLED` (default: `false`; enables downloader+Whisper fallback for YouTube)
- `YOUTUBE_FETCH_DURATION` (default: `false`; when `true`, probes YouTube duration via `yt-dlp` for metadata)
- `YOUTUBE_DOWNLOADER` (default: `pytubefix`; options: `pytubefix`, `yt-dlp`, `auto`)

## Extraction notes

- Cobalt is not used.
- Instagram defaults to API-based extraction (`INSTAGRAM_EXTRACTOR_PROVIDER=apify`) to avoid cookie/login issues.
- Recommended IG actor for reel URLs: `apify~instagram-reel-scraper`.
- `apify~instagram-reel-scraper` requires `input.username`; this pipeline sends the reel URL in that field.
- If actor output already includes `transcript`, pipeline uses it directly; otherwise it downloads media URL and transcribes with Whisper.
- YouTube extraction order is:
  1) TranscriptAPI (if `TRANSCRIPT_API_KEY` is set)
  2) `youtube-transcript-api`
  3) `yt-dlp` subtitle/auto-caption extraction (only when `YOUTUBE_CAPTION_FALLBACK_ENABLED=true`)
  4) downloader + Whisper transcription (only when `YOUTUBE_AUDIO_FALLBACK_ENABLED=true`)
- By default, YouTube `yt-dlp` fallbacks are disabled.
- If YouTube fallback returns `403 Forbidden`, set `YTDLP_COOKIES_FROM_BROWSER` (or `YTDLP_COOKIES_FILE`) and retry.
- If Groq returns `413 Request Entity Too Large`, the pipeline auto-compresses/chunks audio (requires `ffmpeg`).
- Playbook writing now runs as a single-pass OpenAI generation (no transcript chunk/merge compression).
- If inline transcript input hits request-size limits, the writer automatically retries in file-input mode to preserve one-pass behavior.

## Output format

- Single mode only: longform playbook/article output (no short summary mode).
- Output is generated as structured JSON with:
  - `title`
  - `introduction`
  - `sections[]` (`heading`, `content`, `notes_pack[]`)
  - `conclusion`
  - `implementation_checklist[]`
  - `tags[]`
- Notion rendering writes sectioned narrative content with per-section notes-pack bullets.

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
  If it exceeds GitHub secret size limits, use split secrets instead:
  `YTDLP_COOKIES_TXT_P1` ... `YTDLP_COOKIES_TXT_P6`.

## Google Sheet columns

Minimum columns:
- `URL`
- `Status`

Recommended extra columns:
- `Notion URL`
- `Notion Title`
- `Error`

Rows with `Status = New` will be processed.
