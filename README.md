# trans

Production-ready Python CLI for:
- YouTube URL(s) or local audio input
- audio extraction + normalization + chunking
- chunk transcription via `gpt-4o-transcribe`
- merged transcript cleanup
- dataset export (`full`, `segmented`, `merged`) in JSONL/CSV

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# system deps: ffmpeg, ffprobe, yt-dlp
```

## Usage

```bash
python main.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --language "ar" \
  --dialect "moroccan_darija" \
  --output-dir "./out" \
  --dataset-format "jsonl"
```

Multiple inputs:

```bash
python main.py --url "..." --url "..."
python main.py --urls-file urls.txt
python main.py --audio-file ./sample.wav
```

Dry run/resume:

```bash
python main.py --audio-file ./sample.wav --dry-run
```

## Output layout

- `data/raw/<video_id>/` extracted audio + `metadata.json`
- `data/processed/<video_id>/audio.wav` normalized mono 16kHz wav
- `data/processed/<video_id>/chunks/chunk_0001.wav...`
- `data/transcripts/<video_id>/chunks/chunk_0001.response.json`
- `data/transcripts/<video_id>/chunks/chunk_0001.transcript.txt`
- `data/transcripts/<video_id>/merged_raw.txt`
- `data/transcripts/<video_id>/merged_clean.txt`
- `data/datasets/<video_id>.<mode>.jsonl|csv`

## Notes

- Use only content you have rights to process.
- `--dry-run` avoids API calls and writes placeholders.
- Re-runs skip chunk transcription outputs that already exist.
- Architecture keeps transcription response parsing isolated from segmentation/merging.
