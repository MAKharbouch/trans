#!/usr/bin/env python3
"""YouTube/Audio -> transcription -> dataset CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputItem:
    source: str  # youtube|local
    value: str


@dataclass
class MediaMetadata:
    source: str
    original_url: str | None
    local_input_path: str | None
    video_id: str
    title: str
    channel: str | None
    upload_date: str | None
    duration: float | None


@dataclass
class ChunkResult:
    chunk_id: str
    audio_filepath: str
    duration_sec: float
    transcript: str
    response_path: str


def run_cmd(cmd: list[str], *, retries: int = 1, retry_delay: float = 2.0) -> subprocess.CompletedProcess[str]:
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            last_err = err
            if attempt < retries:
                time.sleep(retry_delay * attempt)
    assert last_err is not None
    raise last_err


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", sanitized)[:180] or "untitled"


def read_urls_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def resolve_inputs(args: argparse.Namespace) -> list[InputItem]:
    items: list[InputItem] = []
    for u in args.url or []:
        items.append(InputItem(source="youtube", value=u))
    if args.urls_file:
        for u in read_urls_file(Path(args.urls_file)):
            items.append(InputItem(source="youtube", value=u))
    if args.audio_file:
        items.append(InputItem(source="local", value=args.audio_file))
    if not items:
        raise ValueError("Provide at least one --url, --urls-file, or --audio-file")
    return items


def ensure_dirs(base: Path, video_id: str) -> dict[str, Path]:
    raw = base / "data" / "raw" / video_id
    processed = base / "data" / "processed" / video_id
    transcribed = base / "data" / "transcripts" / video_id
    datasets = base / "data" / "datasets"
    for p in [raw, processed / "chunks", transcribed / "chunks", datasets]:
        p.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "processed": processed, "transcribed": transcribed, "datasets": datasets}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_video_id(url: str) -> str:
    m = re.search(r"v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"yt_{digest}"


def download_audio(item: InputItem, base_out: Path) -> tuple[MediaMetadata, Path]:
    if item.source == "local":
        src = Path(item.value)
        if not src.exists():
            raise FileNotFoundError(f"Local input not found: {src}")
        vid = f"local_{hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:10]}"
        dirs = ensure_dirs(base_out, vid)
        ext = src.suffix or ".wav"
        dest = dirs["raw"] / f"input{ext}"
        if not dest.exists():
            shutil.copy2(src, dest)
        meta = MediaMetadata(
            source="local",
            original_url=None,
            local_input_path=str(src),
            video_id=vid,
            title=src.stem,
            channel=None,
            upload_date=None,
            duration=probe_duration(dest),
        )
        save_json(dirs["raw"] / "metadata.json", asdict(meta))
        return meta, dest

    url = item.value
    vid = extract_video_id(url)
    dirs = ensure_dirs(base_out, vid)
    meta_path = dirs["raw"] / "metadata.json"
    audio_path = dirs["raw"] / "audio.%(ext)s"

    info_path = dirs["raw"] / "yt_info.json"
    if not info_path.exists():
        cp = run_cmd(["yt-dlp", "--no-playlist", "--dump-single-json", url], retries=3)
        info_path.write_text(cp.stdout, encoding="utf-8")

    if not any(dirs["raw"].glob("audio.*")):
        run_cmd([
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            str(audio_path),
            url,
        ], retries=3)

    info = load_json(info_path)
    title = sanitize_filename(info.get("title") or vid)
    audio_files = sorted(dirs["raw"].glob("audio.*"))
    if not audio_files:
        raise RuntimeError(f"Audio extraction failed for {url}")
    audio_file = audio_files[0]
    meta = MediaMetadata(
        source="youtube",
        original_url=url,
        local_input_path=None,
        video_id=info.get("id") or vid,
        title=title,
        channel=info.get("channel") or info.get("uploader"),
        upload_date=info.get("upload_date"),
        duration=float(info["duration"]) if info.get("duration") else probe_duration(audio_file),
    )
    save_json(meta_path, asdict(meta))
    return meta, audio_file


def probe_duration(audio_path: Path) -> float:
    cp = run_cmd([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ])
    return float(cp.stdout.strip())


def preprocess_audio(base_out: Path, meta: MediaMetadata, input_audio: Path, chunk_minutes: float, overlap_sec: float) -> list[Path]:
    dirs = ensure_dirs(base_out, meta.video_id)
    mono = dirs["processed"] / "audio.wav"
    if not mono.exists():
        run_cmd([
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(mono),
        ])

    duration = probe_duration(mono)
    chunk_sec = chunk_minutes * 60.0
    chunks_dir = dirs["processed"] / "chunks"
    expected = math.ceil(duration / chunk_sec)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.wav"))
    if len(chunk_paths) >= expected:
        return chunk_paths

    start = 0.0
    idx = 1
    while start < duration:
        out = chunks_dir / f"chunk_{idx:04d}.wav"
        seg_len = min(chunk_sec + overlap_sec, duration - start)
        if not out.exists():
            run_cmd([
                "ffmpeg",
                "-y",
                "-i",
                str(mono),
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{seg_len:.3f}",
                str(out),
            ])
        start += chunk_sec
        idx += 1
    return sorted(chunks_dir.glob("chunk_*.wav"))


def transcribe_chunks(
    base_out: Path,
    meta: MediaMetadata,
    chunk_paths: list[Path],
    language: str | None,
    prompt_hint: str | None,
    dry_run: bool,
    request_logprobs: bool,
) -> list[ChunkResult]:
    dirs = ensure_dirs(base_out, meta.video_id)
    out_dir = dirs["transcribed"] / "chunks"

    client = None
    if not dry_run:
        from openai import OpenAI

        client = OpenAI()

    results: list[ChunkResult] = []
    for chunk in chunk_paths:
        chunk_id = chunk.stem
        resp_path = out_dir / f"{chunk_id}.response.json"
        txt_path = out_dir / f"{chunk_id}.transcript.txt"
        if txt_path.exists() and resp_path.exists():
            results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    audio_filepath=str(chunk),
                    duration_sec=probe_duration(chunk),
                    transcript=txt_path.read_text(encoding="utf-8"),
                    response_path=str(resp_path),
                )
            )
            continue

        if dry_run:
            transcript = f"[DRY RUN] Transcript placeholder for {chunk_id}."
            payload: dict[str, Any] = {"dry_run": True, "chunk_id": chunk_id, "text": transcript}
        else:
            assert client is not None
            payload = call_transcription_api(
                client=client,
                chunk_path=chunk,
                language=language,
                prompt_hint=prompt_hint,
                request_logprobs=request_logprobs,
            )
            transcript = extract_text(payload)

        txt_path.write_text(transcript.strip() + "\n", encoding="utf-8")
        save_json(resp_path, payload)
        results.append(
            ChunkResult(
                chunk_id=chunk_id,
                audio_filepath=str(chunk),
                duration_sec=probe_duration(chunk),
                transcript=transcript,
                response_path=str(resp_path),
            )
        )

    return sorted(results, key=lambda x: x.chunk_id)


def call_transcription_api(
    *,
    client: Any,
    chunk_path: Path,
    language: str | None,
    prompt_hint: str | None,
    request_logprobs: bool,
    max_retries: int = 5,
) -> dict[str, Any]:
    base_delay = 1.5
    for attempt in range(max_retries):
        try:
            with chunk_path.open("rb") as f:
                kwargs: dict[str, Any] = {
                    "model": "gpt-4o-transcribe",
                    "file": f,
                    "response_format": "json",
                }
                if language:
                    kwargs["language"] = language
                if prompt_hint:
                    kwargs["prompt"] = prompt_hint
                if request_logprobs:
                    kwargs["logprobs"] = True
                resp = client.audio.transcriptions.create(**kwargs)
                if hasattr(resp, "model_dump"):
                    return resp.model_dump()
                if isinstance(resp, dict):
                    return resp
                return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("Unreachable")


def extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("transcript"), str):
        return payload["transcript"]
    for key in ["output_text", "content"]:
        if isinstance(payload.get(key), str):
            return payload[key]
    return ""


def dedupe_overlap(prev_text: str, cur_text: str, max_tokens: int = 60) -> str:
    prev_tokens = prev_text.split()
    cur_tokens = cur_text.split()
    max_n = min(max_tokens, len(prev_tokens), len(cur_tokens))
    overlap = 0
    for n in range(max_n, 0, -1):
        if prev_tokens[-n:] == cur_tokens[:n]:
            overlap = n
            break
    return " ".join(cur_tokens[overlap:])


def merge_and_clean(base_out: Path, meta: MediaMetadata, chunks: list[ChunkResult]) -> tuple[Path, Path, str, str]:
    dirs = ensure_dirs(base_out, meta.video_id)
    merged_parts: list[str] = []
    prev = ""
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        txt = c.transcript.strip()
        if prev:
            txt = dedupe_overlap(prev, txt)
        merged_parts.append(txt.strip())
        prev = c.transcript.strip()

    merged_raw = "\n".join(p for p in merged_parts if p).strip()
    clean = clean_text(merged_raw)

    raw_path = dirs["transcribed"] / "merged_raw.txt"
    clean_path = dirs["transcribed"] / "merged_clean.txt"
    raw_path.write_text(merged_raw + "\n", encoding="utf-8")
    clean_path.write_text(clean + "\n", encoding="utf-8")
    return raw_path, clean_path, merged_raw, clean


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([!?.,؛،]){2,}", r"\1", text)
    return text.strip()


def split_text_segments(text: str, min_chars: int = 30, max_chars: int = 220) -> list[str]:
    parts = re.split(r"(?<=[.!?؟؛])\s+", text)
    segments: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = (buf + " " + part).strip() if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf and len(buf) >= min_chars:
                segments.append(buf)
                buf = part
            else:
                # hard split fallback
                for i in range(0, len(candidate), max_chars):
                    piece = candidate[i : i + max_chars].strip()
                    if len(piece) >= min_chars:
                        segments.append(piece)
                buf = ""
    if buf and len(buf) >= min_chars:
        segments.append(buf)
    return segments


def export_dataset(
    base_out: Path,
    meta: MediaMetadata,
    chunks: list[ChunkResult],
    merged_clean: str,
    language: str | None,
    dialect: str | None,
    fmt: str,
    modes: list[str],
) -> list[Path]:
    datasets_dir = ensure_dirs(base_out, meta.video_id)["datasets"]
    out_files: list[Path] = []

    for mode in modes:
        rows: list[dict[str, Any]] = []
        if mode == "full":
            for c in chunks:
                rows.append(
                    {
                        "id": f"{meta.video_id}_{c.chunk_id}",
                        "source": meta.source,
                        "video_id": meta.video_id,
                        "url": meta.original_url,
                        "title": meta.title,
                        "channel": meta.channel,
                        "language": language,
                        "dialect": dialect,
                        "audio_filepath": c.audio_filepath,
                        "text": c.transcript.strip(),
                        "duration_sec": round(c.duration_sec, 3),
                        "split": "train",
                    }
                )
        elif mode == "segmented":
            segments = split_text_segments(merged_clean)
            for i, seg in enumerate(segments, start=1):
                rows.append(
                    {
                        "id": f"{meta.video_id}_seg_{i:06d}",
                        "source": meta.source,
                        "video_id": meta.video_id,
                        "audio_filepath": "",
                        "text": seg,
                        "language": language,
                        "dialect": dialect,
                        "split": "train",
                    }
                )
        elif mode == "merged":
            rows.append(
                {
                    "id": f"{meta.video_id}_merged",
                    "source": meta.source,
                    "video_id": meta.video_id,
                    "url": meta.original_url,
                    "title": meta.title,
                    "channel": meta.channel,
                    "language": language,
                    "dialect": dialect,
                    "text": merged_clean,
                    "split": "train",
                }
            )
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        suffix = "jsonl" if fmt == "jsonl" else "csv"
        out_path = datasets_dir / f"{meta.video_id}.{mode}.{suffix}"
        if fmt == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            keys = sorted({k for row in rows for k in row.keys()})
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        out_files.append(out_path)

    return out_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube/Audio to transcription and dataset builder")
    p.add_argument("--url", action="append", help="YouTube URL (repeatable)")
    p.add_argument("--urls-file", help="Text file with one URL per line")
    p.add_argument("--audio-file", help="Local audio file path fallback input")
    p.add_argument("--language", help="Transcription language code, e.g. ar")
    p.add_argument("--dialect", help="Dialect tag, e.g. moroccan_darija")
    p.add_argument("--output-dir", default=".", help="Base output directory")
    p.add_argument("--chunk-minutes", type=float, default=10.0)
    p.add_argument("--overlap-seconds", type=float, default=2.0)
    p.add_argument("--prompt-hint", default="Moroccan Darija speech. Keep Arabic names and French words as spoken.")
    p.add_argument("--dataset-format", choices=["jsonl", "csv"], default="jsonl")
    p.add_argument("--dataset-modes", default="full,segmented,merged", help="Comma-separated: full,segmented,merged")
    p.add_argument("--dry-run", action="store_true", help="Skip OpenAI API call")
    p.add_argument("--request-logprobs", action="store_true")
    return p.parse_args()


def process_item(args: argparse.Namespace, item: InputItem) -> dict[str, Any]:
    out_root = Path(args.output_dir)
    meta, src_audio = download_audio(item, out_root)
    chunks = preprocess_audio(out_root, meta, src_audio, args.chunk_minutes, args.overlap_seconds)
    chunk_results = transcribe_chunks(
        out_root,
        meta,
        chunks,
        language=args.language,
        prompt_hint=args.prompt_hint,
        dry_run=args.dry_run,
        request_logprobs=args.request_logprobs,
    )
    raw_path, clean_path, _raw, clean = merge_and_clean(out_root, meta, chunk_results)
    modes = [m.strip() for m in args.dataset_modes.split(",") if m.strip()]
    dataset_files = export_dataset(
        out_root,
        meta,
        chunk_results,
        clean,
        language=args.language,
        dialect=args.dialect,
        fmt=args.dataset_format,
        modes=modes,
    )
    return {
        "video_id": meta.video_id,
        "metadata": str((out_root / "data" / "raw" / meta.video_id / "metadata.json")),
        "merged_raw": str(raw_path),
        "merged_clean": str(clean_path),
        "datasets": [str(p) for p in dataset_files],
    }


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args)
    summary = [process_item(args, item) for item in inputs]
    print(json.dumps({"processed": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
