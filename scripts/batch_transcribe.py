# scripts/batch_transcribe.py
import os, json, time, shutil, argparse
from pathlib import Path
import whisper

# --- Paths relative to this file ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}

def save_outputs(result: dict, out_base: Path, src: Path, model_name: str):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # Whisper-style outputs
    with open(out_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip())
    # Echowhisper meta
    meta = {
        "source_file": str(src),
        "transcript_json": str(out_base.with_suffix(".json")),
        "created_at": time.time(),
        "whisper_model": model_name,
    }
    with open(out_base.with_suffix(".ewmeta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_dir",  default=str(PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--out", dest="out_dir", default=str(PROJECT_ROOT / "data" / "transcripts"))
    ap.add_argument("--model", default=os.environ.get("EW_WHISPER_MODEL", "small"))
    args = ap.parse_args()

    in_dir  = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    print(f"[Echowhisper] Input:  {in_dir}")
    print(f"[Echowhisper] Output: {out_dir}")
    print(f"[Echowhisper] Model:  {args.model}")

    # Ensure ffmpeg is available (new shell may be needed after installing)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found in PATH. Install via `winget install Gyan.FFmpeg`, then reopen the shell.")

    if not in_dir.exists():
        print("!! Input directory does not exist:", in_dir)
        return

    files = sorted(p for p in in_dir.rglob("*") if p.is_file() and is_audio(p))
    print(f"[Echowhisper] Found {len(files)} audio file(s).")
    if not files:
        return

    # Load model ONCE
    print(f">> Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    for fp in files:
        out_base = out_dir / fp.stem
        # Skip if already processed (exists .json)
        if out_base.with_suffix(".json").exists():
            print(f"-- Skipping (exists): {fp.name}")
            continue

        print(f">> Transcribing: {fp.name}")
        result = model.transcribe(str(fp), language="en", task="transcribe")
        save_outputs(result, out_base, src=fp, model_name=args.model)

if __name__ == "__main__":
    main()
