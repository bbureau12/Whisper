from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
import math
from datetime import timedelta

# ASR (faster-whisper)
from faster_whisper import WhisperModel
import numpy as np
# TEMP fix for older pyannote using np.NaN:
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "NAN"):
    np.NAN = np.nan
# Diarization (pyannote)
from pyannote.audio import Pipeline

AUDIO = r"D:\Projects\echoWhisper\Whisper\data\raw\test.wav"
MODEL_ASR = "small"   # change to "medium" for better quality
DEVICE = "cuda"       # or "cpu"



def to_srt_time(s):
    td = timedelta(seconds=max(0, s))
    # SRT requires , for milliseconds
    return str(td)[:-3].replace(".", ",")

def main():
    audio = Path(AUDIO)
    out_srt = audio.with_suffix(".spk.srt")

    # 1) Diarize (Speaker turns)
    print("Loading diarization pipeline...")
    dia = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization = dia(audio, min_speakers=2, max_speakers=8)  # adjust as needed

    # Collect speaker segments as [(start, end, speaker)]
    spk_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        spk_segments.append((turn.start, turn.end, speaker))

    # 2) Transcribe (word timestamps)
    print("Loading ASR model...")
    asr = WhisperModel(MODEL_ASR, device="cpu", compute_type="int8")  # CPU-safe
    segments, info = asr.transcribe(str(audio), task="transcribe", language="en",
                                    word_timestamps=True, vad_filter=True)

    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({
                "start": w.start,
                "end": w.end,
                "text": w.word
            })

    # 3) Assign each word to the overlapping speaker segment
    def find_speaker(t):
        for s, e, spk in spk_segments:
            if s <= t <= e:
                return spk
        return "UNK"

    # Group consecutive words with the same speaker into SRT cues
    cues = []
    if words:
        cur_spk = find_speaker(words[0]["start"])
        cur_start = words[0]["start"]
        cur_words = [words[0]["text"]]
        last_end = words[0]["end"]

        for w in words[1:]:
            spk = find_speaker(w["start"])
            gap = (w["start"] - last_end) if (last_end and w["start"] and w["start"] > last_end) else 0.0
            # new cue if speaker changes or large time gap
            if spk != cur_spk or gap > 1.0:
                cues.append((cur_start, last_end, cur_spk, " ".join(cur_words).strip()))
                cur_spk = spk
                cur_start = w["start"]
                cur_words = [w["text"]]
            else:
                cur_words.append(w["text"])
            last_end = w["end"]

        # flush final
        cues.append((cur_start, last_end, cur_spk, " ".join(cur_words).strip()))

    # 4) Write SRT
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, (s, e, spk, text) in enumerate(cues, start=1):
            f.write(f"{i}\n{to_srt_time(s)} --> {to_srt_time(e)}\n{spk}: {text}\n\n")

    print("Wrote speaker-labeled SRT:", out_srt)

if __name__ == "__main__":
    main()
