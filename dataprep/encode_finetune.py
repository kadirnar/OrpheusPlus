"""Encode Vyvo/ElevenLabs-EN-iP95p4xoKVk53GoZ742B into Qwen3 + Mimi `input_ids`.

Single-speaker English finetune set (2000 clips, 44.1 kHz). Each (audio, text)
row becomes one Orpheus training sequence, IDENTICAL in layout to the pretrain
data (encode_mimi.py) so the pretrained checkpoint can be finetuned directly:

    [SOH] text_ids [eot] [EOH] [SOA] [SOS] <mimi audio tokens> [EOS_speech]

    base_vocab          = len(Qwen3 tokenizer)
    <custom_token_i>    = base_vocab + i
    SOS/EOS/SOH/EOH/SOA = base_vocab + 1/2/3/4/5
    audio (cb, code)    = base_vocab + 10 + cb*2048 + code

Multi-GPU: the 2000 rows are split round-robin across NUM_GPUS workers, each
writing its own parquet shard under OUTPUT_DIR. Config hardcoded - no CLI args.
"""
import io
import os
import time
import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel

# --------------------------------------------------------------------------
# Config (hardcoded on purpose - no CLI args)
# --------------------------------------------------------------------------
SRC_REPO = "Vyvo/ElevenLabs-EN-iP95p4xoKVk53GoZ742B"
SRC_FILE = "data/train-00000-of-00001.parquet"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"          # same vocab length as the trained model
MIMI_MODEL_ID = "kyutai/mimi"
OUTPUT_ROOT = "/scratch/kadirnar/elevenlabs-en-ft-mimi32"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
NUM_GPUS = 8
BATCH_SIZE = 32                 # clips per Mimi forward pass
MIN_AUDIO_SECONDS = 0.2         # skip clips shorter than this
MAX_AUDIO_SECONDS = 30.0        # skip clips longer than this
TOKEN = os.environ.get("HF_TOKEN")

# Orpheus / Mimi token layout - must match encode_mimi.py + pretrain/train.py
MIMI_NUM_CODEBOOKS = 32
MIMI_CODEBOOK_SIZE = 2048
AUDIO_OFFSET = 10
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5


def decode_audio(audio_bytes, target_sr):
    """Decode encoded audio bytes -> mono float32 waveform at target_sr."""
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
    except Exception:  # noqa: BLE001 - fall back to the ffmpeg/sox backend
        t, sr = torchaudio.load(io.BytesIO(audio_bytes))
        wav = t.mean(0).numpy().astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        t = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        wav = torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()
    return wav


def build_input_ids(codes_i, n_frames, text_ids, base_vocab, eot):
    """codes_i: (32, T) numpy ints -> one Orpheus training token sequence."""
    c = codes_i[:, :n_frames]                       # trim padding frames
    flat = c.T.reshape(-1)                          # per-frame interleaved
    cb = np.arange(flat.size) % MIMI_NUM_CODEBOOKS
    audio_ids = flat + AUDIO_OFFSET + cb * MIMI_CODEBOOK_SIZE + base_vocab
    head = [base_vocab + SOH] + list(text_ids) + [
        eot, base_vocab + EOH, base_vocab + SOA, base_vocab + SOS]
    return head + audio_ids.astype(np.int64).tolist() + [base_vocab + EOS_SPEECH]


def worker(gpu_id, audios, texts):
    out_path = os.path.join(OUTPUT_DIR, "train-%02d.parquet" % gpu_id)
    if os.path.exists(out_path):
        print("[gpu%d] output exists, skip" % gpu_id, flush=True)
        return

    device = "cuda:%d" % gpu_id
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    base_vocab = len(tok)
    eot = tok.eos_token_id
    mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(MIMI_MODEL_ID)
    sr = fe.sampling_rate
    hop = sr / mimi.config.frame_rate               # 24000 / 12.5 = 1920

    # --- decode audio (CPU); drop unusable / too-short / too-long clips ---
    waves, keep_text = [], []
    lo, hi = int(MIN_AUDIO_SECONDS * sr), int(MAX_AUDIO_SECONDS * sr)
    for au, tx in zip(audios, texts):
        if not au or not au.get("bytes") or not tx:
            continue
        try:
            w = decode_audio(au["bytes"], sr)
        except Exception:  # noqa: BLE001 - skip a corrupt clip
            continue
        if w.size < lo or w.size > hi:
            continue
        waves.append(w)
        keep_text.append(tx)
    print("[gpu%d] %d clips kept (base_vocab=%d)" % (gpu_id, len(waves), base_vocab),
          flush=True)

    # --- Mimi-encode in GPU batches, build token sequences ---
    rows = []
    t0 = time.time()
    for s in range(0, len(waves), BATCH_SIZE):
        chunk = waves[s:s + BATCH_SIZE]
        chunk_text = keep_text[s:s + BATCH_SIZE]
        inputs = fe(raw_audio=chunk, sampling_rate=sr,
                    return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(device)
        mask = inputs.get("padding_mask")
        mask = mask.to(device) if mask is not None else None
        with torch.no_grad():
            codes = mimi.encode(iv, padding_mask=mask,
                                num_quantizers=MIMI_NUM_CODEBOOKS).audio_codes
        codes = codes.cpu().numpy()                 # (B, 32, Tmax)
        for j, w in enumerate(chunk):
            n_frames = min(int(np.ceil(w.size / hop)), codes.shape[-1])
            if n_frames <= 0:
                continue
            text_ids = tok(chunk_text[j], add_special_tokens=False).input_ids
            rows.append(build_input_ids(codes[j], n_frames, text_ids,
                                        base_vocab, eot))

    tmp = out_path + ".tmp"
    pq.write_table(
        pa.table({"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}),
        tmp, compression="snappy")
    os.replace(tmp, out_path)
    print("[gpu%d] DONE %d rows -> %s | %.0fs" %
          (gpu_id, len(rows), out_path, time.time() - t0), flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src = hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset", token=TOKEN)
    table = pq.read_table(src, columns=["audio", "text"])
    audios = table.column("audio").to_pylist()
    texts = table.column("text").to_pylist()
    print("loaded %d rows -> %d GPUs" % (len(audios), NUM_GPUS))

    procs = []
    for g in range(NUM_GPUS):
        sub_a = audios[g::NUM_GPUS]
        sub_t = texts[g::NUM_GPUS]
        if not sub_a:
            continue
        p = mp.Process(target=worker, args=(g, sub_a, sub_t))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    n_rows = sum(pq.read_metadata(os.path.join(OUTPUT_DIR, f)).num_rows
                 for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet"))
    print("ALL DONE: %d total rows in %s" % (n_rows, OUTPUT_DIR))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
