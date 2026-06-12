"""Encode YT-Data/ja-en-syn into Qwen3 + Mimi `input_ids` for OrpheusPlus training.

pretrain/train.py expects a dataset whose rows are already token sequences.
This script turns every (audio, text) row of the raw merged dataset into one
Orpheus training sequence:

    [SOH] text_ids [eot] [EOH] [SOA] [SOS] <mimi audio tokens> [EOS_speech]

Audio is Mimi-encoded (8 codebooks x 2048), flattened per-frame interleaved.
Token-id scheme matches pretrain/train.py and inference/infer.py exactly:

    base_vocab          = len(Qwen3 tokenizer)         (151669 for Qwen3-4B)
    <custom_token_i>    = base_vocab + i
    SOS/EOS/SOH/EOH/SOA = base_vocab + 1/2/3/4/5
    audio (cb, code)    = base_vocab + 10 + cb*2048 + code

Each of the 8 B200s runs one worker process; source shards are split
round-robin. Output: one parquet per source shard holding a single
`input_ids` column, uploaded to TARGET_REPO (the training config points there).
Re-running skips shards whose output already exists.
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
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel

# --------------------------------------------------------------------------
# Config (hardcoded on purpose - no CLI args)
# --------------------------------------------------------------------------
SRC_REPO = "YT-Data/ja-en-syn"
TARGET_REPO = "YT-Data/ja-en-syn-qwen3-mimi32"
TOKENIZER_NAME = "Qwen/Qwen3-4B"
MIMI_MODEL_ID = "kyutai/mimi"
LOCAL_SRC_ROOT = "/scratch/kadirnar/ja-en-syn"          # data/ + data2/ already here
OUTPUT_ROOT = "/scratch/kadirnar/ja-en-syn-qwen3-mimi32"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
LOG_PATH = OUTPUT_ROOT + "/encode.log"
NUM_GPUS = 8
BATCH_SIZE = 32                 # clips per Mimi forward pass
MIN_AUDIO_SECONDS = 0.2         # skip clips shorter than this
MAX_AUDIO_SECONDS = 30.0        # skip clips longer than this
DELETE_SOURCE_CACHE = True      # drop downloaded data3 shards after encoding
PUSH_TO_HUB = False
TOKEN = os.environ.get("HF_TOKEN")

# Orpheus / Mimi token layout - must match pretrain/train.py + inference/infer.py
MIMI_NUM_CODEBOOKS = 32         # full Mimi RVQ depth (was 8) -> ~4x longer seqs
MIMI_CODEBOOK_SIZE = 2048
AUDIO_OFFSET = 10
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def list_shards():
    api = HfApi(token=TOKEN)
    files = [f for f in api.list_repo_files(SRC_REPO, repo_type="dataset")
             if f.endswith(".parquet")]
    files.sort()
    return files


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
    """codes_i: (8, T) numpy ints -> one Orpheus training token sequence."""
    c = codes_i[:, :n_frames]                       # trim padding frames
    flat = c.T.reshape(-1)                          # per-frame interleaved
    cb = np.arange(flat.size) % MIMI_NUM_CODEBOOKS
    audio_ids = flat + AUDIO_OFFSET + cb * MIMI_CODEBOOK_SIZE + base_vocab
    head = [base_vocab + SOH] + list(text_ids) + [
        eot, base_vocab + EOH, base_vocab + SOA, base_vocab + SOS]
    return head + audio_ids.astype(np.int64).tolist() + [base_vocab + EOS_SPEECH]


def process_shard(fn, ctx):
    tok, base_vocab, eot, mimi, fe, device, sr, hop = ctx
    out_path = os.path.join(OUTPUT_DIR, fn.replace("/", "-"))
    if os.path.exists(out_path):
        return "skip", 0, 0

    local = os.path.join(LOCAL_SRC_ROOT, fn)
    downloaded = False
    if os.path.exists(local):
        src = local
    else:
        src = hf_hub_download(SRC_REPO, fn, repo_type="dataset", token=TOKEN)
        downloaded = True

    table = pq.read_table(src, columns=["audio", "text"])
    audios = table.column("audio").to_pylist()
    texts = table.column("text").to_pylist()

    # --- decode audio (CPU); drop unusable / too-short / too-long clips ---
    waves, keep_text = [], []
    lo, hi = int(MIN_AUDIO_SECONDS * sr), int(MAX_AUDIO_SECONDS * sr)
    for au, tx in zip(audios, texts):
        if not au or not au.get("bytes") or not tx:
            continue
        try:
            w = decode_audio(au["bytes"], sr)
        except Exception:  # noqa: BLE001 - skip a corrupt clip, keep the shard
            continue
        if w.size < lo or w.size > hi:
            continue
        waves.append(w)
        keep_text.append(tx)

    # --- Mimi-encode in GPU batches, build token sequences ---
    rows = []
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
        codes = codes.cpu().numpy()                 # (B, 8, Tmax)
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

    if downloaded and DELETE_SOURCE_CACHE:
        for p in (src, os.path.realpath(src)):
            try:
                os.remove(p)
            except OSError:
                pass
    return "ok", len(rows), len(audios) - len(rows)


def worker(gpu_id, shards):
    device = "cuda:%d" % gpu_id
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    base_vocab = len(tok)
    eot = tok.eos_token_id
    mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(MIMI_MODEL_ID)
    sr = fe.sampling_rate
    hop = sr / mimi.config.frame_rate               # 24000 / 12.5 = 1920
    ctx = (tok, base_vocab, eot, mimi, fe, device, sr, hop)

    _log("gpu%d: start, %d shards (base_vocab=%d)" %
         (gpu_id, len(shards), base_vocab))
    done = rows = dropped = 0
    t0 = time.time()
    for fn in shards:
        try:
            status, n, skipped = process_shard(fn, ctx)
        except Exception as exc:  # noqa: BLE001
            _log("gpu%d ERROR %s: %r" % (gpu_id, fn, exc))
            continue
        done += 1
        rows += n
        dropped += skipped
        if done % 50 == 0 or done == len(shards):
            _log("gpu%d: %d/%d shards | %d rows | %d dropped | %.0fs" %
                 (gpu_id, done, len(shards), rows, dropped, time.time() - t0))
    _log("gpu%d: DONE %d shards | %d rows | %d dropped" %
         (gpu_id, done, rows, dropped))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shards = list_shards()
    _log("encoding %d shards -> %s on %d GPUs" %
         (len(shards), TARGET_REPO, NUM_GPUS))

    procs = []
    for g in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(g, shards[g::NUM_GPUS]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    n_out = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")])
    _log("all workers finished | %d output shards" % n_out)

    if PUSH_TO_HUB:
        api = HfApi(token=TOKEN)
        api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        readme = (
            "---\n"
            "license: apache-2.0\n"
            "task_categories:\n- text-to-speech\n"
            "language:\n- en\n- ja\n"
            "configs:\n- config_name: default\n  data_files:\n"
            "  - split: train\n    path: data/*.parquet\n"
            "---\n\n"
            "# ja-en-syn-qwen3-mimi\n\n"
            "`YT-Data/ja-en-syn` encoded for OrpheusPlus training: Qwen3-4B text\n"
            "tokens + Mimi audio tokens in the Orpheus prompt layout. Single\n"
            "column `input_ids`. Built by `encode_mimi.py`.\n")
        api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                        path_in_repo="README.md", repo_id=TARGET_REPO,
                        repo_type="dataset")
        api.upload_large_folder(repo_id=TARGET_REPO, repo_type="dataset",
                                folder_path=OUTPUT_ROOT,
                                allow_patterns=["data/*.parquet"])
        _log("DONE: pushed to %s" % TARGET_REPO)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
