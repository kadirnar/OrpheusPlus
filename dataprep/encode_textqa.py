"""Encode SynDataLab/DeepSeekFlash-3M-{en,ja} into Qwen3 text-QA `input_ids`.

These become the text_QA_dataset stream for pretrain/train.py, so the LM keeps
its text reasoning while learning TTS. We mirror the Orpheus TTS layout but the
AI turn is a TEXT answer instead of speech (no SOS/audio/EOS_speech):

    TTS : [SOH] text [eot] [EOH] [SOA] [SOS] <audio> [EOS_speech]
    QA  : [SOH] question [eot] [EOH] [SOA] answer [eot]

EN and JA shards are written side by side (en-*, ja-*) into one dir; train.py
shuffles them so both languages interleave through training.

Config hardcoded on purpose (no CLI args).
"""
import os
import time
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

SOURCES = [
    ("en", "SynDataLab/DeepSeekFlash-3M-en"),
    ("ja", "SynDataLab/DeepSeekFlash-3M-ja"),
]
TOKENIZER_NAME = "Qwen/Qwen3-4B"
OUTPUT_ROOT = "/scratch/kadirnar/textqa-qwen3"
OUTPUT_DIR = OUTPUT_ROOT + "/data"
LOG_PATH = OUTPUT_ROOT + "/encode.log"
NUM_WORKERS = 16
MAX_LEN = 2048                  # drop QA pairs longer than this (tokens)
MIN_LEN = 4
DELETE_SOURCE_CACHE = True
TOKEN = os.environ.get("HF_TOKEN")

# Orpheus special-token offsets (relative to base vocab) - must match training.
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def build_row(q, a, tok, base, eot):
    q_ids = tok(q, add_special_tokens=False).input_ids
    a_ids = tok(a, add_special_tokens=False).input_ids
    ids = ([base + SOH] + q_ids + [eot, base + EOH, base + SOA]
           + a_ids + [eot])
    return ids


def worker(args):
    lang, repo, fn = args
    out_path = os.path.join(OUTPUT_DIR, f"{lang}-{os.path.basename(fn)}")
    if os.path.exists(out_path):
        return ("skip", 0, 0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    base = len(tok)
    eot = tok.eos_token_id
    src = hf_hub_download(repo, fn, repo_type="dataset", token=TOKEN)

    t = pq.read_table(src, columns=["question", "answer"])
    qs = t.column("question").to_pylist()
    as_ = t.column("answer").to_pylist()
    rows, dropped = [], 0
    for q, a in zip(qs, as_):
        if not q or not a:
            dropped += 1
            continue
        ids = build_row(q, a, tok, base, eot)
        if len(ids) < MIN_LEN or len(ids) > MAX_LEN:
            dropped += 1
            continue
        rows.append(ids)

    tmp = out_path + ".tmp"
    pq.write_table(
        pa.table({"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}),
        tmp, compression="snappy")
    os.replace(tmp, out_path)
    if DELETE_SOURCE_CACHE:
        try:
            os.remove(os.path.realpath(src))
        except OSError:
            pass
    return ("ok", len(rows), dropped)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    for lang, repo in SOURCES:
        shards = sorted(f for f in list_repo_files(repo, repo_type="dataset",
                                                   token=TOKEN)
                        if f.endswith(".parquet"))
        tasks += [(lang, repo, fn) for fn in shards]
    _log("encoding %d shards (%d sources) -> %s with %d workers"
         % (len(tasks), len(SOURCES), OUTPUT_DIR, NUM_WORKERS))

    done = rows = dropped = 0
    t0 = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        for status, n, d in pool.imap_unordered(worker, tasks):
            done += 1
            rows += n
            dropped += d
            if done % 10 == 0 or done == len(tasks):
                _log("%d/%d shards | %d rows | %d dropped | %.0fs"
                     % (done, len(tasks), rows, dropped, time.time() - t0))
    n_out = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")])
    _log("DONE | %d output shards | %d rows total" % (n_out, rows))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
