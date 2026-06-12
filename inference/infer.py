"""Single-GPU, multilingual voice-clone inference for OrpheusPlus (Qwen3 + Mimi 32cb).

Give a reference clip (+ its text) and a target text in ANY language; the model
speaks the target text in the reference voice. Pure continuation cloning:

    prompt = [SOH] ref_text+target_text [eot] [EOH] [SOA] [SOS] <ref audio codes>
    model continues -> <target audio codes> [EOS_speech]

Edit CONFIG and JOBS below, then:  python infer.py   (no CLI args).
"""
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (AutoFeatureExtractor, AutoModelForCausalLM,
                          AutoTokenizer, MimiModel)

# --------------------------------------------------------------------------
# Config (hardcoded - no CLI args)
# --------------------------------------------------------------------------
CKPT = "/home/microway/kadirnar/github/OrpheusPlus/checkpoints/checkpoint-171622"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
MIMI_MODEL_ID = "kyutai/mimi"
OUTPUT_DIR = "/home/microway/kadirnar/github/OrpheusPlus/infer_out"
DEVICE = "cuda:0"

# Generation
MAX_NEW_TOKENS = 9600
MIN_NEW_TOKENS = 960            # floor avoids early EOS truncating the clip
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
MIN_OK_SECONDS = 1.5           # retry until at least this long (or MAX_RETRIES)
MAX_RETRIES = 3

# Mimi / Orpheus token layout - must match the training data (encode_mimi.py)
MIMI_NUM_CODEBOOKS = 32
MIMI_CODEBOOK_SIZE = 2048
AUDIO_OFFSET = 10
SOS, EOS_SPEECH, SOH, EOH, SOA = 1, 2, 3, 4, 5

# --------------------------------------------------------------------------
# Jobs: (out_name, ref_wav, ref_text, target_text). Any language works.
# --------------------------------------------------------------------------
REF_EN = "/home/microway/kadirnar/github/OrpheusPlus/andrew_00.wav"
REF_EN_TEXT = "all sorts of things, but mainly just browsing FaceTime and streaming most of all."
REF_JA = "/home/microway/kadirnar/github/OrpheusPlus/irodori-tts-v2-demo-2.wav"
REF_JA_TEXT = "ごめん、少し遅れてしまった。終電電車がとても混んでいたんだ。へいでも、ちゃんと来られてよかった。"

JOBS = [
    ("en_1", REF_EN, REF_EN_TEXT, "Hello, this is a test of the text to speech model."),
    ("en_2", REF_EN, REF_EN_TEXT, "Artificial intelligence is transforming the way we live and work."),
    ("ja_1", REF_JA, REF_JA_TEXT, "こんにちは、これは音声合成モデルのテストです。"),
    ("ja_2", REF_JA, REF_JA_TEXT, "桜の花が満開で、とても美しい春の一日でした。"),
]


def codes_to_lm_ids(codes, base):
    """Mimi audio_codes (1, nq, T) -> flat per-frame interleaved LM token ids."""
    c = codes[0].cpu()
    flat = c.transpose(0, 1).reshape(-1)            # (T*nq,) frame-major
    nq = c.shape[0]
    return [int(v) + AUDIO_OFFSET + (i % nq) * MIMI_CODEBOOK_SIZE + base
            for i, v in enumerate(flat.tolist())]


def lm_ids_to_codes(ids, base):
    """Generated LM token ids -> Mimi codes (1, nq, T); stop at first invalid."""
    valid = []
    for i, tid in enumerate(ids):
        cb = i % MIMI_NUM_CODEBOOKS
        v = tid - base - AUDIO_OFFSET - cb * MIMI_CODEBOOK_SIZE
        if 0 <= v < MIMI_CODEBOOK_SIZE:
            valid.append(v)
        else:
            break
    T = len(valid) // MIMI_NUM_CODEBOOKS
    if T == 0:
        return None
    arr = torch.tensor(valid[: T * MIMI_NUM_CODEBOOKS], dtype=torch.long)
    return arr.view(T, MIMI_NUM_CODEBOOKS).t().unsqueeze(0)


def encode_ref(path, mimi, fe, device):
    """Load + resample a reference wav and Mimi-encode it to audio_codes."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    tsr = fe.sampling_rate
    if sr != tsr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, tsr).squeeze(0).numpy()
    inp = fe(raw_audio=wav, sampling_rate=tsr, return_tensors="pt")
    with torch.no_grad():
        enc = mimi.encode(inp["input_values"].to(device),
                          num_quantizers=MIMI_NUM_CODEBOOKS)
    return enc.audio_codes


def synth(target_text, ref_lm, model, tok, mimi, base, eos_id, device):
    """Generate one cloned clip; retry for length, return (seconds, waveform)."""
    tids = tok(target_text, add_special_tokens=False).input_ids
    prompt = ([base + SOH] + tids
              + [tok.eos_token_id, base + EOH, base + SOA, base + SOS] + ref_lm)
    ii = torch.tensor([prompt], device=device)
    attn = torch.ones_like(ii)                      # eot==pad would be masked otherwise

    best = None
    for _ in range(MAX_RETRIES):
        with torch.no_grad():
            out = model.generate(
                ii, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=eos_id, pad_token_id=tok.eos_token_id)
        codes = lm_ids_to_codes(out[0, ii.shape[1]:].tolist(), base)
        if codes is None:
            continue
        with torch.no_grad():
            wav = mimi.decode(codes.to(device)).audio_values
        w = wav.squeeze().cpu().to(torch.float32).numpy()
        dur = w.shape[0] / mimi.config.sampling_rate
        if best is None or dur > best[0]:
            best = (dur, w)
        if dur >= MIN_OK_SECONDS:
            break
    return best


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.cuda.set_device(DEVICE)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    base = len(tok)                                 # base_vocab; audio ids start above
    n_add = MIMI_NUM_CODEBOOKS * MIMI_CODEBOOK_SIZE + AUDIO_OFFSET
    tok.add_tokens([f"<custom_token_{i}>" for i in range(0, n_add + 1)])
    eos_id = base + EOS_SPEECH

    model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(DEVICE).eval()
    mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(DEVICE).eval()
    fe = AutoFeatureExtractor.from_pretrained(MIMI_MODEL_ID)

    ref_cache = {}
    for name, ref_wav, ref_text, target in JOBS:
        if ref_wav not in ref_cache:
            ref_cache[ref_wav] = codes_to_lm_ids(
                encode_ref(ref_wav, mimi, fe, DEVICE), base)
        # ref_text primes the voice; target_text is what we want spoken.
        full = (ref_text + " " + target).strip()
        res = synth(full, ref_cache[ref_wav], model, tok, mimi, base, eos_id, DEVICE)
        if res is None:
            print(f"{name}: FAILED (no valid audio)", flush=True)
            continue
        out_path = os.path.join(OUTPUT_DIR, name + ".wav")
        sf.write(out_path, res[1], samplerate=mimi.config.sampling_rate)
        print(f"{name}: {res[0]:.2f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
