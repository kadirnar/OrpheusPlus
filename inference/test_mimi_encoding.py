"""Verify Mimi codec encode/decode on a synthetic + (optional) real sample.

Run: .venv/bin/python inference/test_mimi_encoding.py
"""
import math
import sys
import numpy as np
import torch
from transformers import MimiModel, AutoFeatureExtractor

MODEL_ID = "kyutai/mimi"


def make_sine(seconds: float, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    noise = ref - est
    p_ref = float(np.mean(ref**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    return 10.0 * math.log10(p_ref / p_noise)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {MODEL_ID} on {device}")
    model = MimiModel.from_pretrained(MODEL_ID).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    cfg = model.config
    sr = fe.sampling_rate
    n_q_total = cfg.num_quantizers
    n_q_semantic = getattr(cfg, "num_semantic_quantizers", None)
    cb_size = cfg.codebook_size
    frame_rate = getattr(cfg, "frame_rate", None)

    print(f"[cfg] sampling_rate={sr}  num_quantizers={n_q_total}  "
          f"num_semantic_quantizers={n_q_semantic}  codebook_size={cb_size}  "
          f"frame_rate={frame_rate}")

    seconds = 2.0
    audio = make_sine(seconds, sr)
    inputs = fe(raw_audio=audio, sampling_rate=sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    print(f"[input] shape={tuple(input_values.shape)} dtype={input_values.dtype}")

    # --- Default (all codebooks) ---
    with torch.no_grad():
        enc = model.encode(input_values)
    codes = enc.audio_codes
    print(f"[encode default] codes shape={tuple(codes.shape)}  "
          f"min={int(codes.min())} max={int(codes.max())}")
    assert codes.dtype in (torch.long, torch.int32, torch.int64), codes.dtype
    assert codes.min() >= 0 and codes.max() < cb_size, "code OOR"
    assert codes.shape[1] == n_q_total, (codes.shape, n_q_total)
    expected_T = int(round(seconds * frame_rate)) if frame_rate else None
    print(f"[encode default] T_frames={codes.shape[-1]} (expected ~{expected_T})")

    with torch.no_grad():
        dec = model.decode(codes).audio_values
    rec = dec.squeeze().cpu().numpy().astype(np.float32)
    print(f"[decode default] out shape={tuple(dec.shape)}  "
          f"snr_vs_input={snr_db(audio, rec):.2f} dB")

    # --- 8-codebook (Moshi-style) ---
    n_q_use = 8
    with torch.no_grad():
        enc8 = model.encode(input_values, num_quantizers=n_q_use)
    codes8 = enc8.audio_codes
    print(f"[encode q={n_q_use}] codes shape={tuple(codes8.shape)}  "
          f"min={int(codes8.min())} max={int(codes8.max())}")
    assert codes8.shape[1] == n_q_use
    with torch.no_grad():
        dec8 = model.decode(codes8).audio_values
    rec8 = dec8.squeeze().cpu().numpy().astype(np.float32)
    print(f"[decode q={n_q_use}] snr_vs_input={snr_db(audio, rec8):.2f} dB")

    # --- Round-trip via flatten/unflatten (the layout used to inject codes
    # into an LM token stream: per-frame interleaved across codebooks) ---
    flat = codes8[0].transpose(0, 1).reshape(-1).cpu().numpy()  # (T*n_q,)
    print(f"[flatten q={n_q_use}] tokens={flat.shape[0]}  "
          f"unique_codes={int(np.unique(flat).size)}")
    rebuilt = torch.from_numpy(flat).view(-1, n_q_use).transpose(0, 1).unsqueeze(0).to(device)
    assert torch.equal(rebuilt, codes8), "flatten/unflatten round-trip mismatch"
    print("[flatten] round-trip OK")

    # --- Vocab-expansion sanity for the LM tokenizer ---
    # SNAC layout was: 7 codebooks * 4096 codes + 10 specials = 28682 tokens.
    # Mimi (8 codebooks * codebook_size + 10 specials):
    add_tokens_8 = n_q_use * cb_size + 10
    add_tokens_full = n_q_total * cb_size + 10
    print(f"[lm-vocab] mimi-8q  add_tokens = {n_q_use}*{cb_size}+10 = {add_tokens_8}")
    print(f"[lm-vocab] mimi-{n_q_total}q add_tokens = {n_q_total}*{cb_size}+10 = {add_tokens_full}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
