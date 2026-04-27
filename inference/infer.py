"""Reference-audio voice-cloning inference for OrpheusPlus.

Matches the Orpheus prompt convention (canopyai/Orpheus-TTS, engine_class.py:82-84):
    <custom_token_3>  text  <|eot|>  <custom_token_4>  <custom_token_5>  <custom_token_1>  [audio]

Audio tokens are Mimi (8 codebooks * 2048) flattened per-frame interleaved.
LM token id = code + AUDIO_OFFSET + codebook_idx * MIMI_CODEBOOK_SIZE + base_vocab.

Run:
    .venv/bin/python inference/infer.py \
        --config configs/qwen3_config.yaml \
        --ckpt ./checkpoints \
        --ref-audio ref.wav --ref-text "Reference transcript" \
        --text "Target text to synthesize" \
        --out out.wav
"""
import argparse
import math
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from transformers import (AutoFeatureExtractor, AutoModelForCausalLM,
                          AutoTokenizer, MimiModel)

# --- Orpheus special-token offsets (relative to base LLM vocab). Do not change. ---
START_OF_SPEECH = 1   # <custom_token_1>  — Orpheus SOS
END_OF_SPEECH   = 2   # <custom_token_2>  — Orpheus EOS_speech (stop signal)
START_OF_HUMAN  = 3   # <custom_token_3>
END_OF_HUMAN    = 4   # <custom_token_4>
START_OF_AI     = 5   # <custom_token_5>

# --- Mimi codec layout, must match pretrain/train.py:264 ---
MIMI_NUM_CODEBOOKS = 8
MIMI_CODEBOOK_SIZE = 2048
AUDIO_OFFSET       = 10   # 10 reserved special slots before audio tokens
MIMI_MODEL_ID      = "kyutai/mimi"


def load_reference_codes(wav_path: str, mimi: MimiModel, fe, device: str) -> torch.Tensor:
    """Encode a reference wav with Mimi → (1, 8, T_ref) integer codes."""
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    target_sr = fe.sampling_rate
    if sr != target_sr:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_t, sr, target_sr).squeeze(0).numpy()
    inputs = fe(raw_audio=wav, sampling_rate=target_sr, return_tensors="pt")
    with torch.no_grad():
        enc = mimi.encode(inputs["input_values"].to(device),
                          num_quantizers=MIMI_NUM_CODEBOOKS)
    return enc.audio_codes  # (1, 8, T_ref)


def codes_to_lm_ids(codes: torch.Tensor, base_vocab: int) -> list[int]:
    """(1, 8, T) Mimi codes → flat list of LM token ids, per-frame interleaved."""
    c = codes[0].cpu()                     # (8, T)
    n_q, T = c.shape
    flat = c.transpose(0, 1).reshape(-1)   # (T*8,) = c0[0],c1[0],...,c7[0],c0[1],...
    out = []
    for i, code in enumerate(flat.tolist()):
        cb = i % n_q
        out.append(code + AUDIO_OFFSET + cb * MIMI_CODEBOOK_SIZE + base_vocab)
    return out


def lm_ids_to_codes(ids: list[int], base_vocab: int) -> torch.Tensor | None:
    """Inverse of codes_to_lm_ids. Truncates to a multiple of 8; stops at first OOR."""
    valid = []
    for i, tid in enumerate(ids):
        cb = i % MIMI_NUM_CODEBOOKS
        v = tid - base_vocab - AUDIO_OFFSET - cb * MIMI_CODEBOOK_SIZE
        if 0 <= v < MIMI_CODEBOOK_SIZE:
            valid.append(v)
        else:
            break
    T = len(valid) // MIMI_NUM_CODEBOOKS
    if T == 0:
        return None
    arr = torch.tensor(valid[: T * MIMI_NUM_CODEBOOKS], dtype=torch.long)
    return arr.view(T, MIMI_NUM_CODEBOOKS).t().unsqueeze(0)  # (1, 8, T)


def build_tokenizer(tokenizer_name: str):
    """Load base tokenizer, capture base_vocab, add the same custom tokens as training."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    base_vocab = len(tok)
    n_add = MIMI_NUM_CODEBOOKS * MIMI_CODEBOOK_SIZE + AUDIO_OFFSET
    new_tokens = [f"<custom_token_{i}>" for i in range(0, n_add + 1)]
    tok.add_tokens(new_tokens)
    return tok, base_vocab


def build_prompt(target_text: str, ref_text: str, ref_codes_lm: list[int],
                 tokenizer, base_vocab: int) -> torch.Tensor:
    """Orpheus prompt + reference audio appended after START_OF_SPEECH."""
    text = (ref_text + " " + target_text).strip() if ref_text else target_text
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    eot = tokenizer.eos_token_id  # Llama: 128009; Qwen3: 151645/151643. Portable.
    prompt = (
        [base_vocab + START_OF_HUMAN]
        + text_ids
        + [eot,
           base_vocab + END_OF_HUMAN,
           base_vocab + START_OF_AI,
           base_vocab + START_OF_SPEECH]
        + ref_codes_lm
    )
    return torch.tensor([prompt], dtype=torch.long)


def synthesize(args):
    cfg = yaml.safe_load(open(args.config))
    ckpt = args.ckpt or f"./{cfg['save_folder']}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] tokenizer={cfg['tokenizer_name']}")
    tokenizer, base_vocab = build_tokenizer(cfg["tokenizer_name"])
    print(f"[load] base_vocab={base_vocab}; total_vocab={len(tokenizer)}")

    print(f"[load] model={ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    ).to(device).eval()
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print(f"[warn] embedding size {model.get_input_embeddings().num_embeddings} "
              f"!= tokenizer len {len(tokenizer)}. Token id alignment may be off.")

    print(f"[load] mimi={MIMI_MODEL_ID}")
    mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(MIMI_MODEL_ID)

    print(f"[encode] ref={args.ref_audio}")
    ref_codes = load_reference_codes(args.ref_audio, mimi, fe, device)
    print(f"[encode] ref_codes shape={tuple(ref_codes.shape)}")
    ref_lm_ids = codes_to_lm_ids(ref_codes, base_vocab)

    input_ids = build_prompt(
        args.text, args.ref_text, ref_lm_ids, tokenizer, base_vocab
    ).to(device)
    print(f"[gen] prompt_len={input_ids.shape[1]} (text+ref_audio)")

    eos_speech_id = base_vocab + END_OF_SPEECH
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos_speech_id,
            pad_token_id=cfg.get("pad_token", tokenizer.eos_token_id),
        )
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    print(f"[gen] generated {len(gen_ids)} tokens")

    gen_codes = lm_ids_to_codes(gen_ids, base_vocab)
    if gen_codes is None or gen_codes.shape[-1] == 0:
        print("[err] no valid Mimi codes in generation", file=sys.stderr)
        sys.exit(1)
    print(f"[decode] gen_codes shape={tuple(gen_codes.shape)} "
          f"({gen_codes.shape[-1]/12.5:.2f}s of audio)")

    with torch.no_grad():
        wav = mimi.decode(gen_codes.to(device)).audio_values  # (1, 1, S)
    wav_np = wav.squeeze().cpu().to(torch.float32).numpy()
    sf.write(args.out, wav_np, samplerate=fe.sampling_rate)
    print(f"[done] wrote {args.out} ({wav_np.shape[0]/fe.sampling_rate:.2f}s)")


def main():
    p = argparse.ArgumentParser(description="OrpheusPlus reference-voice TTS inference")
    p.add_argument("--config", required=True,
                   help="path to qwen3_config.yaml or llama3_config.yaml")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint dir (default: ./{save_folder} from config)")
    p.add_argument("--ref-audio", required=True, help="reference audio wav path")
    p.add_argument("--ref-text", default="", help="transcript of the reference clip")
    p.add_argument("--text", required=True, help="target text to synthesize")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--max-new-tokens", type=int, default=2400)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    args = p.parse_args()
    synthesize(args)


if __name__ == "__main__":
    main()
