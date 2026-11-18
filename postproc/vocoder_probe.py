"""Probe the source-filter vocoder internals on a fixed utterance.

We mirror SourceFilter.forward (inference-only, no training code) but keep the
harmonic and noise branches separate and dump per-frame parameters. This tells
us whether audible artifacts come from the cepstral envelope, the pulse train,
or noise leaking into voiced frames.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DIZIONE_DATA", str(Path(__file__).resolve().parent.parent / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import numpy as np
import torch
from torch.nn import functional as F

from tiny_voca.infer import load_model
from tiny_voca.text import phonemize
from tiny_voca.config import Config
from tiny_voca.model import cepstrum_response, pulse_train

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ROOT = Path(__file__).resolve().parent.parent


@torch.inference_mode()
def split_forward(model, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0):
    """Run tiny-voca up to the vocoder, then the vocoder branch by branch."""
    c = model.config
    hidden, mask = model.encode(tokens, stress, lengths, speakers)
    durations = model.predict_durations(hidden, mask, tokens, speed)
    frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, speakers)
    prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
    if True:  # synthesize() keeps vowels voiced; do the same
        ends = durations.cumsum(1)
        frame = torch.arange(frames.shape[1])[None].expand(len(lengths), -1).contiguous()
        owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
        vowel = model.vowel_tokens[tokens].gather(1, owner) & frame_mask
        prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                             prosody[..., 2:]), dim=-1)
    acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
    f0 = model.frequency(prosody[..., 0] + pitch_shift)

    v = model.vocoder
    samples = frames.shape[1] * c.hop
    audible = (torch.arange(samples)[None] < (frame_lengths * c.hop)[:, None])
    last = (frame_lengths - 1).clamp_min(0)[:, None]

    h = v.harmonic(acoustic)
    split = c.cep_causal + c.cep_anti
    sub = v._subframes(frames.shape[1], frame_lengths, v.harmonic_steps)
    here, following = sub // v.harmonic_steps, torch.minimum((sub + 1) // v.harmonic_steps, last)
    causal_raw = 0.5 * (v._take(h[..., :c.cep_causal], here) + v._take(h[..., :c.cep_causal], following))
    anti_raw = 0.5 * (v._take(h[..., c.cep_causal:split], here) + v._take(h[..., c.cep_causal:split], following))
    causal = causal_raw * v.causal_lifter
    anti = anti_raw * v.anti_lifter
    gains = h[..., split:].flatten(1).gather(1, sub)
    causal = torch.cat((causal[..., :1] + gains[..., None], causal[..., 1:]), dim=-1)
    harmonic_response = cepstrum_response(causal, anti, c.synth_fft)

    n = v.noise(acoustic)
    sub_n = v._subframes(frames.shape[1], frame_lengths, v.noise_steps)
    shape = v._take(n[..., :c.noise_cep], sub_n // v.noise_steps) * v.noise_lifter
    ngains = n[..., c.noise_cep:].flatten(1).gather(1, sub_n)
    shape = torch.cat((shape[..., :1] + ngains[..., None], shape[..., 1:]), dim=-1)
    noise_response = cepstrum_response(shape, None, c.noise_fft)

    f0h = f0.float().gather(1, torch.minimum(torch.arange(frames.shape[1])[None], last))
    pulses = model.vocoder.__class__.__mro__[0]  # unused
    pulses = pulse_train(f0h, c.sample_rate, c.hop, c.harmonic_limit) * audible
    noise = torch.randn(hidden.shape[0], samples, generator=torch.Generator().manual_seed(seed))

    harm = v._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, v.synth_window)
    nois = v._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, v.noise_window)
    return {
        "wave": (harm + nois) * audible,
        "harmonic": harm * audible,
        "noise": nois * audible,
        "pulses": pulses,
        "f0": f0h, "durations": durations, "prosody": prosody,
        "causal": causal_raw, "anti": anti_raw, "ngains": ngains,
        "frame_lengths": frame_lengths, "acoustic": acoustic,
    }


def main():
    model = load_model(ROOT / "weights" / "tiny-voca.pt", "cpu")
    text = "The quick brown fox jumps over the lazy dog while the band plays softly."
    for voice in ("F2", "M2"):
        phones = phonemize(text)
        tokens = torch.tensor([phones.ids])
        stress = torch.tensor([phones.stress])
        speaker = torch.tensor([model.config.speakers.index(voice)])
        out = split_forward(model, tokens, stress, torch.tensor([len(phones)]), speaker)
        n = int(out["frame_lengths"][0]) * model.config.hop
        wave = out["wave"][0, :n].numpy()
        harm = out["harmonic"][0, :n].numpy()
        nois = out["noise"][0, :n].numpy()
        d = C.DATA / "branches"
        d.mkdir(parents=True, exist_ok=True)
        for name, sig in (("wave", wave), ("harmonic", harm), ("noise", nois)):
            C.save(d / f"{voice}_{name}.wav", sig)
        # per-frame branch energy
        hop = model.config.hop
        frames = n // hop
        eh = (harm[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1)
        en = (nois[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1)
        snr = C.db(eh) - C.db(en)
        f0 = out["f0"][0, :frames].numpy()
        print(f"\n{voice}: {n/C.SR:.2f}s, {frames} frames")
        print(f"  harmonic RMS {C.db(np.mean(eh)):.1f} dB, noise RMS {C.db(np.mean(en)):.1f} dB")
        print(f"  per-frame noise/harmonic ratio (dB): mean {np.mean(-snr):.1f}, "
              f"p10 {np.percentile(-snr,10):.1f}, p90 {np.percentile(-snr,90):.1f}")
        print(f"  noise louder than harmonic in {np.mean(snr < 0)*100:.1f}% of frames")
        print(f"  f0 {np.percentile(f0,[5,50,95]).round(1)} Hz")

        # frame-boundary discontinuity: second difference at multiples of 256
        # vs random offsets, normalised by local signal power
        dd = np.abs(np.diff(wave, 2))
        local = np.sqrt(np.convolve(wave ** 2, np.ones(64) / 64, mode="same"))[:len(dd)]
        ratio = dd / (local + 1e-6)
        idx = np.arange(hop, len(dd) - 1, hop)
        rnd = np.random.default_rng(0).integers(1, len(dd) - 1, len(idx))
        print(f"  boundary |d2| {np.mean(ratio[idx]):.4f} vs random {np.mean(ratio[rnd]):.4f} "
              f"(ratio {np.mean(ratio[idx])/np.mean(ratio[rnd]):.2f})")

        # harmonic vs noise spectral character
        for name, sig in (("harm", harm), ("noise", nois)):
            mag = C.stft_mag(sig, 1024, 256)
            m = (mag ** 2).sum(axis=1) > 1e-8
            st = C.spectral_stats(mag[m], 1024)
            print(f"  {name}: flatness {np.mean(st['flatness']):.3f}, centroid {np.mean(st['centroid']):.0f} Hz, "
                  f"rolloff {np.mean(st['rolloff']):.0f} Hz")

        # cepstral envelope statistics
        for name, arr in (("causal0", out["causal"][0, :, 0]), ("anti_absmax", out["anti"][0].abs().amax(1)),
                          ("noise_gain0", out["ngains"][0])):
            a = arr.float().numpy()
            print(f"  {name}: mean {a.mean():+.3f} std {a.std():.3f} "
                  f"p1 {np.percentile(a,1):+.3f} p99 {np.percentile(a,99):+.3f}")


if __name__ == "__main__":
    main()
