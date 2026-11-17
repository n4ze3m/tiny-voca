"""Core ML vocoder and the host steps around the Neural Engine models.

The vocoder is the source-filter head of ``TinyVoca`` (``SourceFilter``) for one
unpadded utterance, without its two linear heads (those run at the end of the
decoder): sub-frame interpolation, lifters, cepstrum -> complex response, and
the short-time Fourier filtering of the pulse train and of the noise, with the
fixed transform matrices of ``export.Exportable``. It stays in float32: in
float16 the audio is only about 52 dB above the error (120 dB in float32), and
the graph does not compile for the Neural Engine anyway. At flexible length
Core ML puts the float32 graph on the GPU (``CPU_AND_GPU``, the default);
``CPU_ONLY`` is as fast for short utterances and 5x slower at 3000 frames.

The host helpers are plain numpy (no torch call at run time) and mirror
``TinyVoca.predict_durations``, ``TinyVoca.expand``, the vowel mask of
``TinyVoca.synthesize`` and ``model.pulse_train``.

Noise: torch's Gaussian generator (Mersenne Twister + Box-Muller in its own
order) cannot be reproduced bit for bit in numpy, so the Core ML path draws its
own seeded Gaussian noise (``noise``). Audio is then not sample-identical to
PyTorch for the same seed, only statistically the same; parity tests pass
torch's noise array in as the ``noise`` input instead.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import os

import numpy as np

from ..config import Config
from ..model import UtteranceTooLong
from ..text import PHONE_IDS, SYMBOL_TO_ID, VOWELS

INPUTS = ("harmonic", "noise_params", "pulses", "noise")
OUTPUTS = ("audio",)
COMPUTE_UNITS = "CPU_AND_GPU"     # measured fastest; nothing of this graph runs on the Neural Engine


def token_tables(vocab_size=Config.vocab_size):
    """Boolean tables [vocab] of phoneme tokens and vowel tokens (the model's
    ``phone_tokens`` and ``vowel_tokens`` buffers)."""
    phones = np.zeros(vocab_size, dtype=bool)
    phones[sorted(PHONE_IDS)] = True
    vowels = np.zeros(vocab_size, dtype=bool)
    vowels[[SYMBOL_TO_ID[v] for v in VOWELS]] = True
    return phones, vowels


PHONE_TOKENS, VOWEL_TOKENS = token_tables()


# Host steps (numpy).

def durations(raw_duration, tokens, speed=1.0, phone_tokens=PHONE_TOKENS):
    """Frames per token [N] (int64) from the raw duration output [2, N] of one
    utterance, as ``TinyVoca.predict_durations``. Float32 like the model, so the
    rounding is the same."""
    raw = np.asarray(raw_duration, dtype=np.float32)
    tokens = np.asarray(tokens, dtype=np.int64)
    frames = np.round(np.expm1(np.clip(raw[0], np.float32(0), np.float32(math.log1p(400)))) / np.float32(speed))
    frames = np.maximum(frames.astype(np.int64), 1)
    present = phone_tokens[tokens] | (raw[1] > 0)
    return np.where(present, frames, 0)


def expand(hidden, durations, max_frames=Config.max_frames):
    """Repeat token states [C, N] by ``durations`` [N] -> (frames [C, T],
    position [3, T] float32, index [T] int64), as ``TinyVoca.expand`` without
    ``frame_position`` and ``frame_voices`` (they are in the decoder graph).
    Position features: relative place in the token, its complement and
    log1p(token length) / 4."""
    durations = np.asarray(durations, dtype=np.int64)
    if durations.ndim != 1 or (durations < 0).any():
        raise ValueError("invalid durations")
    ends = np.cumsum(durations)
    total = int(ends[-1]) if len(ends) else 0
    if total < 1:
        raise ValueError("utterance has no frames")
    if total > max_frames:
        raise UtteranceTooLong("utterance exceeds max_frames; split the input")
    frame = np.arange(total)
    index = np.minimum(np.searchsorted(ends, frame, side="right"), len(durations) - 1)
    owned = np.maximum(durations[index], 1).astype(np.float32)
    offset = (frame - (ends[index] - durations[index])).astype(np.float32)
    relative = (offset + np.float32(0.5)) / owned
    position = np.stack((relative, np.float32(1) - relative, np.log1p(owned) / np.float32(4)))
    return np.asarray(hidden)[:, index], position, index


def vowel_mask(tokens, index, vowel_tokens=VOWEL_TOKENS):
    """Frames [T] owned by a vowel token: their voicing is held at >= 2 in the
    decoder (``voiced_vowels`` in ``TinyVoca.synthesize``)."""
    return vowel_tokens[np.asarray(tokens, dtype=np.int64)][index]


def f0_from_pitch(pitch, pitch_shift=0.0, config=Config()):
    """F0 in Hz [T] float32 from the decoder's log-F0 output (widened, before the shift)."""
    pitch = np.asarray(pitch, dtype=np.float32) + np.float32(pitch_shift)
    f0 = np.float32(config.pitch_reference_hz) * np.exp(pitch)
    return np.clip(f0, np.float32(config.f0_min), np.float32(config.f0_max))


@lru_cache(maxsize=1)
def _pool():
    return ThreadPoolExecutor(min(8, os.cpu_count() or 1), thread_name_prefix="tiny-voca-dsp")


def _each(function, frames, least=32, parts=8):
    """``function(start, stop)`` over up to ``parts`` frame ranges, in threads (numpy
    releases the GIL in its loops). Each range writes its own slice of the output."""
    size = max(least, -(-frames // parts))
    bounds = [(start, min(start + size, frames)) for start in range(0, frames, size)]
    if len(bounds) == 1:
        function(*bounds[0])
    else:
        list(_pool().map(lambda bound: function(*bound), bounds))


def pulse_train(f0, sample_rate=Config.sample_rate, hop=Config.hop, limit=Config.harmonic_limit):
    """``model.pulse_train`` for one utterance (no start phase): f0 [T] in Hz ->
    band-limited pulses [T * hop] float32, computed in float64. The frequency is
    built per frame by broadcasting (sample j of a frame has weight j / hop, the
    same bits as the model's position - floor(position) for a power-of-two hop);
    all but the phase sum runs in threads, with operations in place."""
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    frames = len(f0)
    knots = np.log(np.maximum(f0.astype(np.float64), 1.0))
    knots = np.append(knots, knots[-1])
    weight = np.arange(hop, dtype=np.float64) / hop
    rest = 1 - weight
    frequency = np.empty((frames, hop))

    def frequency_of(start, stop):
        part = frequency[start:stop]
        np.multiply(knots[start:stop, None], rest, out=part)
        part += knots[start + 1:stop + 1, None] * weight
        np.exp(part, out=part)

    _each(frequency_of, frames)
    frequency = frequency.reshape(-1)
    theta = np.cumsum(frequency / sample_rate)
    theta *= 2 * math.pi
    pulses = np.empty(frames * hop, np.float32)
    top = limit * sample_rate

    def wave_of(start, stop):
        span = slice(start * hop, stop * hop)
        angle = np.remainder(theta[span], 2 * math.pi)
        harmonics = top / frequency[span]
        np.maximum(harmonics, 1.0, out=harmonics)
        whole = np.floor(harmonics)
        denominator = np.sin(angle / 2)
        denominator *= 2
        tiny = np.abs(denominator) < 1e-7
        wave = np.sin((whole + 0.5) * angle)
        wave /= np.where(tiny, 1.0, denominator)
        wave -= 0.5
        wave[tiny] = whole[tiny]
        top_harmonic = np.cos((whole + 1) * angle)
        top_harmonic *= harmonics - whole
        wave += top_harmonic
        np.divide(2.0, harmonics, out=harmonics)
        wave *= np.sqrt(harmonics, out=harmonics)
        pulses[span] = wave

    _each(wave_of, frames)
    return pulses


def noise(samples, seed=0):
    """Seeded Gaussian noise [samples] float32 for the noise branch. Not the
    same numbers as torch.randn for that seed (see the module docstring)."""
    return np.random.default_rng(seed).standard_normal(samples, dtype=np.float32)


def vocoder_inputs(harmonic, noise_params, f0, noise_samples=None, seed=0, config=Config()):
    """Inputs of the Core ML vocoder for one utterance: decoder outputs [98, T]
    and [36, T], f0 [T] in Hz, and optional noise [T * hop] (torch's, for a
    parity test); without it the noise is ``noise(T * hop, seed)``."""
    frames = np.asarray(harmonic).shape[-1]
    samples = frames * config.hop
    if noise_samples is None:
        noise_samples = noise(samples, seed)
    noise_samples = np.asarray(noise_samples, dtype=np.float32).reshape(-1)
    if len(noise_samples) != samples:
        raise ValueError("noise needs frames * hop samples")
    pulses = pulse_train(f0, config.sample_rate, config.hop, config.harmonic_limit)
    as_input = lambda value: np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(1, -1, frames))
    return {"harmonic": as_input(harmonic), "noise_params": as_input(noise_params),
            "pulses": pulses[None], "noise": noise_samples[None]}


# Vocoder graph (torch, for conversion).

def vocoder_module(model):
    """The vocoder graph of a loaded ``TinyVoca`` as a float32 torch module. Torch
    is imported here, so that the host steps above need none of it."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    from ..export import _analysis_kernels, _cepstrum_matrices, _synthesis_kernel

    class Vocoder(nn.Module):
        """harmonic [1, 98, T], noise_params [1, 36, T], pulses and noise
        [1, T * hop] -> audio [1, T * hop]. Channel-first: the decoder emits
        [1, C, T], so no transpose is needed on either side."""

        def __init__(self, model):
            super().__init__()
            c = self.config = model.config
            v = model.vocoder
            self.split = c.cep_causal + c.cep_anti
            causal = list(range(c.cep_causal))
            anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
            for name, (real, imaginary) in (("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
                                            ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft))):
                # [bins, coefficients]: response [1, bins, S] = matrix @ cepstrum [1, coefficients, S].
                self.register_buffer(f"{name}_real", real.T.contiguous(), persistent=False)
                self.register_buffer(f"{name}_imaginary", imaginary.T.contiguous(), persistent=False)
            self.register_buffer("harmonic_lifter", torch.cat((v.causal_lifter, v.anti_lifter))[None, :, None].clone(), persistent=False)
            self.register_buffer("noise_lifter", v.noise_lifter[None, :, None].clone(), persistent=False)
            for name, window in (("synth", v.synth_window), ("noise", v.noise_window)):
                real, imaginary = _analysis_kernels(window)
                self.register_buffer(f"{name}_analysis", torch.cat((real, imaginary))[:, 0].contiguous(), persistent=False)   # [2 bins, fft]
                self.register_buffer(f"{name}_synthesis", _synthesis_kernel(window)[:, 0].contiguous(), persistent=False)    # [2 bins, fft]
                self.register_buffer(f"{name}_scale", self._inverse_envelope(window, len(window) // 4), persistent=False)
            self.harmonic_steps, self.noise_steps = v.harmonic_steps, v.noise_steps

        @staticmethod
        def _inverse_envelope(window, hop):
            """1 / overlap-added squared window over the first ``hop`` output samples
            (the only ones missing a frame) and, as the last value, in the interior."""
            fft = len(window)
            square = window.double() ** 2
            envelope = torch.zeros(3 * fft, dtype=torch.float64)
            for start in range(0, 2 * fft, hop):
                envelope[start:start + fft] += square
            kept = envelope[fft // 2:fft // 2 + hop + 1]
            return (1 / kept).float()

        def _filter(self, source, real, imaginary, name, fft, hop):
            """STFT of ``source`` [1, samples] times the response [1, bins, frames],
            resynthesized by overlap-add: ``Exportable._filter`` with the strided
            convolutions written as framing (reshape) + matrix products. At flexible T
            Core ML runs the convolution form on the CPU only, and slowly (0.5 s instead
            of 1.3 ms at 200 frames); this form runs on the GPU or fast on the CPU.
            ``fft`` is 4 * ``hop`` (Config checks it)."""
            bins = fft // 2 + 1
            blocks = F.pad(source, (fft // 2, fft)).reshape(1, -1, hop)
            frames = torch.cat((blocks[:, :-3], blocks[:, 1:-2], blocks[:, 2:-1], blocks[:, 3:]), dim=2)   # [1, frames, fft]
            spectrum = torch.matmul(getattr(self, f"{name}_analysis"), frames.transpose(1, 2))           # [1, 2 bins, frames]
            s_real, s_imaginary = spectrum[:, :bins], spectrum[:, bins:]
            product = torch.cat((s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real), dim=1)
            pieces = torch.matmul(product.transpose(1, 2), getattr(self, f"{name}_synthesis"))           # [1, frames, fft]
            # Overlap-add: output block b is the sum of quarter k of frame b - k.
            wave = sum(F.pad(pieces[:, :, k * hop:(k + 1) * hop], (0, 0, k, 3 - k)) for k in range(4)).flatten(1)[:, fft // 2:-fft]
            scale = getattr(self, f"{name}_scale")
            # Only the first ``hop`` samples lack an overlapping frame; elsewhere the envelope is constant.
            return torch.cat((wave[:, :hop] * scale[:hop], wave[:, hop:] * scale[hop]), dim=1)

        @staticmethod
        def _response(cepstrum, real, imaginary):
            magnitude = torch.exp(torch.matmul(real, cepstrum).clamp(-14, 6))
            angle = torch.matmul(imaginary, cepstrum)
            return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

        @staticmethod
        def _tail(x, count=3):
            """Append ``count`` copies of the last column: sub-frames past the end
            repeat the last one (``SourceFilter._subframes``)."""
            last = x[..., -1:]
            return torch.cat((x,) + (last,) * count, dim=-1)

        def forward(self, harmonic, noise_params, pulses, noise):
            c = self.config
            # Harmonic branch: two sub-frames per frame, the second halfway to the next frame.
            h = harmonic[:, :self.split]
            following = torch.cat((h, h[..., -1:]), dim=-1)[..., 1:]   # never empty: Core ML has no zero-size tensors at T = 1
            envelope = torch.stack((h, 0.5 * (h + following)), dim=-1).flatten(2)
            envelope = self._tail(envelope) * self.harmonic_lifter
            gain = self._tail(harmonic[:, self.split:].transpose(1, 2).flatten(1))
            envelope = torch.cat((envelope[:, :1] + gain[:, None], envelope[:, 1:]), dim=1)
            harmonic_real, harmonic_imaginary = self._response(envelope, self.harmonic_real, self.harmonic_imaginary)
            # Noise branch: one envelope per frame, one gain per sub-frame.
            n = noise_params[:, :c.noise_cep]
            shape = torch.cat((n[..., None],) * self.noise_steps, dim=-1).flatten(2)
            shape = self._tail(shape) * self.noise_lifter
            gain = self._tail(noise_params[:, c.noise_cep:].transpose(1, 2).flatten(1))
            shape = torch.cat((shape[:, :1] + gain[:, None], shape[:, 1:]), dim=1)
            noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)
            return (self._filter(pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop)
                    + self._filter(noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop))

    return Vocoder(model).eval()


def convert(model, path=None, compute_units=COMPUTE_UNITS, max_frames=None, example_frames=64):
    """Core ML vocoder (mlprogram, float32, frames 1..max_frames). Saved to
    ``path`` when given; returns the ``MLModel``."""
    import coremltools as ct
    import torch
    c = model.config
    max_frames = max_frames or c.max_frames
    module = vocoder_module(model)
    t = example_frames
    example = (torch.zeros(1, c.cep_causal + c.cep_anti + module.harmonic_steps, t), torch.zeros(1, c.noise_cep + module.noise_steps, t),
               torch.zeros(1, t * c.hop), torch.zeros(1, t * c.hop))
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    frames = ct.RangeDim(1, max_frames, default=t)
    samples = ct.RangeDim(c.hop, max_frames * c.hop, default=t * c.hop)
    inputs = [ct.TensorType(name="harmonic", shape=(1, example[0].shape[1], frames), dtype=np.float32),
              ct.TensorType(name="noise_params", shape=(1, example[1].shape[1], frames), dtype=np.float32),
              ct.TensorType(name="pulses", shape=(1, samples), dtype=np.float32),
              ct.TensorType(name="noise", shape=(1, samples), dtype=np.float32)]
    mlmodel = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="audio", dtype=np.float32)],
                         convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                         minimum_deployment_target=ct.target.macOS15, compute_units=getattr(ct.ComputeUnit, compute_units))
    mlmodel.short_description = "tiny-voca source-filter vocoder: decoder filter parameters, pulse train and noise -> 24 kHz audio"
    if path is not None:
        mlmodel.save(str(path))
    return mlmodel
