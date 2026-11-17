"""Encoder and decoder of tiny-voca rewritten for the Apple Neural Engine, and their Core ML conversion.

The ANE works on 4-D tensors [B, C, 1, S] in float16 and is fastest with 1x1
convolutions, elementwise ops and reductions over the channel axis. The modules
here compute the same values as ``TinyVoca`` in that layout:

- every Linear is a 1x1 Conv2d, the depthwise and predictor convolutions are (1, k) Conv2d;
- LayerNorm normalizes the channel axis;
- rotary positions: the q/k channels of each head are permuted offline from
  interleaved pairs to (first half, second half), and the rotated copy
  ([-b, a]) comes out of the same qkv convolution, so the rotation is
  ``q * cos + q_rotated * sin`` with constant tables and no slicing. q . k does
  not change when q and k get the same channel permutation; v is untouched.
  The 1/sqrt(head width) scale is folded into the q weights;
- attention is computed per head on [B, C, 1, S] slices with the keys on the
  channel axis, and padding is an additive -1e4 bias on the padded keys;
- embeddings are one-hot rows times the table (``relu(1 - |id - k|)``, exact for
  integer ids): the compute plan puts a gather (and its neighbours) on the CPU;
- attention queries are processed in chunks of ``QUERY_CHUNK`` (see there).

The ANE cannot read int32, so the casts of the int32 id inputs run on the CPU and the
planner keeps the embeddings there with them (about 5% of the encoder's estimated cost,
no measurable latency). ``id_dtype=np.float16`` makes every op run on the NE.
Load the models from a kept ``.mlmodelc`` (``ct.utils.compile_model`` once): then the
ANE compilation is cached across processes (0.13 s per function instead of 3-11 s).

Two models with static shapes, each a multifunction ``.mlpackage`` with one
function per length bucket (weights are stored once):

Encoder, function ``n{N}``
    in   tokens int32 [1,N], stress int32 [1,N], voice int32 [1], token_mask fp16 [1,N]
    out  hidden fp16 [1,192,1,N] (masked encoder output), duration fp16 [1,2,N] (raw predictor output)
Decoder, function ``t{T}``
    in   frames fp16 [1,192,1,T] (encoder state of each frame's token), position fp16 [1,3,1,T]
         (rel, 1-rel, log1p(owned)/4), voice int32 [1], frame_mask fp16 [1,T], vowel_mask fp16 [1,T], pitch_range fp16 [1]
    out  pitch fp16 [1,T] (widened log F0 / 200 Hz, before pitch_shift and clamp),
         harmonic fp16 [1,98,T], noise_params fp16 [1,36,T] (raw vocoder Linear outputs)

Inputs are padded at the end and zero on padding; outputs past the real length are meaningless.
"""

import math
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..model import rotary_table

ENCODER_BUCKETS = (32, 64, 128, 256, 512)
# Measured on the ANE (M4 Max, ms): 128 0.7, 256 0.7, 512 1.6, 768 2.4, 1024 4.0, 1536 7.2, 2048 11.2, 3072 24.3.
# The 768 and 1536 steps cut the padding cost where latency starts to grow fast; 3072 holds max_frames 3000.
DECODER_BUCKETS = (128, 256, 512, 768, 1024, 1536, 2048, 3072)
MASK_BIAS = -1e4            # finite in fp16; exp underflows to exactly 0 after the softmax max shift
# Attention queries are processed in chunks of this many positions. The scores of one head are
# [1, S, 1, chunk] instead of [1, S, 1, S]; on the ANE (M4 Max) one attention layer at S=2048 takes
# 1.8 ms with chunks of 256 against 3.7 ms unchunked (3.8 against 8.2 ms at S=3072). Results are the same.
QUERY_CHUNK = 256


def _conv(linear):
    """nn.Linear -> 1x1 Conv2d with the same weights."""
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1)
    conv.weight.data = linear.weight.data.clone()[:, :, None, None]
    conv.bias.data = linear.bias.data.clone()
    return conv


def _conv_kernel(conv1d):
    """Conv1d -> Conv2d with a (1, k) kernel."""
    k = conv1d.kernel_size[0]
    conv = nn.Conv2d(conv1d.in_channels, conv1d.out_channels, (1, k), padding=(0, k // 2), groups=conv1d.groups)
    conv.weight.data = conv1d.weight.data.clone()[:, :, None, :]
    conv.bias.data = conv1d.bias.data.clone()
    return conv


class ChannelNorm(nn.Module):
    """LayerNorm over the channel axis of [B, C, 1, S]."""

    def __init__(self, norm):
        super().__init__()
        self.eps = norm.eps
        self.weight = nn.Parameter(norm.weight.data.clone()[None, :, None, None])
        self.bias = nn.Parameter(norm.bias.data.clone()[None, :, None, None])

    def forward(self, x):
        centred = x - x.mean(1, keepdim=True)
        return centred * torch.rsqrt((centred * centred).mean(1, keepdim=True) + self.eps) * self.weight + self.bias


class OneHot(nn.Module):
    """Embedding as one-hot rows times the table: a 1x1 convolution on the ANE."""

    def __init__(self, embedding):
        super().__init__()
        rows, width = embedding.weight.shape
        self.register_buffer("ids", torch.arange(rows, dtype=torch.float32)[None, :, None, None], persistent=False)
        self.table = nn.Conv2d(rows, width, 1, bias=False)
        self.table.weight.data = embedding.weight.data.T.clone()[:, :, None, None]

    def forward(self, ids):
        """ids [1, 1, 1, S] float (exact integers) -> [1, width, 1, S]."""
        return self.table(F.relu(1 - (ids - self.ids).abs()))


class AttentionANE(nn.Module):
    def __init__(self, attention, channels, heads, query_chunk=None):
        super().__init__()
        self.channels, self.heads, self.query_chunk = channels, heads, query_chunk
        width = channels // heads
        half = width // 2
        weight, bias = attention.qkv.weight.data, attention.qkv.bias.data
        # Per head: interleaved pairs (0::2, 1::2) -> first half, second half.
        order = torch.cat([h * width + torch.cat((torch.arange(0, width, 2), torch.arange(1, width, 2))) for h in range(heads)])
        # Rotated copy [-b, a] of the permuted layout, per head.
        rotated = torch.cat([h * width + torch.cat((torch.arange(half, width), torch.arange(0, half))) for h in range(heads)])
        sign = torch.cat([torch.cat((-torch.ones(half), torch.ones(half))) for _ in range(heads)])
        scale = width ** -0.5
        parts = []
        for start, factor in ((0, scale), (channels, 1.0)):
            w, b = weight[start:start + channels][order] * factor, bias[start:start + channels][order] * factor
            parts += [(w, b), (w[rotated] * sign[:, None], b[rotated] * sign)]
        parts.append((weight[2 * channels:], bias[2 * channels:]))
        self.qkv = nn.Conv2d(channels, 5 * channels, 1)
        self.qkv.weight.data = torch.cat([w for w, _ in parts])[:, :, None, None].clone()
        self.qkv.bias.data = torch.cat([b for _, b in parts]).clone()
        self.out = _conv(attention.out)

    def _attend(self, q, k, v, bias):
        """q [1,w,1,Sq], k, v [1,w,1,S], bias [1,S,1,1] -> [1,w,1,Sq]. Keys on the channel axis."""
        weights = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3)) + bias
        return torch.einsum("bkhq,bchk->bchq", weights.softmax(dim=1), v)

    def forward(self, x, bias, rope):
        cos, sin, size = rope
        c = self.channels
        q, q_rotated, k, k_rotated, v = self.qkv(x).split(c, dim=1)
        q = q * cos + q_rotated * sin
        k = k * cos + k_rotated * sin
        width = c // self.heads
        chunk = self.query_chunk if self.query_chunk and self.query_chunk < size else size
        heads = []
        for h in range(self.heads):
            span = slice(h * width, (h + 1) * width)
            qh, kh, vh = q[:, span], k[:, span], v[:, span]
            pieces = [self._attend(qh[..., i:i + chunk], kh, vh, bias) for i in range(0, size, chunk)]
            heads.append(pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1))
        return self.out(torch.cat(heads, dim=1))


class BlockANE(nn.Module):
    def __init__(self, block, c, query_chunk=None):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(block.norm1), ChannelNorm(block.norm2)
        self.attention = AttentionANE(block.attention, c.channels, c.heads, query_chunk)
        self.depthwise = _conv_kernel(block.depthwise)
        self.expand, self.project = _conv(block.expand), _conv(block.project)

    def forward(self, x, keep, bias, rope):
        x = (x + self.attention(self.norm1(x), bias, rope)) * keep
        y = self.depthwise(self.norm2(x) * keep)
        return (x + self.project(F.gelu(self.expand(y)))) * keep


class StackANE(nn.Module):
    def __init__(self, stack, c, query_chunk=None):
        super().__init__()
        self.blocks = nn.ModuleList(BlockANE(b, c, query_chunk) for b in stack.blocks)
        self.norm = ChannelNorm(stack.norm)

    def forward(self, x, keep, bias, rope):
        for block in self.blocks:
            x = block(x, keep, bias, rope)
        return self.norm(x) * keep


class PredictorANE(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.conv1, self.conv2 = _conv_kernel(predictor.conv1), _conv_kernel(predictor.conv2)
        self.norm1, self.norm2 = ChannelNorm(predictor.norm1), ChannelNorm(predictor.norm2)
        self.out = _conv(predictor.out)

    def forward(self, x, keep):
        x = self.norm1(F.gelu(self.conv1(x * keep))) * keep
        x = self.norm2(F.gelu(self.conv2(x)))
        return self.out(x) * keep


class _Rotary(nn.Module):
    """(cos, sin, S): tables [1, C, 1, S] for the permuted q/k layout, sliced to the input length S (a Python int).
    They come from ``model.rotary_table`` so the angles round exactly as in the original."""

    def __init__(self, c, size):
        super().__init__()
        width = c.channels // c.heads
        cos, sin = rotary_table(size, width, "cpu")                     # [S, width/2]
        full = lambda t: torch.cat((t, t), dim=1).T.repeat(c.heads, 1)[None, :, None, :].contiguous()
        self.register_buffer("cos", full(cos), persistent=False)
        self.register_buffer("sin", full(sin), persistent=False)

    def forward(self, size):
        return self.cos[..., :size], self.sin[..., :size], size


def _masks(mask, size):
    """mask [1, S] (1 real, 0 pad) -> keep [1,1,1,S] and key bias [1,S,1,1]."""
    return mask.reshape(1, 1, 1, size), (mask.reshape(1, size, 1, 1) - 1) * -MASK_BIAS


class Encoder(nn.Module):
    """Embeddings + text encoder + duration predictor, for one padded utterance."""

    def __init__(self, model, query_chunk=None):
        super().__init__()
        c = model.config
        # Names differ from the inputs (tokens, stress, voice): the converter confuses equal names.
        self.token_table, self.stress_table, self.voice_table = OneHot(model.tokens), OneHot(model.stress), OneHot(model.voices)
        self.encoder = StackANE(model.encoder, c, query_chunk)
        self.duration = PredictorANE(model.duration)
        self.rotary = _Rotary(c, max(c.max_tokens, max(ENCODER_BUCKETS)))
        self.size = None            # set to the bucket while tracing, so shapes become constants

    def forward(self, tokens, stress, voice, token_mask):
        size = self.size or tokens.shape[-1]
        # The int32 -> fp16 casts can only run on the CPU; doing them first keeps the CPU part one short segment.
        tokens, stress, voice = tokens.float(), stress.float(), voice.float()
        keep, bias = _masks(token_mask, size)
        x = self.token_table(tokens.reshape(1, 1, 1, size)) + self.stress_table(stress.reshape(1, 1, 1, size))
        x = (x + self.voice_table(voice.reshape(1, 1, 1, 1))) * keep
        hidden = self.encoder(x, keep, bias, self.rotary(size))
        return hidden, self.duration(hidden, keep).reshape(1, -1, size)


class Decoder(nn.Module):
    """Everything between the frame expansion and the waveform synthesis, for one padded utterance."""

    def __init__(self, model, query_chunk=None):
        super().__init__()
        c = model.config
        self.frame_position, self.frame_voices = _conv(model.frame_position), OneHot(model.frame_voices)
        self.prosody_stack = StackANE(model.prosody_stack, c, query_chunk)
        self.prosody_head = _conv(model.prosody_head)
        self.prosody_projection = _conv(model.prosody_projection)
        self.acoustic_stack = StackANE(model.acoustic_stack, c, query_chunk)
        self.harmonic, self.noise = _conv(model.vocoder.harmonic), _conv(model.vocoder.noise)
        self.rotary = _Rotary(c, max(c.max_frames, max(DECODER_BUCKETS)))
        self.size = None            # set to the bucket while tracing, so shapes become constants

    def forward(self, frames, position, voice, frame_mask, vowel_mask, pitch_range):
        size = self.size or frames.shape[-1]
        keep, bias = _masks(frame_mask, size)
        rope = self.rotary(size)
        x = (frames + self.frame_position(position) + self.frame_voices(voice.float().reshape(1, 1, 1, 1))) * keep
        hidden = self.prosody_stack(x, keep, bias, rope)
        prosody = self.prosody_head(hidden) * keep
        pitch, voicing, energy = prosody[:, 0:1], prosody[:, 1:2], prosody[:, 2:3]
        # A vowel is voiced: voicing = max(voicing, 2) on vowel frames, as relu so it stays elementwise.
        voicing = voicing + vowel_mask.reshape(1, 1, 1, size) * F.relu(2.0 - voicing)
        voiced = (voicing > 0).to(prosody.dtype)          # zero on padding: prosody and vowel_mask are zero there
        mean = (pitch * voiced).sum(-1, keepdim=True) / voiced.sum(-1, keepdim=True).clamp(min=1.0)
        pitch = (mean + (pitch - mean) * pitch_range.reshape(1, 1, 1, 1)) * keep
        condition = torch.cat((pitch.clamp(-2.0, 1.5), voiced, energy.clamp(-4.5, 3.0)), dim=1)
        acoustic = self.acoustic_stack((hidden + self.prosody_projection(condition)) * keep, keep, bias, rope)
        return pitch.reshape(1, size), self.harmonic(acoustic).reshape(1, -1, size), self.noise(acoustic).reshape(1, -1, size)


# ---------------------------------------------------------------- reference and host-side helpers (private)

def _durations(raw, tokens, model, speed=1.0):
    """``TinyVoca.predict_durations`` for one utterance: raw [2, n] numpy, tokens [n] -> int64 [n]."""
    raw = np.asarray(raw, dtype=np.float32)
    frames = np.maximum(np.round(np.expm1(np.clip(raw[0], 0, math.log1p(400))) / speed), 1).astype(np.int64)
    present = model.phone_tokens.numpy()[np.asarray(tokens)] | (raw[1] > 0)
    return np.where(present, frames, 0)


def _expand(hidden, durations, tokens, model, size=None):
    """Host frame expansion mirroring ``TinyVoca.expand``: hidden [C, n], durations [n] ->
    frames [C, size], position [3, size], frame_mask [size], vowel_mask [size] (float32, zero past the end)."""
    durations = np.asarray(durations, dtype=np.int64)
    total = int(durations.sum())
    size = total if size is None else size
    ends = np.cumsum(durations)
    index = np.minimum(np.searchsorted(ends, np.arange(total), side="right"), len(durations) - 1)
    owned = np.maximum(durations[index], 1).astype(np.float32)
    relative = (np.arange(total) - (ends[index] - durations[index])).astype(np.float32)
    relative = (relative + np.float32(0.5)) / owned
    frames, position = np.zeros((hidden.shape[0], size), np.float32), np.zeros((3, size), np.float32)
    frames[:, :total] = hidden[:, index]
    position[:, :total] = np.stack((relative, 1 - relative, np.log1p(owned) / 4))
    mask, vowel = np.zeros(size, np.float32), np.zeros(size, np.float32)
    mask[:total] = 1
    vowel[:total] = model.vowel_tokens.numpy()[np.asarray(tokens)][index]
    return frames, position, mask, vowel


@torch.no_grad()
def _reference(model, tokens, stress, voice, durations=None, speed=1.0, pitch_range=1.0):
    """Original model, one utterance: encoder hidden [C,n], raw duration [2,n], durations,
    pitch [T], harmonic [98,T], noise [36,T] (float32 numpy)."""
    tokens_t, stress_t = torch.as_tensor(tokens)[None], torch.as_tensor(stress)[None]
    n = tokens_t.shape[1]
    speakers = torch.tensor([voice])
    hidden, mask = model.encode(tokens_t, stress_t, torch.tensor([n]), speakers)
    raw = model.duration(hidden, mask)
    if durations is None:
        durations = model.predict_durations(hidden, mask, tokens_t, speed)
    else:
        durations = torch.as_tensor(durations)[None]
    frames, frame_mask, _ = model.expand(hidden, durations, mask, speakers)
    prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
    ends = durations.cumsum(1)
    owner = torch.searchsorted(ends, torch.arange(frames.shape[1])[None].contiguous(), right=True).clamp(max=n - 1)
    vowel = model.vowel_tokens[tokens_t].gather(1, owner) & frame_mask
    prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                         prosody[..., 2:]), dim=-1)
    prosody = model.widen_pitch(prosody, frame_mask, pitch_range)
    acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
    return {"hidden": hidden[0].T.numpy(), "duration": raw[0].T.numpy(), "durations": durations[0].numpy(),
            "pitch": prosody[0, :, 0].numpy(), "harmonic": model.vocoder.harmonic(acoustic)[0].T.numpy(),
            "noise_params": model.vocoder.noise(acoustic)[0].T.numpy()}


def bucket(length, buckets):
    """The smallest bucket that holds ``length``."""
    for size in sorted(buckets):
        if length <= size:
            return size
    raise ValueError(f"length {length} exceeds the largest bucket {max(buckets)}")


def encoder_inputs(tokens, stress, voice, size, dtype=np.float32, id_dtype=np.int32):
    """Encoder inputs padded to ``size``. Core ML wants dtype=np.float16 (and id_dtype as converted)."""
    n = len(tokens)
    pad = lambda v: np.pad(np.asarray(v, dtype=id_dtype), (0, size - n))[None]
    mask = np.zeros((1, size), dtype)
    mask[0, :n] = 1
    return {"tokens": pad(tokens), "stress": pad(stress), "voice": np.array([voice], id_dtype), "token_mask": mask}


def decoder_inputs(hidden, durations, tokens, voice, model, size=None, pitch_range=1.0, dtype=np.float32, id_dtype=np.int32):
    """hidden [C, n] (real tokens only) -> Decoder inputs padded to ``size`` (default: the right bucket)."""
    size = bucket(int(np.sum(durations)), DECODER_BUCKETS) if size is None else size
    frames, position, mask, vowel = _expand(np.asarray(hidden, np.float32), durations, tokens, model, size)
    return {"frames": frames[None, :, None].astype(dtype), "position": position[None, :, None].astype(dtype),
            "voice": np.array([voice], id_dtype), "frame_mask": mask[None].astype(dtype),
            "vowel_mask": vowel[None].astype(dtype), "pitch_range": np.array([pitch_range], dtype)}


def _torch_inputs(inputs):
    return tuple(torch.from_numpy(np.asarray(v)).float() if np.asarray(v).dtype != np.int32 else torch.from_numpy(np.asarray(v))
                 for v in inputs.values())


# ---------------------------------------------------------------- Core ML conversion

ENCODER_INPUTS = ("tokens", "stress", "voice", "token_mask")
ENCODER_OUTPUTS = ("hidden", "duration")
DECODER_INPUTS = ("frames", "position", "voice", "frame_mask", "vowel_mask", "pitch_range")
DECODER_OUTPUTS = ("pitch", "harmonic", "noise_params")


def _example(kind, model, size):
    c = model.config
    if kind == "encoder":
        tokens = (np.arange(size) * 7) % (c.vocab_size - 5) + 5
        return encoder_inputs(tokens, np.arange(size) % 3, 0, size)
    hidden = np.random.default_rng(0).standard_normal((c.channels, 8)).astype(np.float32)
    durations = np.full(8, size // 8)
    durations[-1] += size - durations.sum()
    return decoder_inputs(hidden, durations, np.full(8, 20), 0, model, size)


ID_INPUTS = ("tokens", "stress", "voice")


def _convert_one(module, example, outputs, size, id_dtype):
    import coremltools as ct
    module.size = size
    if id_dtype != np.int32:
        example = {k: v.astype(np.float32) if k in ID_INPUTS else v for k, v in example.items()}
    with torch.no_grad():
        traced = torch.jit.trace(module, _torch_inputs(example))
    inputs = [ct.TensorType(name=name, shape=value.shape, dtype=id_dtype if name in ID_INPUTS else np.float16)
              for name, value in example.items()]
    return ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=name, dtype=np.float16) for name in outputs],
                      convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)


def _convert(kind, model, path, buckets, prefix, outputs, query_chunk, id_dtype):
    import coremltools as ct
    module = (Encoder if kind == "encoder" else Decoder)(model.float().eval(), query_chunk).eval()
    path = Path(path)
    with tempfile.TemporaryDirectory() as directory:
        descriptor = ct.utils.MultiFunctionDescriptor()
        for size in sorted(buckets):
            part = Path(directory) / f"{prefix}{size}.mlpackage"
            _convert_one(module, _example(kind, model, size), outputs, size, id_dtype).save(str(part))
            descriptor.add_function(str(part), src_function_name="main", target_function_name=f"{prefix}{size}")
        descriptor.default_function_name = f"{prefix}{max(buckets)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        ct.utils.save_multifunction(descriptor, str(path))
    return path


def convert_encoder(model, path, buckets=ENCODER_BUCKETS, query_chunk=QUERY_CHUNK, id_dtype=np.int32):
    """Encoder .mlpackage with one function ``n{N}`` per bucket.

    ``id_dtype``: dtype of the tokens/stress/voice inputs. The ANE cannot read int32, so with int32
    the casts (and the planner then puts the embeddings with them) run on the CPU; np.float16
    (ids are small integers, exact in fp16) keeps the whole graph on the Neural Engine."""
    return _convert("encoder", model, path, buckets, "n", ENCODER_OUTPUTS, query_chunk, id_dtype)


def convert_decoder(model, path, buckets=DECODER_BUCKETS, query_chunk=QUERY_CHUNK, id_dtype=np.int32):
    """Decoder .mlpackage with one function ``t{T}`` per bucket. ``id_dtype`` is the voice input's dtype
    (see ``convert_encoder``); ``query_chunk`` splits each attention's queries into chunks of that size."""
    return _convert("decoder", model, path, buckets, "t", DECODER_OUTPUTS, query_chunk, id_dtype)


def functions(path):
    """Function names of a multifunction package, e.g. ['n32', 'n64', ...], sorted by size."""
    import coremltools as ct
    spec = ct.utils.load_spec(str(path))
    names = [f.name for f in spec.description.functions] or ["main"]
    return sorted(names, key=lambda name: int(name[1:]) if name[1:].isdigit() else 0)


def compute_plan_report(path, compute_units=None):
    """Per function: number of compute ops (non-const), the share on the Neural Engine by count and by
    estimated cost, and the ops that are not on the NE as (operator, device) counts.

    MLComputePlan only plans the default function of a multifunction model, so each function is
    copied into a single-function package of its own first."""
    import coremltools as ct
    compute_units = ct.ComputeUnit.CPU_AND_NE if compute_units is None else compute_units
    report = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in functions(path):
            single = Path(directory) / f"{name}.mlpackage"
            descriptor = ct.utils.MultiFunctionDescriptor()
            descriptor.add_function(str(path), src_function_name=name, target_function_name=name)
            descriptor.default_function_name = name
            ct.utils.save_multifunction(descriptor, str(single))
            report[name] = _plan(ct.utils.compile_model(str(single), str(Path(directory) / f"{name}.mlmodelc")), name, compute_units)
    return report


def _plan(compiled, name, compute_units):
    from coremltools.models.compute_device import MLNeuralEngineComputeDevice
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(str(compiled), compute_units)
    ops, on_ne, cost, cost_ne, others = 0, 0, 0.0, 0.0, {}
    for op in plan.model_structure.program.functions[name].block.operations:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = usage.preferred_compute_device if usage else None
        estimate = plan.get_estimated_cost_for_mlprogram_operation(op)
        weight = estimate.weight if estimate else 0.0
        ops += 1
        cost += weight
        if isinstance(device, MLNeuralEngineComputeDevice):
            on_ne += 1
            cost_ne += weight
        else:
            key = (op.operator_name, type(device).__name__.replace("ML", "").replace("ComputeDevice", "") if device else "unknown")
            others[key] = others.get(key, 0) + 1
    return {"ops": ops, "ne_ops": on_ne, "ne_share": on_ne / max(ops, 1), "ne_cost_share": cost_ne / cost if cost else 0.0,
            "not_ne": others}


def predict(mlmodel, inputs):
    """``mlmodel.predict`` that keeps the input arrays alive until the next call. Core ML (macOS 26)
    releases the bound inputs of an idle model later on a queue of its own, without the GIL; when that
    drops the last reference to a numpy array, the process crashes (SIGSEGV in _PyObject_Free)."""
    mlmodel._tiny_voca_inputs = inputs
    return mlmodel.predict(inputs)


def time_predict(path, function_name, inputs, compute_units, runs=20, warmup=3):
    """Median and minimum ``predict`` latency in milliseconds, and the load (compile) time in seconds."""
    import coremltools as ct
    start = time.perf_counter()
    mlmodel = ct.models.MLModel(str(path), compute_units=compute_units, function_name=function_name)
    load = time.perf_counter() - start
    for _ in range(warmup):
        predict(mlmodel, inputs)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        predict(mlmodel, inputs)
        times.append(1e3 * (time.perf_counter() - start))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "load_s": load}


def example_inputs(kind, model, size, id_dtype=np.int32):
    """Valid inputs for one bucket, in the Core ML input dtypes (for timing)."""
    return {k: v.astype(id_dtype if k in ID_INPUTS else np.float16) for k, v in _example(kind, model, size).items()}
