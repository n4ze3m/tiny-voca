"""Compare the Core ML models with the PyTorch model and time them.

    python check_coreml.py --checkpoint tiny-voca.pt --models coreml [--output coreml-check]

``--models`` is a directory written by ``python -m tiny_voca.coreml.export``.
Parity: Core ML is given PyTorch's durations and noise, so the audio can be
compared sample by sample (log-mel distance and SNR); free-running durations
are compared token by token. Speed: each stage of the Core ML pipeline, with
the Encoder and Decoder on the Neural Engine and on the CPU only, against
PyTorch on ``--threads`` CPU threads. Last, the share of the ops that the
compute plan puts on the Neural Engine. WAV files are written for listening.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import soundfile as sf
import torch

from tiny_voca.coreml import ane, vocoder as V
from tiny_voca.coreml.runtime import STAGES, CoreMLVoca
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.losses import SpectralLoss
from tiny_voca.text import phonemize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_onnx import SENTENCES  # noqa: E402


def snr(reference, value):
    return float(10 * np.log10(np.sum(reference.astype(np.float64) ** 2) / max(np.sum((reference.astype(np.float64) - value) ** 2), 1e-30)))


def median_times(run, runs, warmup=5):
    for _ in range(warmup):
        run()
    rows = [run() for _ in range(runs)]
    return {key: statistics.median(row[key] for row in rows) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("coreml-check"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    rate, hop = model.config.sample_rate, model.config.hop
    voca = CoreMLVoca(args.models).preload()
    cpu = CoreMLVoca(args.models, {"encoder": "CPU_ONLY", "decoder": "CPU_ONLY"}).preload()
    mel = SpectralLoss(rate)
    args.output.mkdir(parents=True, exist_ok=True)

    print("# parity: Core ML with PyTorch's durations and noise; free-running durations")
    matched = total_tokens = 0
    for number, text in enumerate(SENTENCES):
        phones = phonemize(text)
        for voice, name in enumerate(model.config.speakers):
            tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
            reference, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]))
            reference = reference[0, :int(lengths[0])].numpy()
            durations = details["durations"][0].numpy()
            noise = torch.randn(len(reference), generator=torch.Generator().manual_seed(0)).numpy()
            audio, same_details = voca.synthesize(phones.ids, phones.stress, voice, durations=durations, noise=noise)
            free, free_details = voca.synthesize(phones.ids, phones.stress, voice)
            same = int((free_details["durations"] == durations).sum())
            matched, total_tokens = matched + same, total_tokens + len(durations)
            # The phase of the pulse train integrates F0, so a tiny F0 error moves the waveform after a few
            # periods: the sample SNR of the full pipeline says little. Run the Core ML vocoder with PyTorch's
            # F0 and Core ML's parameters too: its SNR measures the parameter error alone.
            f0 = model.frequency(details["prosody"][0, :, 0]).numpy()
            torch_f0 = ane.predict(voca._model("vocoder"), V.vocoder_inputs(same_details["harmonic"], same_details["noise_params"], f0,
                                                                            noise, config=model.config))["audio"][0]
            other_seed, _, _ = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]), seed=1)
            cents = 1200 * np.abs(np.log2(same_details["f0"] / f0))
            log_mel = lambda wave: mel.log_mel(torch.as_tensor(np.asarray(wave)).reshape(1, -1))
            distance = lambda wave: round(float((log_mel(wave) - log_mel(reference)).abs().mean()), 4)
            print(json.dumps({"sentence": number, "voice": name, "tokens": len(phones), "seconds": round(len(reference) / rate, 2),
                              "log_mel_l1": distance(audio), "log_mel_l1_torch_other_noise": distance(other_seed[0, :len(reference)]),
                              "snr_db": round(snr(reference, audio), 1), "snr_db_torch_f0": round(snr(reference, torch_f0), 1),
                              "f0_cents_mean": round(float(cents.mean()), 2), "f0_cents_max": round(float(cents.max()), 2),
                              "free_durations_exact": round(same / len(durations), 4),
                              "free_frames": int(free_details["durations"].sum()), "torch_frames": int(durations.sum())}))
            sf.write(args.output / f"s{number}-{name}-torch.wav", reference, rate)
            sf.write(args.output / f"s{number}-{name}-coreml-same.wav", audio, rate)
            sf.write(args.output / f"s{number}-{name}-coreml.wav", free, rate)
    print(json.dumps({"free_durations_exact_all": round(matched / total_tokens, 4), "tokens": total_tokens}))

    print(f"\n# speed (median of {args.runs} runs, ms; rtfx = audio seconds / compute seconds)")
    for number, text in enumerate(() if args.skip_speed else SENTENCES):
        phones = phonemize(text)
        tokens, stress, length = torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)])
        for voice, name in enumerate(model.config.speakers):
            seconds = len(voca.synthesize(phones.ids, phones.stress, voice)[0]) / rate
            row = {"sentence": number, "voice": name, "tokens": len(phones), "audio_s": round(seconds, 2)}
            for label, runtime in (("ane", voca), ("cpu", cpu)):
                def run():
                    start = time.perf_counter()
                    _, details = runtime.synthesize(phones.ids, phones.stress, voice)
                    return {**details["times"], "total": time.perf_counter() - start}
                times = median_times(run, args.runs)
                row[label] = {key: round(1e3 * times[key], 2) for key in (*STAGES, "total")}
                row[label]["rtfx"] = round(seconds / times["total"])

            def run_torch():
                start = time.perf_counter()
                model.synthesize(tokens, stress, length, torch.tensor([voice]))
                return {"total": time.perf_counter() - start}
            times = median_times(run_torch, args.runs)
            row["torch"] = {"total": round(1e3 * times["total"], 2), "rtfx": round(seconds / times["total"]), "threads": args.threads}
            row["ane_vs_cpu"] = round(row["cpu"]["total"] / row["ane"]["total"], 2)
            row["ane_vs_torch"] = round(row["torch"]["total"] / row["ane"]["total"], 2)
            print(json.dumps(row))

    if not args.skip_plan:
        print("\n# compute plan (CPU_AND_NE): share of ops on the Neural Engine")
        for kind in ("encoder", "decoder"):
            entry = voca.metadata["models"][kind]
            for function, row in ane.compute_plan_report(args.models / entry["package"]).items():
                print(json.dumps({"model": kind, "function": function, "ops": row["ops"], "ne_ops": row["ne_ops"],
                                  "ne_share": round(row["ne_share"], 4), "ne_cost_share": round(row["ne_cost_share"], 4),
                                  "not_ne": {f"{op}@{device}": count for (op, device), count in row["not_ne"].items()}}))


if __name__ == "__main__":
    main()
