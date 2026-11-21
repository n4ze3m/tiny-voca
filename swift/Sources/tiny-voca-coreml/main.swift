// tiny-voca-coreml: speak phoneme ids (from swift/phonemize.py) with the Core ML models,
// time the stages, or print where Core ML runs the ops.

import CoreML
import Foundation
import TinyVoca

let usage = """
    usage: tiny-voca-coreml --models DIR --ids ids.json --output out.wav [--voice F2|M2] [--speed X]
                            [--pitch-shift X] [--pitch-range X] [--seed N] [--bench N] [--units ane|cpu|all|gpu]
                            [--vocoder-units gpu|cpu|all] [--noise noise.f32] [--dump DIR] [--load-times]
           tiny-voca-coreml --plan --models DIR [--units ane|cpu|all|gpu]

      --ids          JSON {"ids": [...], "stress": [...]} or a list of them (sentences, joined)
      --bench N      load every function, then time N runs of every utterance (after 3 warm-up runs);
                     print stage medians (without --bench only the needed functions are loaded)
      --units        compute units of Encoder and Decoder (default ane = CPU and Neural Engine)
      --noise        raw float32 noise for all utterances, one after the other (parity checks)
      --dump DIR     write intermediate values of each utterance as raw float32 / int32 files
      --plan         share of ops on the Neural Engine per Encoder/Decoder function (MLComputePlan)
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func parse() -> (options: [String: String], flags: Set<String>) {
    var options: [String: String] = [:], flags = Set<String>()
    var arguments = CommandLine.arguments.dropFirst()
    while let argument = arguments.popFirst() {
        guard argument.hasPrefix("--") else { fail(usage) }
        let key = String(argument.dropFirst(2))
        if ["plan", "load-times", "help"].contains(key) { flags.insert(key); continue }
        guard let value = arguments.popFirst() else { fail("missing value for \(argument)\n\(usage)") }
        options[key] = value
    }
    return (options, flags)
}
let (options, flags) = parse()
if flags.contains("help") { print(usage); exit(0) }
guard let modelsPath = options["models"] else { fail(usage) }
let models = URL(filePath: modelsPath)

func units(_ name: String?, _ fallback: MLComputeUnits) -> MLComputeUnits {
    switch name {
    case nil: return fallback
    case "ane": return .cpuAndNeuralEngine
    case "cpu": return .cpuOnly
    case "gpu": return .cpuAndGPU
    case "all": return .all
    default: fail("unknown compute units \(name!)")
    }
}
let neuralUnits = units(options["units"], .cpuAndNeuralEngine)
let vocoderUnits = units(options["vocoder-units"], .cpuAndGPU)

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.count % 2 == 1 ? sorted[sorted.count / 2] : 0.5 * (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2])
}

func percent(_ part: Double, _ whole: Double) -> String { whole > 0 ? String(format: "%.1f%%", 100 * part / whole) : "-" }

let clockStart = Date()
let voca: TinyVoca
do {
    voca = try await TinyVoca(modelsDirectory: models, neuralUnits: neuralUnits, vocoderUnits: vocoderUnits)
} catch {
    fail("cannot open the models: \(error)")
}

if flags.contains("plan") {
    print("function        ops  on NE  NE share  NE cost share  not on NE")
    for (name, prefix, sizes) in [("Encoder", "n", voca.encoderBuckets), ("Decoder", "t", voca.decoderBuckets)] {
        for size in sizes {
            do {
                let plan = try await TinyVoca.computePlan(voca.url(name), function: "\(prefix)\(size)", units: neuralUnits)
                let others = plan.elsewhere.sorted { $0.key < $1.key }.map { "\($0.key) x\($0.value)" }.joined(separator: ", ")
                let label = "\(name) \(plan.function)".padding(toLength: 14, withPad: " ", startingAt: 0)
                let pad = { (text: String, width: Int) in String(repeating: " ", count: max(0, width - text.count)) + text }
                print(label + String(format: "%5d %6d", plan.ops, plan.neuralEngineOps)
                      + pad(percent(Double(plan.neuralEngineOps), Double(plan.ops)), 10) + pad(percent(plan.neuralEngineCost, plan.cost), 15)
                      + "  " + (others.isEmpty ? "-" : others))
            } catch {
                fail("compute plan of \(name) \(prefix)\(size): \(error)")
            }
        }
    }
    let plan = try await TinyVoca.computePlan(voca.url("Vocoder"), function: nil, units: vocoderUnits)
    let devices = plan.elsewhere.reduce(into: [String: Int]()) { $0[String($1.key.split(separator: "@").last!), default: 0] += $1.value }
    let elsewhere = devices.sorted { $0.key < $1.key }.map { "\($0.value) on \($0.key)" }.joined(separator: ", ")
    print("Vocoder (\(options["vocoder-units"] ?? "gpu")): \(plan.ops) ops, \(plan.neuralEngineOps) on NE, \(elsewhere)")
    exit(0)
}

// Utterances.
guard let idsPath = options["ids"] else { fail(usage) }
struct Utterance { var ids: [Int]; var stress: [Int] }
let utterances: [Utterance]
do {
    let json = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(filePath: idsPath)))
    let items = (json as? [[String: Any]]) ?? ((json as? [String: Any]).map { [$0] } ?? [])
    utterances = try items.map { item in
        guard let ids = item["ids"] as? [Int], let stress = item["stress"] as? [Int] else {
            throw TinyVocaError.badInput("each utterance needs integer lists ids and stress")
        }
        return Utterance(ids: ids, stress: stress)
    }
    guard !utterances.isEmpty else { throw TinyVocaError.badInput("no utterances") }
} catch {
    fail("cannot read \(idsPath): \(error)")
}

let voiceName = options["voice"] ?? Config.speakers[0]
guard let voice = Config.speakers.firstIndex(of: voiceName) ?? Int(voiceName) else { fail("unknown voice \(voiceName)") }
func number(_ key: String, _ fallback: Float) -> Float {
    guard let text = options[key] else { return fallback }
    guard let value = Float(text) else { fail("--\(key) needs a number") }
    return value
}
let speed = number("speed", 1), pitchShift = number("pitch-shift", 0), pitchRange = number("pitch-range", 1)
let seed = UInt64(options["seed"] ?? "0") ?? 0
var externalNoise: [Float]? = nil
if let path = options["noise"] {
    guard let data = FileManager.default.contents(atPath: path) else { fail("cannot read \(path)") }
    externalNoise = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

// --bench and --load-times load every function first (in parallel), so that the timings are steady-state.
// Otherwise only the functions the utterances need are loaded, on first use.
if options["bench"] != nil || flags.contains("load-times") {
    let loadStart = Date()
    do { try voca.preload() } catch { fail("cannot load the models: \(error)") }
    let loaded = Date()
    print(String(format: "load: %.3f s for %d functions (%.3f s since start, with opening the model assets)",
                 loaded.timeIntervalSince(loadStart), voca.loadTimes.count, loaded.timeIntervalSince(clockStart)))
    if flags.contains("load-times") {
        for (name, seconds) in voca.loadTimes { print("  " + name.padding(toLength: 14, withPad: " ", startingAt: 0) + String(format: "%.3f s", seconds)) }
    }
}

func write(_ values: [Float], _ url: URL) throws { try values.withUnsafeBytes { try Data($0).write(to: url) } }

var waves: [[Float]] = []
var noiseOffset = 0
for (number, utterance) in utterances.enumerated() {
    var noise: [Float]? = nil
    do {
        if let all = externalNoise {
            // The frame count is known only after the encoder: run once to get it, then take that much noise.
            let probe = try voca.run(tokens: utterance.ids, stress: utterance.stress, voice: voice, speed: speed,
                                     pitchShift: pitchShift, pitchRange: pitchRange, seed: seed)
            let samples = probe.audio.count
            guard noiseOffset + samples <= all.count else { fail("--noise has too few samples") }
            noise = Array(all[noiseOffset..<(noiseOffset + samples)])
            noiseOffset += samples
        }
        let result = try voca.run(tokens: utterance.ids, stress: utterance.stress, voice: voice, speed: speed, pitchShift: pitchShift,
                                  pitchRange: pitchRange, seed: seed + UInt64(number), noise: noise)
        guard result.audio.allSatisfy({ $0.isFinite }) else { fail("utterance \(number): non-finite audio") }
        waves.append(result.audio)
        if let directory = options["dump"] {
            let base = URL(filePath: directory)
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            for (name, values) in [("hidden", result.hidden), ("duration", result.rawDuration), ("pitch", result.pitch),
                                   ("harmonic", result.harmonic), ("noise_params", result.noiseParams), ("f0", result.f0),
                                   ("pulses", result.pulses), ("noise", result.noise), ("audio", result.audio)] {
                try write(values, base.appending(path: "u\(number)-\(name).f32"))
            }
            try result.durations.map { Int32($0) }.withUnsafeBytes { try Data($0).write(to: base.appending(path: "u\(number)-durations.i32")) }
        }
        if let text = options["bench"] {
            guard let runs = Int(text), runs > 0 else { fail("--bench needs a positive count") }
            for _ in 0..<3 {
                _ = try voca.run(tokens: utterance.ids, stress: utterance.stress, voice: voice, speed: speed, pitchShift: pitchShift,
                                 pitchRange: pitchRange, seed: seed, noise: noise)
            }
            var times: [StageTimes] = []
            for _ in 0..<runs {
                times.append(try voca.run(tokens: utterance.ids, stress: utterance.stress, voice: voice, speed: speed,
                                          pitchShift: pitchShift, pitchRange: pitchRange, seed: seed, noise: noise).times)
            }
            if number == 0 {
                print("utt tokens frames  buckets     audio s  encoder     host  decoder  pulse+noise  vocoder    total (ms)  xRT    RTF")
            }
            let m = { (stage: KeyPath<StageTimes, Double>) in 1e3 * median(times.map { $0[keyPath: stage] }) }
            let seconds = Double(result.audio.count) / Double(Config.sampleRate)
            let total = m(\.total)
            print(String(format: "%3d %6d %6d  n%-4d t%-4d %7.2f %8.2f %8.2f %8.2f %12.2f %8.2f %8.2f %8.1f %6.4f", number,
                         utterance.ids.count, result.durations.reduce(0, +), result.encoderBucket, result.decoderBucket, seconds,
                         m(\.encoder), m(\.host), m(\.decoder), m(\.source), m(\.vocoder), total, seconds / (total / 1e3),
                         (total / 1e3) / seconds))
        }
    } catch {
        fail("utterance \(number): \(error)")
    }
}

if options["bench"] == nil && !flags.contains("load-times") {
    let names = voca.loadTimes.map(\.name).joined(separator: ", ")
    print(String(format: "loaded %@ in %.3f s; %.3f s since start", names, voca.loadTimes.reduce(0) { $0 + $1.seconds },
                 Date().timeIntervalSince(clockStart)))
}
if let output = options["output"] {
    let wave = Host.limitPeak(waves.flatMap { $0 })
    do { try WAV.write(wave, sampleRate: Config.sampleRate, to: URL(filePath: output)) } catch { fail("cannot write \(output): \(error)") }
    print(String(format: "wrote %@: %.2f s of audio from %d utterance(s)", output, Double(wave.count) / Double(Config.sampleRate), waves.count))
}
