import Foundation

/// A mono 32-bit float WAV file (WAVE_FORMAT_IEEE_FLOAT).
enum WAV {
    static func write(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func put<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = samples.count * 4
        data.append(contentsOf: Array("RIFF".utf8)); put(UInt32(4 + 26 + 12 + 8 + bytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); put(UInt32(18))
        put(UInt16(3)); put(UInt16(1)); put(UInt32(sampleRate)); put(UInt32(sampleRate * 4)); put(UInt16(4)); put(UInt16(32)); put(UInt16(0))
        data.append(contentsOf: Array("fact".utf8)); put(UInt32(4)); put(UInt32(samples.count))
        data.append(contentsOf: Array("data".utf8)); put(UInt32(bytes))
        samples.withUnsafeBytes { data.append(contentsOf: $0) }       // little-endian on Apple platforms
        try data.write(to: url)
    }
}
