// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TinyVoca",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "TinyVoca", targets: ["TinyVoca"]),
        .executable(name: "tiny-voca-coreml", targets: ["tiny-voca-coreml"]),
    ],
    targets: [
        .target(name: "TinyVoca"),
        .executableTarget(name: "tiny-voca-coreml", dependencies: ["TinyVoca"]),
    ]
)
