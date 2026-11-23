import CoreML
import Foundation

// Load a .mlmodelc bundle via the native macOS CoreML runtime.
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let config = MLModelConfiguration()
config.computeUnits = .cpuOnly
do {
    let model = try MLModel(contentsOf: url, configuration: config)
    let spec = model.modelDescription
    print("LOADED OK — no integrity verification occurred")
    // Try a prediction with dummy inputs
    do {
        let inputs = spec.inputDescriptionsByName
        var feat: [String: MLFeatureValue] = [:]
        for (name, desc) in inputs {
            if desc.type == .multiArray {
                let shape = desc.multiArrayConstraint!.shape.map { NSNumber(value: max($0.intValue, 1)) }
                let arr = try! MLMultiArray(shape: shape, dataType: .float32)
                feat[name] = MLFeatureValue(multiArray: arr)
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: feat)
        let out = try model.prediction(from: provider)
        print("PREDICTED OK — patched weights executed")
    } catch {
        print("(prediction not attempted: \(error.localizedDescription))")
    }
} catch {
    print("LOAD FAILED: \(error.localizedDescription)")
}
