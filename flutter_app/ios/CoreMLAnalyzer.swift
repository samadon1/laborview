import Foundation
import CoreML
import Vision
import UIKit

/// CoreML-based analyzer for LaborView ultrasound segmentation
/// Uses Neural Engine for optimized on-device inference
class CoreMLAnalyzer: NSObject {

    private var model: MLModel?
    private let inputSize = 448

    /// Initialize and load the CoreML model
    func loadModel() -> Bool {
        guard let modelURL = Bundle.main.url(forResource: "laborview_medsiglip_fp16", withExtension: "mlmodelc") else {
            // Try .mlpackage if .mlmodelc not found
            if let packageURL = Bundle.main.url(forResource: "laborview_medsiglip_fp16", withExtension: "mlpackage") {
                do {
                    let compiledURL = try MLModel.compileModel(at: packageURL)
                    model = try MLModel(contentsOf: compiledURL)
                    return true
                } catch {
                    print("CoreML: Failed to compile mlpackage: \(error)")
                    return false
                }
            }
            print("CoreML: Model not found in bundle")
            return false
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU
            model = try MLModel(contentsOf: modelURL, configuration: config)
            return true
        } catch {
            print("CoreML: Failed to load model: \(error)")
            return false
        }
    }

    /// Analyze an ultrasound image and return segmentation results
    func analyze(imageData: FlutterStandardTypedData) -> [String: Any]? {
        guard let model = model else {
            return ["error": "Model not loaded"]
        }

        guard let uiImage = UIImage(data: imageData.data) else {
            return ["error": "Invalid image data"]
        }

        // Preprocess image
        guard let pixelBuffer = preprocessImage(uiImage) else {
            return ["error": "Failed to preprocess image"]
        }

        // Run inference
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let output = try model.prediction(from: input)

            // Extract outputs
            guard let segProbsFeature = output.featureValue(for: "seg_probs"),
                  let segProbsArray = segProbsFeature.multiArrayValue,
                  let planePredFeature = output.featureValue(for: "plane_pred") else {
                return ["error": "Failed to extract model outputs"]
            }

            // Process segmentation mask
            let segMask = processSegmentation(segProbsArray)

            // Get plane prediction
            let planePred: Int
            if let planePredArray = planePredFeature.multiArrayValue {
                planePred = Int(truncating: planePredArray[0])
            } else {
                planePred = 0
            }

            // Compute metrics from segmentation
            let metrics = computeMetrics(segMask: segMask)

            return [
                "segmentation": segMask,
                "planeClass": planePred == 0 ? "Transperineal" : "Other",
                "aop": metrics["aop"] as Any,
                "hsd": metrics["hsd"] as Any,
                "headCircumference": metrics["hc"] as Any
            ]

        } catch {
            return ["error": "Inference failed: \(error.localizedDescription)"]
        }
    }

    /// Analyze from file path
    func analyzeFromPath(imagePath: String) -> [String: Any]? {
        guard let imageData = FileManager.default.contents(atPath: imagePath) else {
            return ["error": "Failed to read image file"]
        }

        let typedData = FlutterStandardTypedData(bytes: imageData)
        return analyze(imageData: typedData)
    }

    // MARK: - Private Methods

    private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
        let targetSize = CGSize(width: inputSize, height: inputSize)

        // Resize image
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()

        // Create pixel buffer
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputSize,
            inputSize,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: inputSize,
            height: inputSize,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return nil
        }

        guard let cgImage = resizedImage.cgImage else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))

        return buffer
    }

    private func processSegmentation(_ multiArray: MLMultiArray) -> [[Int]] {
        // Output shape: [1, 3, 448, 448] - batch, classes, height, width
        let height = inputSize
        let width = inputSize
        let numClasses = 3

        var mask = [[Int]](repeating: [Int](repeating: 0, count: width), count: height)

        for y in 0..<height {
            for x in 0..<width {
                var maxClass = 0
                var maxProb: Float = -Float.infinity

                for c in 0..<numClasses {
                    let index = c * height * width + y * width + x
                    let prob = multiArray[index].floatValue
                    if prob > maxProb {
                        maxProb = prob
                        maxClass = c
                    }
                }
                mask[y][x] = maxClass
            }
        }

        return mask
    }

    private func computeMetrics(segMask: [[Int]]) -> [String: Double?] {
        let symphysisPoints = findContour(mask: segMask, classId: 1)
        let headPoints = findContour(mask: segMask, classId: 2)

        guard symphysisPoints.count >= 10, headPoints.count >= 10 else {
            return ["aop": nil, "hsd": nil, "hc": nil]
        }

        let aop = computeAoP(symphysis: symphysisPoints, head: headPoints)
        let hsd = computeHSD(symphysis: symphysisPoints, head: headPoints)
        let hc = computeCircumference(contour: headPoints)

        return ["aop": aop, "hsd": hsd, "hc": hc]
    }

    private func findContour(mask: [[Int]], classId: Int) -> [(x: Int, y: Int)] {
        var points: [(x: Int, y: Int)] = []
        let height = mask.count
        let width = mask[0].count

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                if mask[y][x] == classId {
                    // Check if boundary pixel
                    if mask[y-1][x] != classId ||
                       mask[y+1][x] != classId ||
                       mask[y][x-1] != classId ||
                       mask[y][x+1] != classId {
                        points.append((x: x, y: y))
                    }
                }
            }
        }

        return points
    }

    private func computeAoP(symphysis: [(x: Int, y: Int)], head: [(x: Int, y: Int)]) -> Double? {
        // Find lowest points (highest y value)
        guard let symphysisLowest = symphysis.max(by: { $0.y < $1.y }),
              let headLowest = head.max(by: { $0.y < $1.y }) else {
            return nil
        }

        // Fit line to symphysis using simple regression
        let meanX = Double(symphysis.reduce(0) { $0 + $1.x }) / Double(symphysis.count)
        let meanY = Double(symphysis.reduce(0) { $0 + $1.y }) / Double(symphysis.count)

        var covXX: Double = 0
        var covXY: Double = 0
        for p in symphysis {
            let dx = Double(p.x) - meanX
            let dy = Double(p.y) - meanY
            covXX += dx * dx
            covXY += dx * dy
        }

        let slope = covXY / (covXX + 1e-6)

        // Line direction vector
        let lineDir = (x: 1.0, y: slope)

        // Head vector from symphysis lowest to head lowest
        let headVector = (
            x: Double(headLowest.x - symphysisLowest.x),
            y: Double(headLowest.y - symphysisLowest.y)
        )

        // Calculate angle
        let dot = lineDir.x * headVector.x + lineDir.y * headVector.y
        let mag1 = sqrt(lineDir.x * lineDir.x + lineDir.y * lineDir.y)
        let mag2 = sqrt(headVector.x * headVector.x + headVector.y * headVector.y)

        guard mag1 > 0, mag2 > 0 else { return nil }

        var angle = acos(abs(dot) / (mag1 * mag2)) * 180 / .pi

        if headLowest.y > symphysisLowest.y {
            angle = 90 + (90 - angle)
        }

        return angle
    }

    private func computeHSD(symphysis: [(x: Int, y: Int)], head: [(x: Int, y: Int)]) -> Double? {
        guard let symphysisLowest = symphysis.max(by: { $0.y < $1.y }) else {
            return nil
        }

        var minDist = Double.infinity
        for p in head {
            let dist = sqrt(
                pow(Double(p.x - symphysisLowest.x), 2) +
                pow(Double(p.y - symphysisLowest.y), 2)
            )
            if dist < minDist {
                minDist = dist
            }
        }

        return minDist.isFinite ? minDist : nil
    }

    private func computeCircumference(contour: [(x: Int, y: Int)]) -> Double? {
        guard contour.count >= 10 else { return nil }

        var perimeter: Double = 0
        for i in 0..<contour.count {
            let next = contour[(i + 1) % contour.count]
            perimeter += sqrt(
                pow(Double(contour[i].x - next.x), 2) +
                pow(Double(contour[i].y - next.y), 2)
            )
        }

        return perimeter
    }
}
