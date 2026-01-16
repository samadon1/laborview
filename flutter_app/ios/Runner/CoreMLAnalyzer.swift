import Foundation
import CoreML
import UIKit

/// CoreML-based analyzer for LaborView ultrasound segmentation
/// Uses the Xcode-generated model class for type-safe inference
class CoreMLAnalyzer: NSObject {

    private var model: laborview_medsiglip_fp16?
    private let inputSize = 448

    /// Initialize and load the CoreML model
    func loadModel() -> Bool {
        NSLog("CoreML loadModel: Starting with auto-generated class...")

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU
            model = try laborview_medsiglip_fp16(configuration: config)
            NSLog("CoreML loadModel: Model loaded successfully!")
            return true
        } catch {
            NSLog("CoreML loadModel: Failed to load model: %@", error.localizedDescription)
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

        return runInference(on: uiImage, with: model)
    }

    /// Analyze from file path
    func analyzeFromPath(imagePath: String) -> [String: Any]? {
        NSLog("CoreML analyzeFromPath: path = %@", imagePath)

        guard let model = model else {
            NSLog("CoreML analyzeFromPath: Model is nil!")
            return ["error": "Model not loaded"]
        }

        // Load image
        var uiImage: UIImage?

        // Try direct file path
        uiImage = UIImage(contentsOfFile: imagePath)

        // Try via FileManager
        if uiImage == nil, let data = FileManager.default.contents(atPath: imagePath) {
            uiImage = UIImage(data: data)
        }

        guard let image = uiImage else {
            NSLog("CoreML analyzeFromPath: Failed to load image")
            return ["error": "Failed to read image file"]
        }

        NSLog("CoreML analyzeFromPath: Image loaded, size = %@", NSCoder.string(for: image.size))
        return runInference(on: image, with: model)
    }

    // MARK: - Private Methods

    private func runInference(on image: UIImage, with model: laborview_medsiglip_fp16) -> [String: Any]? {
        NSLog("CoreML runInference: Starting...")

        // Create input array
        guard let inputArray = createInputArray(from: image) else {
            NSLog("CoreML runInference: Failed to create input array")
            return ["error": "Failed to preprocess image"]
        }

        NSLog("CoreML runInference: Input array created, running prediction...")

        do {
            // Use the auto-generated prediction method
            let output = try model.prediction(pixel_values: inputArray)

            NSLog("CoreML runInference: Prediction successful!")

            // Process segmentation mask
            let segMask = processSegmentation(output.seg_probs)

            // Get plane prediction
            let planePred = Int(truncating: output.plane_pred[0])

            // Compute metrics
            let metrics = computeMetrics(segMask: segMask)

            return [
                "segmentation": segMask,
                "planeClass": planePred == 0 ? "Transperineal" : "Other",
                "aop": metrics["aop"] as Any,
                "hsd": metrics["hsd"] as Any,
                "headCircumference": metrics["hc"] as Any
            ]

        } catch {
            NSLog("CoreML runInference: Prediction failed: %@", error.localizedDescription)
            return ["error": "Inference failed: \(error.localizedDescription)"]
        }
    }

    private func createInputArray(from image: UIImage) -> MLMultiArray? {
        NSLog("CoreML createInputArray: Starting...")

        // Resize image
        guard let resizedImage = resizeImage(image, to: CGSize(width: inputSize, height: inputSize)),
              let cgImage = resizedImage.cgImage else {
            NSLog("CoreML createInputArray: Failed to resize image")
            return nil
        }

        // Create MLMultiArray
        let inputArray: MLMultiArray
        do {
            inputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)], dataType: .float32)
        } catch {
            NSLog("CoreML createInputArray: Failed to create MLMultiArray: %@", error.localizedDescription)
            return nil
        }

        // Extract pixel data
        let width = inputSize
        let height = inputSize
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width

        var rawData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            NSLog("CoreML createInputArray: Failed to create CGContext")
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Fill MLMultiArray with normalized values [-1, 1]
        let channelStride = inputSize * inputSize

        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let pixelIndex = (y * width + x) * bytesPerPixel
                let spatialIndex = y * inputSize + x

                let r = Float(rawData[pixelIndex]) / 255.0
                let g = Float(rawData[pixelIndex + 1]) / 255.0
                let b = Float(rawData[pixelIndex + 2]) / 255.0

                // Normalize to [-1, 1] for SigLIP
                inputArray[spatialIndex] = NSNumber(value: (r - 0.5) * 2.0)
                inputArray[channelStride + spatialIndex] = NSNumber(value: (g - 0.5) * 2.0)
                inputArray[2 * channelStride + spatialIndex] = NSNumber(value: (b - 0.5) * 2.0)
            }
        }

        NSLog("CoreML createInputArray: Success! Array count = %d", inputArray.count)
        return inputArray
    }

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    private func processSegmentation(_ multiArray: MLMultiArray) -> [[Int]] {
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
        guard let symphysisLowest = symphysis.max(by: { $0.y < $1.y }),
              let headLowest = head.max(by: { $0.y < $1.y }) else {
            return nil
        }

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
        let lineDir = (x: 1.0, y: slope)

        let headVector = (
            x: Double(headLowest.x - symphysisLowest.x),
            y: Double(headLowest.y - symphysisLowest.y)
        )

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
