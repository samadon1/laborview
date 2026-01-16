import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let coreMLAnalyzer = CoreMLAnalyzer()

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    // Setup CoreML platform channel
    let controller = window?.rootViewController as! FlutterViewController
    let coreMLChannel = FlutterMethodChannel(
      name: "com.laborview/coreml",
      binaryMessenger: controller.binaryMessenger
    )

    coreMLChannel.setMethodCallHandler { [weak self] (call, result) in
      guard let self = self else {
        result(FlutterError(code: "UNAVAILABLE", message: "AppDelegate unavailable", details: nil))
        return
      }

      switch call.method {
      case "loadModel":
        NSLog("CoreML AppDelegate: loadModel called")
        let success = self.coreMLAnalyzer.loadModel()
        NSLog("CoreML AppDelegate: loadModel returned %@", success ? "true" : "false")
        result(success)

      case "analyze":
        guard let args = call.arguments as? [String: Any],
              let imageData = args["imageData"] as? FlutterStandardTypedData else {
          result(FlutterError(code: "INVALID_ARGS", message: "Missing imageData", details: nil))
          return
        }

        // Run analysis on background thread
        DispatchQueue.global(qos: .userInitiated).async {
          let analysisResult = self.coreMLAnalyzer.analyze(imageData: imageData)
          DispatchQueue.main.async {
            result(analysisResult)
          }
        }

      case "analyzeFromPath":
        NSLog("CoreML AppDelegate: analyzeFromPath called")
        guard let args = call.arguments as? [String: Any] else {
          NSLog("CoreML AppDelegate: args is nil or wrong type")
          result(FlutterError(code: "INVALID_ARGS", message: "Arguments not a dictionary", details: nil))
          return
        }
        NSLog("CoreML AppDelegate: args = %@", args)
        guard let imagePath = args["imagePath"] as? String else {
          NSLog("CoreML AppDelegate: imagePath missing from args")
          result(FlutterError(code: "INVALID_ARGS", message: "Missing imagePath", details: nil))
          return
        }
        NSLog("CoreML AppDelegate: imagePath = %@", imagePath)

        // Run analysis on background thread
        DispatchQueue.global(qos: .userInitiated).async {
          NSLog("CoreML AppDelegate: calling analyzeFromPath")
          let analysisResult = self.coreMLAnalyzer.analyzeFromPath(imagePath: imagePath)
          NSLog("CoreML AppDelegate: result = %@", String(describing: analysisResult))
          DispatchQueue.main.async {
            result(analysisResult)
          }
        }

      case "isAvailable":
        result(true)

      default:
        result(FlutterMethodNotImplemented)
      }
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
