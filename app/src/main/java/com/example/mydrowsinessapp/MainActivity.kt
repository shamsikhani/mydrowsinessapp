package com.example.mydrowsinessapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.LinearLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var ortSession: OrtSession
    private val frameBuffer = mutableListOf<FloatArray>()
    private var isProcessing = false
    private var lastPrediction = 0f
    private var lastPredictionState = false
    private var lastDebugInfo = ""
    private var faceDetector: FaceDetector? = null
    private var lastValidPredictionTime = 0L
    private val CONFIDENCE_THRESHOLD = 0.8f  // High confidence threshold
    private val FACE_DETECTION_CONFIDENCE = 0.7f  // Face detection confidence
    private val MIN_FACE_SIZE = 0.3f  // Minimum face size relative to image
    private val TEMPERATURE = 30.0f  // Higher values make predictions softer
    private val HIGH_THRESHOLD = 0.90f  // Only consider drowsy at 90% or higher
    private val LOW_THRESHOLD = 0.70f  // Anything below 90% is uncertain or active
    private val recentPredictions = ArrayDeque<Float>(10) // Keep last 10 predictions
    private var stateChangeCounter = 0 // Counter for consistent predictions
    private val REQUIRED_CONSISTENT_PREDICTIONS = 3 // Need 3 consistent predictions to change state

    private fun initializeFaceDetector() {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(MIN_FACE_SIZE)
            .build()
        faceDetector = FaceDetection.getClient(options)
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) {
        if (it) {
            // Camera permission is granted, camera preview is handled in CameraPreview composable
        } else {
            // Handle permission denial
        }
    }

    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                // Camera permission is granted, camera preview is handled in CameraPreview composable
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initializeFaceDetector()
        
        try {
            val modelBytes = assets.open("mobilevit_model.onnx").readBytes()
            val env = OrtEnvironment.getEnvironment()
            ortSession = env.createSession(modelBytes)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading model: ${e.message}")
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            DrowsinessDetectionScreen()
        }

        requestCameraPermission()
    }

    @Composable
    fun DrowsinessDetectionScreen() {
        var isDrowsy by remember { mutableStateOf(false) }
        var debugInfo by remember { mutableStateOf("Waiting for frames...") }
        var rawPrediction by remember { mutableStateOf(0f) }
        var faceDetected by remember { mutableStateOf(true) }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            // Camera Preview (Top Half)
            Box(modifier = Modifier.weight(1f)) {
                CameraPreview(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp)
                ) { imageProxy, faceDetectedValue ->
                    faceDetected = faceDetectedValue
                    processFrame(imageProxy) { drowsy, prediction, info ->
                        isDrowsy = drowsy
                        rawPrediction = prediction
                        debugInfo = info
                    }
                }
                
                Text(
                    text = if (!faceDetected) "NO FACE DETECTED" else if (isDrowsy) "DROWSY" else "ACTIVE",
                    color = if (!faceDetected) Color.Yellow else if (isDrowsy) Color.Red else Color.Green,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(16.dp)
                )
            }
            
            // Debug Info (Bottom Half)
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color.DarkGray)
                    .padding(16.dp)
            ) {
                Text(
                    text = "Debug Information",
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                
                Text(
                    text = "Raw Prediction: ${String.format("%.4f", rawPrediction)}",
                    color = Color.White,
                    fontSize = 16.sp,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
                
                Text(
                    text = "Status: ${if (isDrowsy) "DROWSY" else "ACTIVE"}",
                    color = if (isDrowsy) Color.Red else Color.Green,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
                
                Text(
                    text = "Processing Info:",
                    color = Color.White,
                    fontSize = 16.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                
                Text(
                    text = debugInfo,
                    color = Color.LightGray,
                    fontSize = 14.sp,
                    modifier = Modifier
                        .padding(start = 8.dp)
                        .verticalScroll(rememberScrollState())
                )
            }
        }
    }

    @Composable
    fun CameraPreview(
        modifier: Modifier = Modifier,
        onFrameReceived: (ImageProxy, Boolean) -> Unit
    ) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current

        AndroidView(
            factory = {
                PreviewView(it).apply {
                    this.scaleType = PreviewView.ScaleType.FILL_CENTER
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.MATCH_PARENT
                    )
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            modifier = modifier,
            update = { previewView ->
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder()
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                processImageProxy(imageProxy, onFrameReceived)
                            }
                        }

                    val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )

                    } catch (exc: Exception) {
                        Log.e("CameraPreview", "Use case binding failed", exc)
                    }
                }, ContextCompat.getMainExecutor(context))
            }
        )
    }

    private fun processImageProxy(
        imageProxy: ImageProxy,
        onFrameReceived: (ImageProxy, Boolean) -> Unit
    ) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            onFrameReceived(imageProxy, false)
            return
        }

        val imageForDetection = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        faceDetector?.process(imageForDetection)
            ?.addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    Log.d("FaceDetection", "No face detected")
                    onFrameReceived(imageProxy, false)
                } else {
                    Log.d("FaceDetection", "Face detected")
                    onFrameReceived(imageProxy, true)
                }
                imageProxy.close()
            }
            ?.addOnFailureListener { e ->
                Log.e("FaceDetection", "Face detection failed: ${e.message}")
                onFrameReceived(imageProxy, false)
                imageProxy.close()
            }
    }

    private fun getSmoothedPrediction(newPrediction: Float): Triple<Boolean, Float, String> {
        val debugInfo = StringBuilder()
        
        // Add new prediction to queue
        recentPredictions.addLast(newPrediction)
        if (recentPredictions.size > 10) {
            recentPredictions.removeFirst()
        }
        
        // Calculate average prediction
        val avgPrediction = recentPredictions.average().toFloat()
        debugInfo.append("Average prediction over last ${recentPredictions.size} frames: $avgPrediction\n")
        
        // Determine state with hysteresis
        val currentState = lastPredictionState
        when {
            avgPrediction >= HIGH_THRESHOLD -> {
                if (currentState) {
                    // Already drowsy, maintain state
                    debugInfo.append("Maintaining DROWSY state (>= 90%)\n")
                    return Triple(true, avgPrediction, debugInfo.toString())
                } else {
                    // Check if we have enough consistent high predictions
                    stateChangeCounter++
                    if (stateChangeCounter >= REQUIRED_CONSISTENT_PREDICTIONS) {
                        debugInfo.append("Changed to DROWSY after $REQUIRED_CONSISTENT_PREDICTIONS consistent predictions >= 90%\n")
                        stateChangeCounter = 0
                        return Triple(true, avgPrediction, debugInfo.toString())
                    }
                    debugInfo.append("Building confidence for DROWSY state: $stateChangeCounter/$REQUIRED_CONSISTENT_PREDICTIONS\n")
                    return Triple(false, avgPrediction, debugInfo.toString())
                }
            }
            else -> {
                // Below 90% threshold, consider active
                stateChangeCounter = 0
                if (avgPrediction > LOW_THRESHOLD) {
                    debugInfo.append("In uncertainty margin (${LOW_THRESHOLD}-${HIGH_THRESHOLD}), defaulting to ACTIVE state\n")
                } else {
                    debugInfo.append("Below ${LOW_THRESHOLD}, maintaining ACTIVE state\n")
                }
                return Triple(false, avgPrediction, debugInfo.toString())
            }
        }
    }

    private fun processFrame(
        imageProxy: ImageProxy,
        onResult: (Boolean, Float, String) -> Unit
    ) {
        if (isProcessing) {
            onResult(lastPredictionState, lastPrediction, lastDebugInfo)
            return
        }

        val processedFrame = ImageProcessor.preprocessFrame(imageProxy)
        frameBuffer.add(processedFrame)
        
        if (frameBuffer.size == 30) {
            val debugInfo = StringBuilder()
            debugInfo.append("Frame buffer size: ${frameBuffer.size}/30\n")
            
            isProcessing = true
            try {
                debugInfo.append("Preparing input tensor...\n")
                val shape = longArrayOf(1L, 30L, 3L, ImageProcessor.TARGET_SIZE.toLong(), ImageProcessor.TARGET_SIZE.toLong())
                val flattenedInput = FloatArray(30 * 3 * ImageProcessor.TARGET_SIZE * ImageProcessor.TARGET_SIZE)
                var idx = 0
                for (frame in frameBuffer) {
                    System.arraycopy(frame, 0, flattenedInput, idx, frame.size)
                    idx += frame.size
                }

                OrtEnvironment.getEnvironment().use { env ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(flattenedInput), shape).use { inputTensor ->
                        debugInfo.append("Running inference...\n")
                        val output = ortSession.run(mapOf("input_frames" to inputTensor))
                        val outputTensor = output[0].value as Array<FloatArray>
                        val logit = outputTensor[0][0]
                        
                        // Apply temperature scaling before sigmoid
                        val scaledLogit = logit / TEMPERATURE
                        debugInfo.append("Raw model output (logit): $logit\n")
                        debugInfo.append("Temperature scaled logit: $scaledLogit\n")
                        debugInfo.append("Temperature: $TEMPERATURE\n")
                        
                        // Apply sigmoid to get probability
                        val prediction = 1.0f / (1.0f + Math.exp(-scaledLogit.toDouble())).toFloat()
                        
                        debugInfo.append("Sigmoid prediction: $prediction\n")
                        
                        // Get smoothed prediction with temporal filtering
                        val (isDrowsy, smoothedPrediction, smoothingInfo) = getSmoothedPrediction(prediction)
                        debugInfo.append("\nTemporal Smoothing:\n")
                        debugInfo.append(smoothingInfo)
                        
                        // Update last prediction values
                        lastPrediction = smoothedPrediction
                        lastPredictionState = isDrowsy
                        lastDebugInfo = debugInfo.toString()
                        
                        onResult(lastPredictionState, lastPrediction, lastDebugInfo)
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error running model: ${e.message}")
                debugInfo.append("Error: ${e.message}\n")
                lastDebugInfo = debugInfo.toString()
                onResult(lastPredictionState, lastPrediction, lastDebugInfo)
            } finally {
                frameBuffer.clear()
                isProcessing = false
            }
        } else {
            onResult(lastPredictionState, lastPrediction, lastDebugInfo)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}