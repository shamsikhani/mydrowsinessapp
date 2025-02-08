package com.example.mydrowsinessapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
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
import androidx.compose.material3.Text
import androidx.compose.runtime.*
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
import kotlinx.coroutines.*
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var ortSession: OrtSession
    private val frameBuffer = ArrayList<FloatArray>(30)
    private var isProcessing = false
    private var lastPrediction = 0f
    private var lastPredictionState = false
    private var lastDebugInfo = ""

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
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
                ) { imageProxy ->
                    processFrame(imageProxy) { drowsy, prediction, info ->
                        isDrowsy = drowsy
                        rawPrediction = prediction
                        debugInfo = info
                    }
                }
                
                Text(
                    text = if (isDrowsy) "DROWSY" else "ACTIVE",
                    color = if (isDrowsy) Color.Red else Color.Green,
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
        onFrameReceived: (ImageProxy) -> Unit
    ) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            modifier = modifier,
            update = { previewView ->
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { image ->
                                onFrameReceived(image)
                                image.close()
                            }
                        }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview,
                            imageAnalyzer
                        )
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Camera binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(context))
            }
        )
    }

    private fun processFrame(imageProxy: ImageProxy, onResult: (Boolean, Float, String) -> Unit) {
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
                        
                        // Apply sigmoid to get probability (exactly as in training)
                        val prediction = 1.0f / (1.0f + Math.exp(-logit.toDouble())).toFloat()
                        val isDrowsy = prediction >= 0.5f  // Same threshold as training
                        
                        debugInfo.append("Raw model output (logit): $logit\n")
                        debugInfo.append("Sigmoid prediction: $prediction\n")
                        debugInfo.append("Threshold: 0.5\n")
                        debugInfo.append("State: ${if (isDrowsy) "DROWSY (3AM)" else "ACTIVE (10AM)"}\n")
                        
                        // Update last prediction values
                        lastPrediction = prediction
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

    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        // Camera is started in the CameraPreview composable
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}