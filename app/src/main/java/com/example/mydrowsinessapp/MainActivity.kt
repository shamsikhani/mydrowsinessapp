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
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
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

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize ONNX Runtime session
        initOrtSession()
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            DrowsinessDetectionScreen()
        }

        requestCameraPermission()
    }

    private fun initOrtSession() {
        try {
            val ortEnvironment = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("mobilevit_model.onnx").readBytes()
            ortSession = ortEnvironment.createSession(modelBytes)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading model: ${e.message}")
        }
    }

    @Composable
    fun DrowsinessDetectionScreen() {
        var isDrowsy by remember { mutableStateOf(false) }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(if (isDrowsy) Color.Red else Color.Green),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            CameraPreview(
                modifier = Modifier
                    .weight(1f)
                    .padding(16.dp)
            ) { imageProxy ->
                processFrame(imageProxy) { drowsy ->
                    isDrowsy = drowsy
                }
            }
            
            Text(
                text = if (isDrowsy) "You are Drowsy" else "You are Active",
                modifier = Modifier.padding(16.dp),
                color = Color.White
            )
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

    private fun processFrame(imageProxy: ImageProxy, onResult: (Boolean) -> Unit) {
        if (isProcessing || frameBuffer.size >= 30) return
        
        val processedFrame = ImageProcessor.preprocessFrame(imageProxy)
        frameBuffer.add(processedFrame)
        
        if (frameBuffer.size == 30) {
            isProcessing = true
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    // Prepare input tensor
                    val shape = longArrayOf(1, 30, 3, 256, 256)
                    val flattenedInput = FloatArray(30 * 3 * 256 * 256)
                    var idx = 0
                    for (frame in frameBuffer) {
                        System.arraycopy(frame, 0, flattenedInput, idx, frame.size)
                        idx += frame.size
                    }

                    val inputTensor = OnnxTensor.createTensor(
                        OrtEnvironment.getEnvironment(),
                        FloatBuffer.wrap(flattenedInput),
                        shape
                    )

                    // Run inference
                    val output = ortSession.run(mapOf("input" to inputTensor))
                    val outputTensor = output[0].value as Array<FloatArray>
                    val prediction = 1.0f / (1.0f + Math.exp(-outputTensor[0][0].toDouble())).toFloat()
                    
                    withContext(Dispatchers.Main) {
                        onResult(prediction >= 0.5f)
                    }
                    
                    frameBuffer.clear()
                    isProcessing = false
                } catch (e: Exception) {
                    Log.e("MainActivity", "Error running model: ${e.message}")
                    frameBuffer.clear()
                    isProcessing = false
                }
            }
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