package com.example.mydrowsinessapp

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import android.media.Image
import java.nio.ByteBuffer

object ImageProcessor {
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    const val TARGET_SIZE = 192

    fun preprocessFrame(imageProxy: ImageProxy): FloatArray {
        val bitmap = imageProxy.toBitmap()
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, TARGET_SIZE, TARGET_SIZE, true)
        bitmap.recycle()
        
        return normalizeImage(resizedBitmap).also {
            resizedBitmap.recycle()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val vuBuffer = planes[2].buffer // V
        
        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()
        
        val nv21 = ByteArray(ySize + vuSize)
        
        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)
        
        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun normalizeImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(TARGET_SIZE * TARGET_SIZE)
        bitmap.getPixels(pixels, 0, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE)

        val normalizedPixels = FloatArray(3 * TARGET_SIZE * TARGET_SIZE)
        var idx = 0

        for (pixel in pixels) {
            // Extract RGB values
            val r = (pixel shr 16 and 0xFF) / 255f
            val g = (pixel shr 8 and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            // Apply ImageNet normalization
            normalizedPixels[idx++] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            normalizedPixels[idx++] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            normalizedPixels[idx++] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }

        return normalizedPixels
    }
}
