package com.example.mydrowsinessapp

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

object ImageProcessor {
    fun preprocessFrame(imageProxy: ImageProxy): FloatArray {
        val bitmap = imageProxy.toBitmap()
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        bitmap.recycle()
        
        return normalizeImage(resizedBitmap).also {
            resizedBitmap.recycle()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            width,
            height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        out.close()

        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun normalizeImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(256 * 256)
        bitmap.getPixels(pixels, 0, 256, 0, 0, 256, 256)

        val normalizedPixels = FloatArray(3 * 256 * 256)
        var idx = 0

        for (pixel in pixels) {
            // Extract RGB values
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)

            // Normalize to [0,1] range
            normalizedPixels[idx++] = r / 255f
            normalizedPixels[idx++] = g / 255f
            normalizedPixels[idx++] = b / 255f
        }

        return normalizedPixels
    }
}
