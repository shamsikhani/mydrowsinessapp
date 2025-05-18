# MyDrowsinessApp - Real-time Drowsiness Detection

MyDrowsinessApp is an Android application designed to detect driver/user drowsiness in real-time using the device's front-facing camera and a machine learning model. The app provides visual feedback to the user based on the detected drowsiness state.

## Features

*   **Real-time Drowsiness Detection:** Continuously monitors the user via the front camera.
*   **Machine Learning Powered:** Utilizes a MobileViT (Vision Transformer) model in ONNX format for accurate detection.
*   **Visual Feedback:** Changes the screen's background color (Green for Active, Red for Drowsy) and displays a status message.
*   **Built with Modern Android Technologies:** Leverages Kotlin, Jetpack Compose, and CameraX.

## Technical Stack

*   **Programming Language:** Kotlin
*   **UI Toolkit:** Jetpack Compose
*   **Camera:** CameraX
*   **Machine Learning:**
    *   ONNX Runtime for Android
    *   Pre-trained Model: `mobilevit_model.onnx` (expected in `app/src/main/assets/`)
*   **Asynchronous Programming:** Kotlin Coroutines
*   **Build System:** Gradle

## Project Structure

Key components of the application include:

*   `app/src/main/java/com/example/mydrowsinessapp/`
    *   `MainActivity.kt`: The main entry point of the app. Handles camera setup, ONNX model loading, UI updates, and orchestrates the drowsiness detection pipeline.
    *   `ImageProcessor.kt`: Contains logic for preprocessing camera frames before they are fed into the ML model. This includes conversion to Bitmap, resizing, and pixel normalization.
*   `app/src/main/assets/`
    *   `mobilevit_model.onnx`: The machine learning model file.
*   `app/src/main/AndroidManifest.xml`: Declares app components, permissions (e.g., Camera), and features.
*   `app/build.gradle.kts`: Specifies project dependencies, including CameraX, ONNX Runtime, and Jetpack Compose libraries.

## How It Works

1.  **Camera Initialization:** The app requests camera permission and initializes the front-facing camera using CameraX.
2.  **Frame Capture:** Video frames are continuously captured from the camera.
3.  **Image Preprocessing (`ImageProcessor.kt`):**
    *   Each frame (ImageProxy) is converted to an RGB Bitmap.
    *   The Bitmap is resized to 256x256 pixels.
    *   Pixel values (R, G, B) are normalized to a floating-point range of [0.0, 1.0].
4.  **Frame Buffering (`MainActivity.kt`):**
    *   The app collects a sequence of 30 preprocessed frames.
5.  **ONNX Model Inference (`MainActivity.kt`):**
    *   The `mobilevit_model.onnx` is loaded using ONNX Runtime.
    *   The buffered sequence of 30 frames (input tensor shape: `1x30x3x256x256`) is fed to the model.
6.  **Drowsiness Prediction:**
    *   The model outputs a prediction score.
    *   This score is processed (passed through a sigmoid-like function), and if it exceeds a threshold (0.5), the user is classified as drowsy.
7.  **UI Feedback (`MainActivity.kt`):**
    *   The screen background changes: Green for "Active," Red for "Drowsy."
    *   A text message ("You are Active" or "You are Drowsy") is displayed.

## Setup and Installation

To build and run this project:

1.  **Clone the repository (or ensure you have the code):**
    ```bash
    git clone https://github.com/shamsikhani/mydrowsinessapp.git
    cd mydrowsinessapp
    ```
2.  **Ensure `mobilevit_model.onnx` is present:**
    *   Place your `mobilevit_model.onnx` file in the `app/src/main/assets/` directory. If this directory doesn't exist, create it.
3.  **Open in Android Studio:**
    *   Open Android Studio (latest stable version recommended).
    *   Select "Open an existing Android Studio project."
    *   Navigate to the cloned `mydrowsinessapp` directory and select it.
4.  **Gradle Sync:**
    *   Allow Android Studio to perform a Gradle sync. This will download all necessary dependencies.
5.  **Run the App:**
    *   Connect an Android device (with a front-facing camera) or use an Android Emulator that supports camera passthrough.
    *   Select the device/emulator from the target devices list.
    *   Click the "Run" button (green play icon) in Android Studio.
    *   Grant camera permission when prompted by the app.

## Permissions Required

*   **`android.permission.CAMERA`**: To access the device's camera.

## Model Details

*   **Model Format:** ONNX (`.onnx`)
*   **Expected Name:** `mobilevit_model.onnx`
*   **Expected Location:** `app/src/main/assets/`
*   **Input Shape:** `[1, 30, 3, 256, 256]` (Batch size of 1, 30 frames, 3 color channels, 256x256 pixels per frame)
*   **Output:** A single floating-point value, interpreted as a drowsiness score.

## Potential Improvements / Future Work

*   **Audio Alerts:** Add sound alerts for drowsiness detection.
*   **Sensitivity Adjustment:** Allow users to adjust the drowsiness detection sensitivity.
*   **More Sophisticated Preprocessing:** Explore advanced image preprocessing techniques (e.g., face detection, eye aspect ratio calculation) before feeding data to the model, which might improve accuracy or allow for simpler models.
*   **Model Optimization:** Further optimize the ONNX model for better performance on mobile devices.
*   **Logging/History:** Implement a feature to log drowsiness events for later review.
*   **Background Operation:** (With careful consideration for battery life and user privacy) Explore options for background monitoring if appropriate for the use case.
*   **User Authentication and Profiles:** For personalized settings or history.

---

Feel free to contribute to this project by submitting issues or pull requests!
