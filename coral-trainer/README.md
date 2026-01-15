# Coral Trainer

A cross-platform desktop application that automates the pipeline from raw image datasets to Google Coral-ready `.tflite` models.

## Prerequisites

- **Rust**: Latest stable version
- **Node.js**: LTS version
- **Python**: 3.9 (Strictly recommended for `tflite-model-maker`)
- **Google Coral Edge TPU Compiler**: `edgetpu_compiler` installed in PATH (for Linux)

## Setup

1.  **Install Dependencies**
    ```bash
    cd coral-trainer
    npm install
    ```

2.  **Python Environment**
    The application relies on `tflite-model-maker`.
    ```bash
    # Create a virtual environment (Python 3.9 recommended)
    python3.9 -m venv venv
    source venv/bin/activate
    pip install tflite-model-maker
    ```

    *Note: If dependencies are missing, the application runs in "Mock Mode" for demonstration purposes.*

## Development

Run the application in development mode:

```bash
npm run tauri dev
```

This will spawn the Tauri window. The Rust backend is configured to look for `python3` in your PATH and execute `python/train_engine.py`.

## Production & Sidecar Packaging

To package this as a standalone application with the Python environment bundled:

1.  **Compile Python Script**: Use PyInstaller to create a single-file executable.
    ```bash
    pyinstaller --onefile --name train_engine python/train_engine.py
    ```

2.  **Configure Sidecar**:
    - Rename the executable to include the target triple (e.g., `train_engine-x86_64-unknown-linux-gnu`).
    - Place it in `src-tauri/binaries/`.
    - Update `src-tauri/tauri.conf.json`:
      ```json
      "bundle": {
        "externalBin": ["binaries/train_engine"]
      }
      ```
    - Update `src-tauri/src/lib.rs` to use `Command::new_sidecar("train_engine")`.

## Architecture

- **Frontend**: HTML/JS with Tailwind CSS (via CDN for simplicity).
- **Backend (Rust)**: Tauri handles window management and spawning the Python subprocess.
- **ML Engine (Python)**: `train_engine.py` handles the TensorFlow/TFLite Model Maker pipeline, including quantization and compilation.

## Features

- **Project Setup**: configure epochs, batch size, and paths.
- **Real-time Progress**: Visual progress bar and live log streaming from Python.
- **Edge TPU Support**: Auto-compilation using `edgetpu_compiler`.
- **Mock Mode**: Fallback for environments without GPU/TPU tools.
