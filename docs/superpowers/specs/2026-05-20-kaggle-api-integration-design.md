# Design Spec: Kaggle API Integration Runner

**Date**: 2026-05-20  
**Author**: Antigravity Agent  
**Status**: Draft  

---

## 1. Goal & Context
The user runs all machine learning training jobs (U-Net, GANs on RICE/SEN12MS datasets) on Kaggle GPU instances. Currently, managing these kernels involves manually pushing code, checking status, and downloading logs/outputs.
This tool provides a Python-based command-line interface (`kaggle_runner.py`) using the official Kaggle Python SDK to automate these steps.

## 2. Requirements & Features
- **Authentication**: Authenticate using the system's `KAGGLE_API_TOKEN` environment variable.
- **Commands**:
  1. `push <folder_path>`: Push notebook and metadata from the local folder to Kaggle to trigger a run.
  2. `status <folder_path>`: Query the API and print the current state (`QUEUED`, `RUNNING`, `COMPLETE`, `ERROR`).
  3. `download <folder_path> [--output-dir <path>]`: Download output files and log files. Automatically output log file content to the console.
  4. `watch <folder_path> [--poll-interval <sec>]`: Start a loop that runs `push`, polls `status` every interval, prints execution logs/time, and downloads outputs upon completion.

## 3. Architecture & Data Flow
- **Language**: Python 3.
- **Dependencies**: `kaggle` python package.
- **Directory Structure**:
  - Pushed folder must contain `<notebook>.ipynb` and `kernel-metadata.json`.
  - Output files are saved in `<folder_path>_output` by default unless overridden.

## 4. Error Handling
- Validate presence of API credentials.
- Handle missing files or JSON errors in `kernel-metadata.json`.
- Retry API requests on temporary network timeouts.
