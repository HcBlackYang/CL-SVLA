# Metacognitive Feedback Module (MFM) for Vision-Language-Action Models

This project presents a novel framework, the **Metacognitive Feedback Module (MFM)**, designed to enhance the robustness and recovery capabilities of Vision-Language-Action (VLA) models in robotic manipulation tasks. The MFM actively monitors the agent's execution, detects failures in real-time, and provides corrective feedback to guide the agent back to a successful trajectory.

The core of this repository is a two-phase approach:
1.  **Phase 1: MFM-Supervised Execution**: A standard fine-tuned VLA (OpenVLA) performs the task while being monitored by the MFM. The MFM uses a rule-based system to detect common failures, such as grasping errors.
2.  **Phase 2: End-to-End Corrective Action**: Upon failure detection, the system transitions to a powerful, end-to-end model that leverages a Chain-of-Thought (CoT) large vision-language model (Qwen-VL) to analyze the failure context from a dual-view perspective and generate corrective instructions that are directly injected into the VLA to guide its recovery.

This repository provides the complete pipeline, including failure data collection, expert demonstration generation, a novel end-to-end model architecture, training scripts, and hybrid evaluation scripts.

## 🌟 Key Features

*   **Metacognitive Feedback Module (MFM)**: A lightweight, rule-based system for real-time failure detection (`GraspFailure`, etc.).
*   **Failure Snapshot & Recovery**: Automatically captures the simulation state upon failure, enabling targeted recovery and expert data generation.
*   **Expert Data Generation Pipeline**: Scripts to load failure snapshots and generate expert recovery trajectories using a PD controller for multiple tasks.
*   **Command-Injecting VLA Architecture**: A novel end-to-end model where a CoT model (Qwen-VL) analyzes dual-camera views to generate an "instruction vector" that is injected directly into the VLA's (OpenVLA) embedding space to guide its actions.
*   **Memory-Efficient Decoupled Execution**: The evaluation script uses a sophisticated memory management strategy, loading only one large model (either CoT or VLA) onto the GPU at a time to run on systems with limited VRAM.
*   **Hybrid Evaluation Framework**: A comprehensive evaluation script that compares the performance of the baseline MFM-supervised VLA (Phase 1) against the advanced end-to-end recovery model (Phase 2).
*   **Academic-Style Visualization**: The training script automatically generates high-quality plots for loss convergence, performance metrics, and training summaries, suitable for research papers.

## 📂 Repository Structure

```
MFM/
├── config/                 # Configuration files for MFM and models.
├── core/                   # Core components of the MFM.
│   ├── mfm_core.py         # Main MFM controller for failure capture.
│   ├── mfm_core_eval.py    # MFM controller for hybrid evaluation.
│   ├── state_collector.py  # Collects multi-modal state information.
│   ├── state_summarizer.py # Creates minimal state summaries for analysis.
│   └── task_success_detector.py # Rule-based failure detection.
├── data/                     # Directory for datasets, checkpoints, and logs.
├── experiments/              # Main executable scripts.
│   ├── dark_mfm.py         # Script to run VLA and capture failure snapshots.
│   ├── generate_expert_data.py # Script to generate expert demos from snapshots.
│   ├── eval_new.py         # The main hybrid evaluation script (Phase 1 + Phase 2).
│   └── run.py              # A simple script to run other experiments sequentially.
├── models/                   # Model architecture definitions.
│   ├── improved_unified_model.py # The command-injecting model for training.
│   └── eval_unified_model.py # The model adapted for memory-efficient evaluation.
├── scripts/                  # Scripts for data preparation and training.
│   ├── improved_prepare_dataset.py # Processes expert data into HDF5 format.
│   └── improved_train.py   # The training script for the end-to-end model.
└── utils/                    # Utility functions (e.g., logging).
```

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   NVIDIA GPU with CUDA 11.8+ (for `torch` and `bitsandbytes`)
*   Access to pre-trained model weights for OpenVLA and Qwen-VL.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up the environment and install dependencies:**
    It is highly recommended to use a virtual environment (e.g., Conda or venv).
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install `libero` and `openvla`:**
    This project relies on the `libero` benchmark and `openvla`. Please follow their official installation instructions, as they typically require installation from the source.
    ```bash
    # Example for editable installs
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    pip install -e LIBERO/

    git clone https://github.com/openvla/openvla.git
    pip install -e openvla/
    ```

4.  **Download Model Weights:**
    Download the pre-trained weights for **OpenVLA** and **Qwen-VL** and place them in the specified directories within the codebase (e.g., `/root/autodl-tmp/openvla/weights/`). Update the paths in the scripts (`dark_mfm.py`, `improved_train.py`, etc.) if you place them elsewhere.

## ⚙️ Workflow and Usage

The project follows a three-step workflow: **Data Collection -> Training -> Evaluation**.

### Step 1: Failure Data Collection

This step runs the VLA in a challenging environment to make it fail. The MFM detects these failures and saves a "snapshot" of the exact simulation state at the moment of failure.

**Command:**
```bash
python MFM/experiments/dark_mfm.py
```
*   **What it does:** Runs the VLA agent on a suite of tasks. The `DarkenedEnvironmentWrapper` can be configured to make the task harder (e.g., by reducing lighting).
*   **Output:** Creates a timestamped results directory inside `MFM/data/logs/`. This directory will contain:
    *   `snapshots/`: `.pkl` files, each containing a complete MuJoCo simulation state at the point of a detected failure.
    *   `videos/`: Videos of the failed attempts.
    *   `state_history/`: Detailed JSON logs of the agent's state history.

### Step 2: Expert Demonstration Generation

Using the snapshots from Step 1, this script loads the simulation at the point of failure and executes a hard-coded "expert" policy to successfully complete the task. This generates the ground-truth data needed for training our recovery model.

**Command:**
```bash
python MFM/experiments/generate_expert_data.py \
  --input-path /path/to/your/logs/MFM-FAILURE-CAPTURE.../snapshots \
  --task 1 
```
*   **`--input-path`**: Path to the `snapshots` directory generated in Step 1.
*   **`--task`**: Specify the task ID (e.g., 1-10) for which the expert logic should be applied. This is crucial as different tasks may require different expert trajectories.
*   **What it does:** For each `.pkl` snapshot, it restores the environment, runs the expert controller, and records the dual-view video and state-action pairs.
*   **Output:** Creates an `expert_data/` directory within the same log folder, containing `_EXPERT.json` and `_EXPERT.mp4` files for each processed snapshot.

### Step 3: Preparing the HDF5 Dataset

This script consolidates all the generated expert data into a single, efficient HDF5 file, which is required for training.

**Command:**
```bash
python MFM/scripts/improved_prepare_dataset.py \
  --expert-data-dirs /path/to/task1/expert_data /path/to/task2/expert_data ... \
  --output-dir MFM/data/hdf5_datasets \
  --output-filename spatial_dataset.hdf5
```
*   **`--expert-data-dirs`**: Provide one or more paths to the `expert_data` directories generated in Step 2.
*   **What it does:** Parses the JSON and video files, extracts dual-view images, observations, expert actions, and generates consistent CoT instructions. It then saves everything into a single `.hdf5` file.

### Step 4: Training the End-to-End Model

This script trains the `TrueEndToEndDualViewVLA` model, which learns to map a failure context (from dual-view images and prompts) to a corrective action.

**Command:**
```bash
python MFM/scripts/improved_train.py \
  --dataset-path MFM/data/hdf5_datasets/spatial_dataset.hdf5 \
  --checkpoint-dir MFM/data/spatial_checkpoints \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-epochs 40 \
  --finetune-vla-lora \
  --save-interval 5
```
*   **What it does:** Trains the command-injecting VLA model using LoRA for both Qwen-VL and OpenVLA. It saves checkpoints periodically and generates academic-style plots of the training progress.
*   **Output:** Saves model checkpoints and training plots to the directory specified by `--checkpoint-dir`.

### Step 5: Hybrid Evaluation

This is the final step, where we evaluate the entire system. It runs the two-phase protocol: MFM-supervised VLA (Phase 1) and, upon failure, the full end-to-end model for recovery (Phase 2).

**Command:**
```bash
python MFM/experiments/eval_new.py
```
*   **Configuration**: Before running, ensure the paths to the base models and the fine-tuned checkpoint directory are correctly set inside `eval_new.py`.
*   **What it does:**
    1.  Runs the baseline fine-tuned VLA agent.
    2.  The MFM monitors for failures.
    3.  If a failure is detected, it saves the state, and then the script initiates **Phase 2**.
    4.  In Phase 2, the end-to-end model is loaded. It uses its CoT component to analyze the failure and then its VLA component (guided by the CoT analysis) to attempt recovery.
*   **Output:** A comprehensive `final_report.json` in a new log directory, with detailed statistics comparing the success rates of "VLA-only" episodes vs. "MFM-E2E" recovery episodes. It also saves videos for every attempt, clearly labeled by outcome and mode.

## 📜 Citation

If you use this work in your research, please consider citing it. (Citation details to be added).