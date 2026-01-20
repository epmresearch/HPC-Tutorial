# HPC-Tutorial: Fine-Tuning LLMs on the UCalgary HPC

This repository contains a hands-on tutorial on how to use the University of Calgary
High Performance Computing (HPC) system to fine-tune Large Language Models (LLMs)
(e.g., Llama) using Supervised Fine-Tuning (SFT).

The material is designed for students and researchers who already know basic
Python/ML and want to move their fine-tuning workloads from local machines or
Google Colab to the UCalgary HPC (ARC / Slurm-based cluster).

---

## Contents

- `slides/`
  - Slide decks used in the tutorial sessions (HPC basics, SFT workflow, examples).

- `notebooks/`
  - Example notebooks for:
    - Preparing a QA dataset (e.g., JSONL from a book)
    - Inspecting and verifying data before training

- `scripts/`
  - Shell and Slurm scripts for running fine-tuning jobs on the HPC:
    - `create_env.sh` – create and configure a conda/venv environment
    - `run_finetune.sh` – convenience script to launch training
    - `slurm_finetune_llama.sbatch` – example Slurm submission script

- `configs/`
  - Example configuration files for training (hyperparameters, paths, etc.).

---

## Workflow Overview

1. **Prepare the dataset locally** (or on the HPC) as a JSONL file containing
   instruction/output or question/answer pairs.

2. **Transfer the dataset** to the HPC cluster (e.g., using `scp` or SFTP).

3. **Create a Python environment** on HPC (via `create_env.sh`), install
   required libraries such as `transformers`, `accelerate`, `unsloth`, etc.

4. **Edit the Slurm script** (`slurm_finetune_llama.sbatch`) to select:
   - Account / partition
   - Number of GPUs and CPUs
   - Time limit
   - Paths to your data and output directories

5. **Submit the job** using `sbatch`:
   ```bash
   sbatch scripts/slurm_finetune_llama.sbatch
