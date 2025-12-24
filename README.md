## Dataset

- **Source**: CNN DailyMail dataset (version 3.0.0)
- **Training samples**: 10,000
- **Validation samples**: 100
- **Test samples**: 100

## Model Configuration

- **Base model**: Qwen2.5-0.5B-Instruct
- **Quantization**: 4-bit loading
- **LoRA parameters**:
  - r (rank): 32
  - lora_alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Use RSLoRA: True

## Training Configuration

- **Optimizer**: AdamW 8-bit
- **Learning rate**: 2e-4
- **Scheduler**: Linear
- **Batch size**: 8 per device
- **Gradient accumulation steps**: 4
- **Effective batch size**: 32
- **Training epochs**: 1
- **Warmup steps**: 5
- **Weight decay**: 0.001
- **Training approach**: Response-only (loss computed only on assistant responses)

## Usage

### Running the Notebook

1. Open the notebook in Google Colab
2. Ensure GPU is enabled (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially

### Key Steps

1. **Install dependencies**:
   ```python
   !pip install evaluate unsloth rouge_score
   ```

2. **Load dataset and model**:
   - Dataset: CNN DailyMail via HuggingFace
   - Model: Qwen2.5-0.5B-Instruct via Unsloth

3. **Format data with chat templates**:
   - System prompt: "You are a helpful assistant specialized in summarization."
   - User message: Article text
   - Assistant response: Summary

4. **Fine-tune with LoRA**:
   - Uses SFTTrainer from TRL
   - Response-only training with Unsloth

5. **Evaluate**:
   - Generate summaries on test set
   - Calculate ROUGE scores

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Unsloth
- Datasets
- Evaluate
- TRL (Transformer Reinforcement Learning)
- ROUGE Score
- Google Colab (recommended) or local GPU with 16GB+ VRAM

## Notes

- Training time: ~73 minutes on T4 GPU
- The model uses Qwen's chat template format with special tokens
- Cross-entropy loss is computed only on assistant responses to improve training efficiency
- Model checkpoints are saved every 5 steps with best model selection based on eval_loss# Qwen-0.5-B-for-summarization
A text summarization project that fine-tunes Qwen2.5-0.5B-Instruct on the CNN DailyMail dataset using LoRA for efficient training.

## Overview

This project fine-tunes the Qwen2.5-0.5B-Instruct model for news article summarization. The model is trained on 10,000 examples from the CNN DailyMail dataset and evaluated using ROUGE metrics.

## Results

### Before Fine-tuning
- ROUGE-1: 0.2391
- ROUGE-2: 0.0700
- ROUGE-L: 0.1711
- ROUGE-Lsum: 0.1960
- Average summary length: ~104 words

### After Fine-tuning
- ROUGE-1: 0.2679 (+11.4%)
- ROUGE-2: 0.0815 (+16.4%)
- ROUGE-L: 0.1981 (+15.8%)
- ROUGE-Lsum: 0.2514 (+28.3%)
- Average summary length: ~30 words

The fine-tuned model produces more concise summaries (30 words vs 104 words) while improving all ROUGE scores, indicating better alignment with the reference summaries.
