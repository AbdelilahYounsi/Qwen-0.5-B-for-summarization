# Qwen-0.5-B-for-summarization
A text summarization project that combines BART-large-cnn for initial summarization with Qwen2.5-0.5B fine-tuning for improved performance.
## Overview

This project implements a two-stage summarization approach:
1. **Stage 1**: Use pre-trained BART-large-cnn to generate summaries from CNN DailyMail articles
2. **Stage 2**: Fine-tune Qwen2.5-0.5B on the generated summaries using LoRA techniques

## Features

- Batch processing for efficient summarization
- Memory-optimized training with 4-bit quantization
- LoRA fine-tuning for reduced computational requirements
- ROUGE metric evaluation
- Google Colab compatibility
