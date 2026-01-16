# MedASR Fine-tuning on AfriSpeech-200

Fine-tuning Google's MedASR medical speech recognition model on AfriSpeech-200, a Pan-African accented speech dataset.

## Overview

- **Model**: [google/medasr](https://hf.co/google/medasr) - Medical ASR model based on Conformer architecture
- **Dataset**: [intronhealth/afrispeech-200](https://hf.co/datasets/intronhealth/afrispeech-200) - 200hrs of Pan-African speech with 120 accents from 13 countries
- **Goal**: Improve MedASR's performance on African-accented English for clinical and general domain speech

## Prerequisites

1. **Request access to MedASR**: Visit https://hf.co/google/medasr and accept the terms of use
2. **Hugging Face authentication**:
   ```bash
   huggingface-cli login
   ```

## Installation

```bash
pip install -r requirements.txt
```

Or install the latest transformers from source (recommended for MedASR):
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install datasets evaluate jiwer librosa soundfile tensorboard accelerate
```

## Dataset Details

AfriSpeech-200 contains:
- 200 hours of audio
- 120 African accents from 13 countries
- 2,463 unique speakers
- Clinical and general domain speech
- Train/Dev/Test splits

Countries: Nigeria, Kenya, South Africa, Ghana, Botswana, Uganda, Rwanda, US, Turkey, Zimbabwe, Malawi, Tanzania, Lesotho

## Training Configuration

The training script uses the following default hyperparameters:

- Learning rate: 3e-5
- Batch size: 8 (per device)
- Gradient accumulation: 4 steps
- Epochs: 10
- Early stopping patience: 3
- FP16 training (if GPU available)
- Evaluation metric: Word Error Rate (WER)

## Usage

### Basic Training

```bash
python finetune_medasr.py
```

### Monitor Training

Training logs are saved to TensorBoard:
```bash
tensorboard --logdir=./medasr-afrispeech-200/runs
```

### Expected Output

The script will:
1. Load MedASR model and processor
2. Download and prepare AfriSpeech-200 dataset (~120GB)
3. Resample audio to 16kHz
4. Train for up to 10 epochs with early stopping
5. Evaluate on test set
6. Save final model to `./medasr-afrispeech-200/final/`

### Inference with Fine-tuned Model

```python
from transformers import AutoModelForCTC, AutoProcessor
import librosa

# Load fine-tuned model
model_path = "./medasr-afrispeech-200/final"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCTC.from_pretrained(model_path)

# Load and process audio
audio, sr = librosa.load("your_audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# Generate transcription
outputs = model.generate(**inputs)
transcription = processor.batch_decode(outputs)[0]
print(transcription)
```

## Model Architecture

MedASR is based on:
- **Architecture**: Conformer (105M parameters)
- **Training objective**: CTC (Connectionist Temporal Classification)
- **Pre-training**: 5000+ hours of medical dictation and physician-patient dialogue
- **Input**: 16kHz mono audio
- **Output**: English text with medical terminology

## Performance Metrics

The model is evaluated using:
- **Word Error Rate (WER)**: Primary metric
- Computed on validation set during training
- Final evaluation on held-out test set

## Hardware Requirements

- **Minimum**: 16GB RAM, 10GB disk space
- **Recommended**: GPU with 16GB+ VRAM (e.g., V100, A100)
- **Storage**: ~120GB for dataset, ~2GB for model

## Troubleshooting

### Out of Memory
Reduce batch size or gradient accumulation steps in the script

### Slow Download
Use streaming mode or download specific accent subsets

### Access Denied
Ensure you've requested and been granted access to google/medasr

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{wu2023last,
  title={Last: Scalable Lattice-Based Speech Modelling in Jax},
  author={Wu, Ke and Variani, Ehsan and Bagby, Tom and Riley, Michael},
  booktitle={ICASSP 2023},
  year={2023}
}

@article{afrispeech2023,
  title={AfriSpeech-200: Pan-African accented speech dataset for clinical and general domain ASR},
  author={Intron Health},
  year={2023}
}
```

## License

- MedASR: [Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms)
- AfriSpeech-200: CC-BY-NC-SA-4.0

## Resources

- [MedASR Documentation](https://developers.google.com/health-ai-developer-foundations/medasr)
- [MedASR GitHub](https://github.com/google-health/medasr)
- [AfriSpeech Paper](https://arxiv.org/abs/2310.00274)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
