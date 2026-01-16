---
license: apache-2.0
base_model: google/medasr
tags:
- automatic-speech-recognition
- asr
- speech-to-text
- medical-asr
- african-accents
- ghanaian-english
- twi
- akan
- fante
- transformers
- ctc
datasets:
- intronhealth/afrispeech-200
language:
- en
metrics:
- wer
pipeline_tag: automatic-speech-recognition
library_name: transformers
---

# MedASR-Ghana: Medical ASR for Ghanaian-Accented English

This model is a fine-tuned version of [Google's MedASR](https://huggingface.co/google/medasr) optimized for **Ghanaian-accented English** speech recognition, particularly suited for clinical and medical transcription in Ghana.

## Model Description

MedASR-Ghana is designed to transcribe English speech from speakers with Ghanaian accents, including Twi, Akan, and Fante language backgrounds. It builds on Google's MedASR foundation (a 105M parameter Conformer-based CTC model) and adapts it specifically for West African English pronunciation patterns.

### Key Features

- **Optimized for Ghanaian accents**: Trained on Twi, Akan, and Fante accented English
- **Medical domain ready**: Inherits MedASR's medical vocabulary capabilities
- **Lightweight**: 105M parameters - efficient for deployment
- **CTC-based**: Simple greedy decoding, no language model required

## Performance

| Metric | Score |
|--------|-------|
| **Test WER** | **37.53%** |
| Validation WER | 44.56% |

### Training Progress

The model was trained for 120 epochs, with WER improving steadily:

| Epochs | Test WER |
|--------|----------|
| 10 | 55.26% |
| 40 | 40.21% |
| 80 | 38.00% |
| 120 | **37.53%** |

## Training Data

Fine-tuned on the [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) dataset, using all Ghanaian accent configurations:

| Accent | Train | Validation | Test |
|--------|-------|------------|------|
| Twi | 1,315 | 186 | 58 |
| Akan | 131 | - | 26 |
| Akan-Fante | 230 | 33 | 32 |
| **Total** | **1,676** | **219** | **116** |

**Total audio**: ~5.16 hours of Ghanaian-accented English speech

## Usage

### Basic Usage

```python
from transformers import AutoProcessor, AutoModelForCTC
import torch
import librosa

# Load model and processor
model_id = "samwell/medasr-ghana"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)

# Load and preprocess audio
audio, sr = librosa.load("your_audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Transcribe
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

print(transcription)
```

### With Hugging Face Pipeline

```python
from transformers import pipeline

transcriber = pipeline(
    "automatic-speech-recognition",
    model="samwell/medasr-ghana"
)

result = transcriber("your_audio.wav")
print(result["text"])
```

## Training Procedure

### Hyperparameters

- **Learning rate**: 3e-5
- **Batch size**: 8 (with gradient accumulation of 4 = effective batch size 32)
- **Epochs**: 120
- **Warmup steps**: 300
- **Optimizer**: AdamW
- **Precision**: BF16
- **Hardware**: NVIDIA L4 GPU (24GB)

### Training Configuration

```python
TrainingArguments(
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=120,
    warmup_steps=300,
    bf16=True,
    group_by_length=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
)
```

## Intended Use

### Primary Use Cases

- **Clinical transcription** in Ghanaian healthcare settings
- **Medical dictation** for doctors and nurses with Ghanaian accents
- **Healthcare documentation** automation in Ghana
- **Telemedicine** applications serving Ghanaian patients

### Out of Scope

- Non-English transcription (this model is English-only)
- Accents significantly different from West African English
- Real-time streaming (model is optimized for batch processing)

## Limitations

- **Limited training data**: Only ~5 hours of Ghanaian audio
- **WER of 37.53%**: May require post-processing or language model for production use
- **Domain bias**: Best performance on clinical/medical content
- **Accent coverage**: Primarily Twi, Akan, and Fante - may perform differently on other Ghanaian accents

## Ethical Considerations

- This model should be used to **assist** healthcare professionals, not replace clinical judgment
- Transcription errors in medical contexts can have serious consequences - always verify critical information
- The model inherits biases from its training data and base model

## Citation

If you use this model, please cite:

```bibtex
@misc{medasr-ghana,
  title={MedASR-Ghana: Medical ASR for Ghanaian-Accented English},
  author={samwell},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/samwell/medasr-ghana}
}
```

### Related Work

```bibtex
@article{afrispeech2023,
  title={AfriSpeech-200: Pan-African Accented Speech Dataset for Clinical and General Domain ASR},
  author={Olatunji, Tobi and others},
  journal={arXiv preprint arXiv:2310.00274},
  year={2023}
}

@article{medasr2024,
  title={MedASR: Medical Automatic Speech Recognition},
  author={Google Health AI},
  year={2024}
}
```

## Model Card Contact

For questions or feedback, please open an issue on the [model repository](https://huggingface.co/samwell/medasr-ghana).
