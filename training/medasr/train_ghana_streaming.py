"""
Fine-tune MedASR on ALL Ghanaian accents (Twi + Akan + Fante)
Uses Google's official fine-tuning approach from their notebook

WITH DATA AUGMENTATION:
- SpecAugment (frequency & time masking)
- Speed perturbation (0.9x, 1.0x, 1.1x)

Dataset breakdown:
- twi: 1,315 train, 186 val, 58 test
- akan: 131 train, no val, 26 test
- akan-fante: 230 train, 33 val, 32 test
Total: ~1,676 train -> ~5,028 with speed perturbation
"""

# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5",
#   "datasets==2.20.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "librosa==0.10.0",
#   "soundfile>=0.12.1",
#   "accelerate>=0.24.0",
#   "tensorboard>=2.14.0",
#   "audiomentations>=0.30.0",
# ]
# ///

import os
import random
import multiprocessing
import torch
import torchaudio
from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import AutoModelForCTC, AutoProcessor, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import numpy as np
import evaluate
import librosa

MODEL_ID = "google/medasr"
DATASET_ID = "intronhealth/afrispeech-200"
OUTPUT_DIR = "./medasr-ghana"
HUB_MODEL_ID = "samwell/medasr-ghana"
SAMPLING_RATE = 16000

# Training hyperparameters (optimized for small dataset with augmentation)
LEARNING_RATE = 1e-5  # Lower LR for more stable training
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32
NUM_EPOCHS = 15  # More epochs since we have augmented data
WARMUP_STEPS = 500  # More warmup for stability

# SpecAugment parameters
FREQ_MASK_PARAM = 27  # Max frequency mask width
TIME_MASK_PARAM = 100  # Max time mask width
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# Speed perturbation factors
SPEED_FACTORS = [0.9, 1.0, 1.1]  # 3x data multiplication


@dataclass
class DataCollator:
    """Google's official DataCollator - extracts features and pads within a batch."""
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        batch["labels"] = labels_batch["input_ids"]
        return batch


def apply_spec_augment(input_features, freq_mask_param, time_mask_param, num_freq_masks, num_time_masks):
    """Apply SpecAugment to mel spectrogram features."""
    features = input_features.copy()
    num_mel_bins, time_steps = features.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, num_mel_bins - f))
        features[f0:f0 + f, :] = 0

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, time_steps // 4))  # Limit to 25% of time
        t0 = random.randint(0, max(0, time_steps - t))
        features[:, t0:t0 + t] = 0

    return features


def speed_perturb(audio_array, sr, factor):
    """Apply speed perturbation to audio."""
    if factor == 1.0:
        return audio_array
    # Use librosa to change speed without changing pitch
    return librosa.effects.time_stretch(audio_array, rate=factor)


def create_augmented_dataset(dataset, speed_factors):
    """Create augmented dataset with speed perturbation."""
    augmented_data = []

    for example in dataset:
        audio_array = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]

        for factor in speed_factors:
            new_example = {k: v for k, v in example.items() if k != "audio"}

            if factor == 1.0:
                new_audio = audio_array
            else:
                new_audio = speed_perturb(audio_array, sr, factor)

            new_example["audio"] = {
                "array": new_audio,
                "sampling_rate": sr,
                "path": example["audio"].get("path", "")
            }
            augmented_data.append(new_example)

    return Dataset.from_list(augmented_data)


def main():
    print("=" * 60)
    print("MedASR Fine-tuning on ALL Ghanaian accents (Twi + Akan + Fante)")
    print("Using Google's official fine-tuning approach")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model and processor
    print(f"\nLoading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    print(f"Parameters: {model.num_parameters():,}")

    # Load ALL Ghanaian accent configs: twi, akan, akan-fante
    print(f"\nLoading {DATASET_ID} - ALL Ghanaian accents...")

    # Load twi config (has all 3 splits)
    print("Loading twi config...")
    twi_train = load_dataset(DATASET_ID, "twi", split="train", trust_remote_code=True)
    twi_dev = load_dataset(DATASET_ID, "twi", split="validation", trust_remote_code=True)
    twi_test = load_dataset(DATASET_ID, "twi", split="test", trust_remote_code=True)
    print(f"  twi: train={len(twi_train)}, val={len(twi_dev)}, test={len(twi_test)}")

    # Load akan config (train + test only, no validation)
    print("Loading akan config...")
    akan_train = load_dataset(DATASET_ID, "akan", split="train", trust_remote_code=True)
    akan_test = load_dataset(DATASET_ID, "akan", split="test", trust_remote_code=True)
    print(f"  akan: train={len(akan_train)}, test={len(akan_test)}")

    # Load akan-fante config (has all 3 splits)
    print("Loading akan-fante config...")
    fante_train = load_dataset(DATASET_ID, "akan-fante", split="train", trust_remote_code=True)
    fante_dev = load_dataset(DATASET_ID, "akan-fante", split="validation", trust_remote_code=True)
    fante_test = load_dataset(DATASET_ID, "akan-fante", split="test", trust_remote_code=True)
    print(f"  akan-fante: train={len(fante_train)}, val={len(fante_dev)}, test={len(fante_test)}")

    # Concatenate all datasets
    print("\nConcatenating datasets...")
    train_dataset_raw = concatenate_datasets([twi_train, akan_train, fante_train])
    dev_dataset = concatenate_datasets([twi_dev, fante_dev])  # akan has no validation
    test_dataset = concatenate_datasets([twi_test, akan_test, fante_test])
    print(f"Raw: train={len(train_dataset_raw)}, val={len(dev_dataset)}, test={len(test_dataset)}")

    # Cast audio column to correct sampling rate
    print("\nCasting audio to 16kHz...")
    train_dataset_raw = train_dataset_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Apply speed perturbation to training data (3x augmentation)
    print(f"\nApplying speed perturbation with factors: {SPEED_FACTORS}...")
    train_dataset = create_augmented_dataset(train_dataset_raw, SPEED_FACTORS)
    print(f"Augmented training set: {len(train_dataset)} samples (3x original)")

    # Prepare dataset function with SpecAugment for training
    def prepare_dataset_train(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=SAMPLING_RATE
        ).input_features[0]
        # Apply SpecAugment to training data
        batch["input_features"] = apply_spec_augment(
            input_features,
            FREQ_MASK_PARAM,
            TIME_MASK_PARAM,
            NUM_FREQ_MASKS,
            NUM_TIME_MASKS
        )
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    # Prepare dataset function WITHOUT augmentation for eval/test
    def prepare_dataset_eval(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=SAMPLING_RATE
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    # Process datasets
    print("\nProcessing datasets (with SpecAugment on train)...")
    train_dataset = train_dataset.map(prepare_dataset_train, remove_columns=train_dataset.column_names)
    dev_dataset = dev_dataset.map(prepare_dataset_eval, remove_columns=dev_dataset.column_names)
    test_dataset = test_dataset.map(prepare_dataset_eval, remove_columns=test_dataset.column_names)
    print("Dataset processing complete!")

    # Create data collator
    data_collator = DataCollator(processor=processor, padding=True)

    # Load WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments (from Google's notebook)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=10,
        logging_first_step=True,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="end",
        report_to=["tensorboard"],
    )

    # Create Trainer (using Google's exact approach with processing_class)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest WER: {test_results['eval_wer']:.4f}")

    print(f"\nPushing to hub: {HUB_MODEL_ID}")
    trainer.push_to_hub(commit_message=f"Fine-tuned on ALL Ghanaian accents with augmentation (SpecAugment + SpeedPerturb) | Test WER: {test_results['eval_wer']:.4f}")
    processor.push_to_hub(HUB_MODEL_ID)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model: https://huggingface.co/{HUB_MODEL_ID}")
    print(f"Test WER: {test_results['eval_wer']:.4f}")


if __name__ == "__main__":
    main()
