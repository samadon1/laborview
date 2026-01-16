"""
Fine-tune MedASR on AfriSpeech-200 Dataset

This script fine-tunes the Google MedASR model on the AfriSpeech-200 dataset,
which contains 200 hours of Pan-African accented speech for clinical and general domain ASR.

Requirements:
- Access to google/medasr (gated model - request at https://hf.co/google/medasr)
- HuggingFace authentication with write access
"""

import os
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import evaluate

# Configuration
MODEL_ID = "google/medasr"
DATASET_ID = "intronhealth/afrispeech-200"
OUTPUT_DIR = "./medasr-afrispeech-200"
CACHE_DIR = "./cache"

# Training hyperparameters
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
WARMUP_STEPS = 500
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100
EARLY_STOPPING_PATIENCE = 3


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC models that pads inputs and labels
    """
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def prepare_dataset(batch, processor):
    """
    Prepare audio and text for training
    """
    # Load audio
    audio = batch["audio"]

    # Process audio to input values
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    # Process text to labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids

    return batch


def compute_metrics(pred, processor, wer_metric):
    """
    Compute WER metric for evaluation
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    """
    Main training function
    """
    print("=" * 50)
    print("Fine-tuning MedASR on AfriSpeech-200")
    print("=" * 50)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load processor and model
    print(f"\nLoading model and processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCTC.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    print(f"Model loaded. Total parameters: {model.num_parameters():,}")

    # Load dataset
    print(f"\nLoading dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, "all", cache_dir=CACHE_DIR)

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['dev'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Resample audio to 16kHz (MedASR requirement)
    print("\nResampling audio to 16kHz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare dataset
    print("\nPreparing dataset for training...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=4,
        desc="Processing audio and text"
    )

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize WER metric
    wer_metric = evaluate.load("wer")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    print("=" * 50)
    test_results = trainer.evaluate(dataset["test"])
    print(f"\nTest WER: {test_results['eval_wer']:.4f}")

    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}/final...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    processor.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Model saved to: {OUTPUT_DIR}/final")
    print(f"Test WER: {test_results['eval_wer']:.4f}")


if __name__ == "__main__":
    main()
