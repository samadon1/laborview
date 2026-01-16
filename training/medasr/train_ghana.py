"""
Fine-tune MedASR on Ghanaian accents only from AfriSpeech-200
Much smaller dataset (~5 hours) - fits in disk limits
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
# ]
# ///

import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoModelForCTC, AutoProcessor, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import evaluate

MODEL_ID = "google/medasr"
DATASET_ID = "intronhealth/afrispeech-200"
OUTPUT_DIR = "./medasr-ghana"
HUB_MODEL_ID = "samwell/medasr-ghana"
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 10
WARMUP_STEPS = 100
SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 20

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt", return_attention_mask=True)
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"]
        return batch

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

def compute_metrics(pred, processor, wer_metric):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def main():
    print("=" * 60)
    print("MedASR Fine-tuning on Ghanaian accents only")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nLoading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    print(f"Parameters: {model.num_parameters():,}")

    print(f"\nLoading {DATASET_ID} (all accents)...")
    dataset = load_dataset(DATASET_ID, "all", trust_remote_code=True)

    # Filter for Ghana only (country == "GH")
    print("\nFiltering for Ghanaian accents (country=GH)...")
    dataset = dataset.filter(lambda x: x["country"] == "GH")

    print(f"Ghana subset - Train: {len(dataset['train'])} | Dev: {len(dataset['dev'])} | Test: {len(dataset['test'])}")

    print("\nPreparing dataset...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=4,
        desc="Processing"
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = evaluate.load("wer")

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
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="end",
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_results = trainer.evaluate(dataset["test"])
    print(f"\nTest WER: {test_results['eval_wer']:.4f}")

    print(f"\nPushing to hub: {HUB_MODEL_ID}")
    trainer.push_to_hub(commit_message=f"Fine-tuned on Ghanaian accents | Test WER: {test_results['eval_wer']:.4f}")
    processor.push_to_hub(HUB_MODEL_ID)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model: https://huggingface.co/{HUB_MODEL_ID}")
    print(f"Test WER: {test_results['eval_wer']:.4f}")

if __name__ == "__main__":
    main()
