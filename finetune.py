from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor
from datasets import load_dataset
import torch

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

dataset = load_dataset("nlphuji/flickr30k")

loc = "nlpconnect/vit-gpt2-image-captioning"

tokenizer = AutoTokenizer.from_pretrained(loc, use_fast=True)
image_processor = AutoImageProcessor.from_pretrained(loc, use_fast=True)
# processor = AutoProcessor.from_pretrained(loc, use_fast=True)


def preprocess_function(examples):
    # Process images
    images = [
        image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        for image in examples["image"]
    ]
    # Debugging: print type of pixel_values
    if len(images) > 0:
        print(f"Type of pixel_values: {type(images[0])}")
        print(f"Shape of pixel_values: {images[0].shape}")
        print(f"Dtype of pixel_values: {images[0].dtype}")
    # images = torch.cat(images, dim=0)  # Stack tensors along batch dimension
    image_features = {"pixel_values": images}

    # Process captions - select first caption for each image
    captions = [
        captions[0] for captions in examples["caption"]
    ]  # Take first caption from each list
    text_features = {
        "labels": tokenizer(
            captions, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
    }

    # Combine features
    features = {**image_features, **text_features}
    return features


small_train_dataset = dataset["test"].shuffle(seed=42).select(range(1))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1))

# preprocessed_datasets = dataset.map(preprocess_function, batched=True)
small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
small_eval_dataset = small_eval_dataset.map(preprocess_function, batched=True)

# small_train_dataset = preprocessed_datasets["test"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = preprocessed_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(loc, torch_dtype=torch.float16, low_cpu_mem_usage=True,)
model.gradient_checkpointing_enable=True

# model.config.decoder_start_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size
# model.config.
model.accepts_loss_kwargs = False

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduce from default
    per_device_eval_batch_size=1,   # Reduce from default
    gradient_accumulation_steps=1,   # Accumulate gradients
    # fp16=True,                      # Use mixed precision
    gradient_checkpointing=True,    # Use gradient checkpointing
)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1
)  #


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    # compute_metrics=compute_metrics,
)

trainer.train()
