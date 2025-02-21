import torch
import numpy as np
from transformers import (
    VisionEncoderDecoderModel,
    AutoModelForVision2Seq,
    ViTImageProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoProcessor,
    logging,
)
from datasets import load_dataset
import evaluate
import PIL
import torchvision


# logging.set_verbosity_debug()

# 1. Load the dataset (single split with a 'split' column)
raw_dataset = load_dataset(
    "nlphuji/flickr30k", split="test", keep_in_memory=True, num_proc=10
)  # [:1%]

# Filter the raw dataset into train, val, and test splits based on the "split" column.
train_dataset = raw_dataset.filter(
    lambda x: x["split"] == "train", num_proc=10, keep_in_memory=True
)  # .select(range(1))
val_dataset = raw_dataset.filter(
    lambda x: x["split"] == "val", num_proc=10, keep_in_memory=True
)  # .select(range(1))
test_dataset = raw_dataset.filter(
    lambda x: x["split"] == "test",
    num_proc=10,
    keep_in_memory=False,
)  # .select(range(1))

loc = "./vit-gpt2-finetuned-flickr30k/checkpoint-5000"

# 2. Load the pre-trained model and associated processors
model = AutoModelForVision2Seq.from_pretrained(
    loc#, torch_dtype="float16"
)
feature_extractor = AutoImageProcessor.from_pretrained(
    loc,
    use_fast=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    loc,
    use_fast=True,
)
# 2. Load the pre-trained model and associated processors
# image_encoder_model = "google/vit-base-patch16-224-in21k"
# text_decode_model = "gpt2"

# feature_extractor = AutoImageProcessor.from_pretrained(
#     image_encoder_model, use_fast=True
# )
# # text tokenizer
# tokenizer = AutoTokenizer.from_pretrained(text_decode_model, use_fast=True)

# model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
#     image_encoder_model, text_decode_model, pad_token_id=tokenizer.eos_token_id
# )
# image feature extractor

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.get_decoder().resize_token_embeddings(len(tokenizer), mean_resizing=True)

# # update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# processor = AutoProcessor.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning",
#     use_fast=True,
# )

# Maximum caption length (adjust as needed)
# max_target_length = 50

# # 3. Define a preprocessing function
# def preprocess_examples(batch):
#     # Convert images to RGB (if not already)
#     # images = [img.convert("RGB") for img in batch["image"]]
#     # Process images to get pixel values
#     pixel_values = feature_extractor(
#         images=batch["image"], return_tensors="pt"
#     ).pixel_values
#     # The caption column contains a list of 5 captions.
#     # For training/evaluation, we take the first caption as the target.
#     captions = [cap[0] if isinstance(cap, list) else cap for cap in batch["caption"]]
#     tokenized = tokenizer(
#         captions, padding="max_length", truncation=True, max_length=max_target_length,
#     )

#     # Add processed data to the batch
#     batch["pixel_values"] = pixel_values  # Tensor of shape (batch_size, channels, height, width)

#     batch[label_name] = tokenized.input_ids.copy()
#     batch["decoder_attention_mask"] = tokenized.attention_mask

#     # Replace pad token id's in labels with -100 so they are ignored by the loss
#     batch[label_name] = [
#         [l if l != tokenizer.pad_token_id else -100 for l in label]
#         for label in batch[label_name]
#     ]

#     return batch


# # Apply preprocessing to each split
# train_dataset = train_dataset.map(
#     preprocess_examples, batched=True, remove_columns=train_dataset.column_names
# )
# val_dataset = val_dataset.map(
#     preprocess_examples, batched=True, remove_columns=val_dataset.column_names
# )
# test_dataset = test_dataset_base.map(
#     preprocess_examples, batched=True, remove_columns=test_dataset_base.column_names
# )
# # (Optionally, do the same for test_dataset if you plan to evaluate later)

# # Set the dataset format to PyTorch tensors
# train_dataset.set_format(
#     type="torch", columns=["pixel_values", label_name, "decoder_attention_mask"]
# )
# val_dataset.set_format(
#     type="torch", columns=["pixel_values", label_name, "decoder_attention_mask"]
# )
# test_dataset.set_format(
#     type="torch", columns=["pixel_values", label_name, "decoder_attention_mask"]
# )

if not torch.cuda.is_bf16_supported():
    print()
    print("Warning: bf16 not supported!")
    print()


# 4. Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit-gpt2-finetuned-flickr30k",
    eval_strategy="epoch",
    # eval_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=5,
    # logging_steps=1,
    # max_grad_norm=0.0,
    # gradient_accumulation_steps=1
    # log_level="debug",
    predict_with_generate=True,  # required for caption generation during eval
    # generation_max_length=50,
    # bf16=torch.cuda.is_available(),  # enable mixed precision if using GPU
    # fp16=torch.cuda.is_available(),  # enable mixed precision if using GPU
    # fp16_opt_level="O3",
    # fp16_full_eval=True,
    # bf16_full_eval=False,
    # debug="underflow_overflow",
    # torch_empty_cache_steps=1,
    # eval_accumulation_steps=1,
    # skip_memory_metrics=False,# when false slows down training and eval
    # gradient_checkpointing=True,  # slowdown but helps memory
    auto_find_batch_size=True,
    # adam_epsilon=1e-4,
    optim="adafactor",
    dataloader_pin_memory=True,
    dataloader_num_workers=6,
)
# training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

# 5. Define a compute_metrics function (using BLEU)
# blue_metric = evaluate.load("bleu", keep_in_memory=True)
metric = evaluate.load("rouge", keep_in_memory=True)
# cider_metric = evaluate.load("cider", keep_in_memory=True)

import numpy as np

ignore_pad_token_for_loss = True

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print(labels)
    # print(len(labels))
    # print(len(labels[0]))
    # print("-------------")
    # print(len(preds))
    # print(len(preds[0]))
    # print(preds)
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(decoded_labels)
    # print("-------------")
    # print(decoded_preds)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# TODO check this
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     if isinstance(predictions, tuple):
#         predictions = predictions[0]
#     # Decode predictions and labels into strings (do NOT split them into tokens)
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Pass lists of strings to the metric
#     result = blue_metric.compute(predictions=decoded_preds, references=decoded_labels)
#     return {"bleu": result["bleu"]}


import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    # texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["image"] for example in examples]
    captions = [min(example["caption"], key=len) for example in examples]
    
    # Tokenize the texts and process the images
    outputs = tokenizer(
        captions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    pixel_values = feature_extractor(images, return_tensors="pt").pixel_values

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    decoder_input_ids = outputs.input_ids
    labels = outputs.input_ids.copy()
    decoder_attention_mask = outputs.attention_mask

    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    labels = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(decoder_attention_mask, labels)]
    ]

    # assert all([len(x) == encoder_length for x in inputs.input_ids])
    # assert all([len(x) == decoder_length for x in outputs.input_ids])

    # labels[labels == tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask":decoder_attention_mask}


def custom_data_collator(features):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    # attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    # return {
    #     "pixel_values": pixel_values,
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "return_loss": True,
    # }
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    label_tensors = [torch.as_tensor(f["labels"]) for f in features]
    labels = pad_sequence(
        label_tensors, batch_first=True, padding_value=-100
    )  # model.config.pad_token_id)

    # Pad decoder_attention_mask if available
    if "decoder_attention_mask" in features[0]:
        mask_tensors = [torch.as_tensor(f["decoder_attention_mask"]) for f in features]
        decoder_attention_mask = pad_sequence(
            mask_tensors, batch_first=True, padding_value=-100
        )
    else:
        decoder_attention_mask = None

    batch = {"pixel_values": pixel_values, "labels": labels}  # , "return_loss":True}
    if decoder_attention_mask is not None:
        batch["decoder_attention_mask"] = decoder_attention_mask
    return batch


model.accepts_loss_kwargs = False
# model.config.max_length= 50
# model.config.decoder.max_length= 50
# 6. Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=feature_extractor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 7. Fine-tune the model
train_result = trainer.train()

trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()


# 8. Evaluate the model on the validation set
metrics = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="val",  # max_length=50, num_beams=4
)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
# print("Evaluation results:", metrics)


def get_ansi_color_code(r, g, b):
    return (
        16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
    )


def grayscale_code(p):
    if p < 8:
        return 16
    if p > 248:
        return 231
    return round(((p - 8) / 247) * 24) + 232


def format_pixel(pix):
    return "\x1b[48;5;{}m \x1b[0m".format(pix)


def show_image(img: PIL.Image.Image, grayscale: bool = True):

    h = img.height // 8
    w = img.width // 4

    img = img.resize((w, h), PIL.Image.Resampling.LANCZOS)
    img_arr = np.asarray(img)

    for x in range(h):
        for y in range(w):
            pix = img_arr[x][y]
            if grayscale:
                print(
                    format_pixel(grayscale_code(pix)),
                    sep="",
                    end="",
                ),
            else:
                print(
                    format_pixel(get_ansi_color_code(pix[0], pix[1], pix[2])),
                    sep="",
                    end="",
                )
        print()


# pixel_values = feature_extractor(images=test_dataset["image"], return_tensors="pt").pixel_values
# pixel_values = pixel_values.to("cuda")
def run_val(gen_kwargs):
    # max_length = 50
    pixel_values = feature_extractor(
        images=test_dataset["image"], return_tensors="pt"
    ).pixel_values

    print(gen_kwargs)
    for i, value in enumerate(pixel_values):
        # show_image(torchvision.transforms.v2.functional.to_pil_image(img), grayscale=False)
        output_ids = model.generate(
            value.unsqueeze(0).to("cuda"),
            **gen_kwargs,  # max_length=max_length
        )
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        print(test_dataset["caption"][i][0])
        print("-----Pred----")
        print(preds)


top_p_gen_kwargs = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    # "num_return_sequences": 3,
}
beam_gen_kwargs = {
    "do_sample": False,
    "num_beams": 5,
    # "num_return_sequences": 5,
}
beam_multinomal_gen_kwargs = {
    "do_sample": True,
    "num_beams": 5,
    # "num_return_sequences": 5,
}
run_val(beam_gen_kwargs)
