from transformers import pipeline
import gradio as gr

# pipe = pipeline("image-classification", model="openai/clip-vit-base-patch32", torch_dtype='float16')
pipe = pipeline("visual-question-answering", model="Salesforce/blip2-opt-2.7b", device="cpu", use_fast=True)
# pipe = pipeline("image-to-text", model="./vit-gpt2-finetuned-flickr30k/checkpoint-5000")#, torch_dtype='float16')

gr.Interface.from_pipeline(pipe).launch()