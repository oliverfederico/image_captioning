from transformers import pipeline
import gradio as gr

# pipe = pipeline("image-classification", model="openai/clip-vit-base-patch32", torch_dtype='float16')
pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", torch_dtype='float16')

gr.Interface.from_pipeline(pipe).launch()