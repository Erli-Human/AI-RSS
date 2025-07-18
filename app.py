import gradio as gr
import numpy as np
import onnxruntime as ort
import os
import requests

# Define URLs for the models
whisper_encoder_model_url = "https://huggingface.co/onnx-community/whisper-base-ONNX/resolve/main/onnx/encoder_model.onnx"
whisper_decoder_model_url = "https://huggingface.co/onnx-community/whisper-base-ONNX/resolve/main/onnx/decoder_model.onnx"  # Make sure to use the correct decoder model link if available
smollm_model_url = "https://huggingface.co/{your_smollm_model}/resolve/main/model.onnx"  # Replace with the actual Smollm model link
kokoro_model_url = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"

# Define local paths for the models
whisper_encoder_model_path = "encoder_model.onnx"
whisper_decoder_model_path = "decoder_model.onnx"
smollm_model_path = "smollm.onnx"
kokoro_model_path = "kokoro_model.onnx"

# Function to download a file
def download_file(url, dest_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download {url}: Status Code {response.status_code}")

# Function to check and download models
def download_models():
    model_paths = {
        "encoder": whisper_encoder_model_path,
        "decoder": whisper_decoder_model_path,
        "smollm": smollm_model_path,
        "kokoro": kokoro_model_path
    }
    model_urls = {
        "encoder": whisper_encoder_model_url,
        "decoder": whisper_decoder_model_url,
        "smollm": smollm_model_url,
        "kokoro": kokoro_model_url
    }

    for key in model_paths:
        if not os.path.exists(model_paths[key]):
            print(f"{model_paths[key]} does not exist. Downloading...")
            download_file(model_urls[key], model_paths[key])
        else:
            print(f"{model_paths[key]} already exists.")

# Download models upon starting the app
download_models()

# Initialize ONNX Runtime sessions
encoder_session = ort.InferenceSession(whisper_encoder_model_path)
decoder_session = ort.InferenceSession(whisper_decoder_model_path)
smollm_session = ort.InferenceSession(smollm_model_path)
kokoro_session = ort.InferenceSession(kokoro_model_path)

# Function for Whisper model to transcribe audio
def whisper_transcribe(audio):
    if audio is None or len(audio) == 0:
        return "No audio input provided."

    audio_data = np.frombuffer(audio, dtype=np.float32).flatten()
    
    # Ensure the encoder expects the correct input size
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_output = encoder_session.run(None, {encoder_input_name: audio_data})
    
    # Process through the decoder
    decoder_input_name = decoder_session.get_inputs()[0].name
    decoder_output = decoder_session.run(None, {decoder_input_name: encoder_output[0]})

    # Return the transcription - this must match the output of your decoder
    transcription = decoder_output[0]  # Adjust according to your model's output needs
    return transcription

# Function for Smollm model for text generation
def generate_text(prompt):
    if not prompt:
        return "No prompt provided."
    
    # Encode the input text appropriately for your model
    input_ids = np.array([ord(c) for c in prompt]).reshape(1, -1)  # Adjust based on your text encoding strategy
    input_name = smollm_session.get_inputs()[0].name
    output = smollm_session.run(None, {input_name: input_ids})
    generated_text = ''.join(chr(id) for id in output[0][0])  # Adjust according to your model's output needs
    return generated_text

# Function for Kokoro model (if needed)
def run_kokoro(input_text):
    if not input_text:
        return "No input text provided."
    
    input_ids = np.array([ord(c) for c in input_text]).reshape(1, -1)  # Adjust accordingly
    input_name = kokoro_session.get_inputs()[0].name
    output = kokoro_session.run(None, {input_name: input_ids})
    output_text = ''.join(chr(id) for id in output[0][0])  # Adjust according to your model's output needs
    return output_text

# Create a Gradio interface
def chat_interface(audio, text_input):
    transcription = whisper_transcribe(audio)
    generated_text = generate_text(text_input)
    kokoro_response = run_kokoro(text_input)

    return transcription, generated_text, kokoro_response

# Gradio app layout
with gr.Blocks() as app:
    gr.Markdown("# DataNacci Chat App")

    with gr.Row():
        audio_input = gr.Audio(source="microphone", type="numpy", label="Speak:")
        text_input = gr.Textbox(label="Type your message:")
    
    output_transcription = gr.Textbox(label="Transcription:", interactive=False)
    output_response = gr.Textbox(label="Response from Smollm:", interactive=False)
    output_kokoro = gr.Textbox(label="Response from Kokoro:", interactive=False)

    submit_button = gr.Button("Submit")
    submit_button.click(chat_interface, inputs=[audio_input, text_input], outputs=[output_transcription, output_response, output_kokoro])

# Launch the app
if __name__ == "__main__":
    app.launch()
