import gradio as gr
import numpy as np
import onnxruntime as ort
import torch

# Load models (update with your specific model paths)
whisper_model_path = "path/to/whisper.onnx"
smollm_model_path = "path/to/smollm.onnx"

# Initialize ONNX Runtime sessions
whisper_session = ort.InferenceSession(whisper_model_path)
smollm_session = ort.InferenceSession(smollm_model_path)

# Function to run Whisper model for Speech to Text
def whisper_transcribe(audio):
    audio_data = np.frombuffer(audio, dtype=np.float32).flatten()
    # Prepare the input based on the Whisper model requirements
    # This may require adjusting based on the actual model
    input_name = whisper_session.get_inputs()[0].name
    transcription = whisper_session.run(None, {input_name: audio_data})
    return transcription[0]

# Function to run Smollm model for Text Generation
def generate_text(prompt):
    input_ids = np.array([ord(c) for c in prompt]).reshape(1, -1)  # Simple encoding; adjust this
    input_name = smollm_session.get_inputs()[0].name
    output = smollm_session.run(None, {input_name: input_ids})
    generated_text = ''.join(chr(id) for id in output[0][0])  # This also should match your decoding method
    return generated_text

# Create a Gradio interface
def chat_interface(audio, text_input):
    if audio:
        transcription = whisper_transcribe(audio)
    else:
        transcription = ""

    if text_input:
        generated_text = generate_text(text_input)
    else:
        generated_text = ""

    return transcription, generated_text

with gr.Blocks() as app:
    gr.Markdown("# DataNacci Chat App")
    with gr.Row():
        audio_input = gr.Audio(source="microphone", type="numpy", label="Speak:")
        text_input = gr.Textbox(label="Type your message:")
    
    output_transcription = gr.Textbox(label="Transcription:", interactive=False)
    output_response = gr.Textbox(label="Response:", interactive=False)

    submit_button = gr.Button("Submit")
    submit_button.click(chat_interface, inputs=[audio_input, text_input], outputs=[output_transcription, output_response])

# Launch the app
if __name__ == "__main__":
    app.launch()
