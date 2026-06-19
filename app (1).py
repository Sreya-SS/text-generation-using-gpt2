from transformers import pipeline
import gradio as gr

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_text(prompt):

    if not prompt.strip():
        return "Please enter a prompt."

    prompt = f"Write a meaningful paragraph about: {prompt}"

    result = generator(
        prompt,
        max_new_tokens=250,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=50256
    )

    return result[0]["generated_text"]


demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your topic"
    ),
    outputs=gr.Textbox(
        lines=20,
        label="Generated Text"
    ),
    title="GPT-2 Text Generator",
    description="Text generation using GPT-2 from Hugging Face Transformers"
)

demo.launch()