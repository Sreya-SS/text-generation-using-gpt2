from transformers import pipeline
import gradio as gr

# Load GPT-2
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Generate Function
def generate_text(prompt):

    if not prompt.strip():
        return "⚠️ Please enter a topic."

    formatted_prompt = f"Write a meaningful paragraph about: {prompt}"

    result = generator(
        formatted_prompt,
        max_new_tokens=250,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=50256
    )

    return result[0]["generated_text"]


# Custom Styling
custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 20px;
}
"""

# UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class='main-title'>
            🤖 GPT-2 Text Generator
        </div>

        <div class='sub-title'>
            Generate meaningful content using Hugging Face GPT-2
        </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            prompt = gr.Textbox(
                label="Enter Topic",
                lines=6,
                placeholder="Example: Artificial Intelligence, Climate Change, Space Exploration..."
            )

            generate_btn = gr.Button(
                "✨ Generate Content",
                variant="primary"
            )

            clear_btn = gr.Button("🗑 Clear")

        with gr.Column(scale=1):

            output = gr.Textbox(
                label="Generated Text",
                lines=18,
                show_copy_button=True
            )

    generate_btn.click(
        fn=generate_text,
        inputs=prompt,
        outputs=output
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[prompt, output]
    )

    gr.HTML("""
        <div class='footer'>
            Built using GPT-2 • Hugging Face Transformers • Gradio
        </div>
    """)

demo.launch()
