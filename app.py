from flask import Flask, request, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Load GPT-2 model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Text Generator</title>
    <style>
        body{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }
        h1{
            text-align:center;
        }
        textarea{
            width:100%;
            padding:10px;
            margin-top:10px;
        }
        button{
            margin-top:10px;
            padding:10px 20px;
            cursor:pointer;
        }
        input{
            padding:5px;
        }
    </style>
</head>
<body>
    <h1>AI Text Generator</h1>
    <form method="POST">
        <label>Enter Prompt:</label><br>
        <textarea name="prompt" rows="6" required>{{ prompt }}</textarea>
        <br><br>
        <label>Temperature:</label>
        <input
            type="number"
            name="temperature"
            min="0.7"
            max="1.0"
            step="0.1"
            value="{{ temperature }}"
        >
        <br><br>
        <button type="submit">Generate Text</button>
    </form>
    {% if output %}
    <h3>Generated Text:</h3>
    <textarea rows="12" readonly>{{ output }}</textarea>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    prompt = ""
    temperature = 0.8

    if request.method == "POST":
        prompt = request.form["prompt"]
        temperature = float(request.form["temperature"])

        result = generator(
            prompt,
            max_new_tokens=250,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        output = result[0]["generated_text"]

    return render_template_string(
        HTML,
        output=output,
        prompt=prompt,
        temperature=temperature
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
