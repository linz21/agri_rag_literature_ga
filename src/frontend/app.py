"""
Gradio frontend — simple chat-style interface over the RAG API.

Run:
    python src/frontend/app.py
Then open the printed local URL (typically http://127.0.0.1:7860)
"""

import requests
import yaml
import gradio as gr

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

API_URL = f"http://localhost:{cfg['api']['port']}/query"


def ask_question(question: str, history: list) -> tuple[str, list]:
    if not question.strip():
        return "", history

    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        response.raise_for_status()
        data = response.json()

        answer = data["answer"]
        sources_text = "\n".join(
            f"  - {s['title']} ({s['year']})" for s in data["sources"]
        )
        full_response = f"{answer}\n\n**Sources:**\n{sources_text}"

    except requests.exceptions.ConnectionError:
        full_response = (
            "⚠ Could not connect to the API. Make sure it's running: "
            f"`uvicorn src.api.main:app --reload --port {cfg['api']['port']}`"
        )
    except requests.exceptions.HTTPError as e:
        full_response = f"⚠ API error: {e.response.json().get('detail', str(e))}"

    history.append((question, full_response))
    return "", history


with gr.Blocks(title="Agricultural Research Assistant") as demo:
    gr.Markdown("# 🌾 Agricultural Research Assistant")
    gr.Markdown(
        "Ask questions about corn yield, precision agriculture, and crop science. "
        "Answers are grounded in retrieved research paper abstracts with citations."
    )

    chatbot = gr.Chatbot(height=450)
    question_box = gr.Textbox(
        placeholder="e.g. How does nitrogen application timing affect corn yield?",
        label="Your question",
    )
    submit_btn = gr.Button("Ask", variant="primary")

    submit_btn.click(ask_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    question_box.submit(ask_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])

if __name__ == "__main__":
    demo.launch(server_port=cfg["frontend"]["port"])
