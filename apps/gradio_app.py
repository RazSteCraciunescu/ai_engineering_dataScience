"""Optional Gradio interface for local model inference and grounded retrieval."""
from __future__ import annotations

from feedback_intelligence.modeling import predict_priority, predict_resolution
from feedback_intelligence.rag import answer


def assess(title: str, message: str, category: str, affected_users: int) -> tuple[str, str]:
    payload = {
        "title": title,
        "message": message,
        "category": category,
        "affected_users": affected_users,
        "channel": "web",
        "product": "workspace",
        "attachment_count": 0,
        "priority": "medium",
    }
    priority = predict_priority(payload)
    resolution = predict_resolution(payload)
    return priority["priority"], f"{resolution['estimated_resolution_hours']:.2f} hours"


def retrieve(question: str) -> str:
    result = answer(question, limit=3)
    sources = ", ".join(item["source"] for item in result["evidence"])
    return f"{result['answer']}\n\nSources: {sources}"


def main() -> None:
    import gradio as gr

    with gr.Blocks(title="Feedback Intelligence") as demo:
        gr.Markdown("# Feedback Intelligence\nLocal ML inference and evidence-labelled retrieval.")
        with gr.Tab("Predict"):
            title = gr.Textbox(label="Title")
            message = gr.Textbox(label="Message", lines=4)
            category = gr.Dropdown(
                ["billing", "account", "performance", "usability", "feature request", "reliability"],
                value="reliability",
                label="Category",
            )
            users = gr.Number(value=25, minimum=1, label="Affected users")
            run = gr.Button("Assess")
            predicted_priority = gr.Textbox(label="Predicted priority")
            predicted_resolution = gr.Textbox(label="Estimated resolution")
            run.click(assess, [title, message, category, users], [predicted_priority, predicted_resolution])
        with gr.Tab("Knowledge search"):
            question = gr.Textbox(label="Question")
            response = gr.Textbox(label="Grounded answer", lines=7)
            gr.Button("Retrieve").click(retrieve, question, response)
    demo.launch()


if __name__ == "__main__":
    main()
