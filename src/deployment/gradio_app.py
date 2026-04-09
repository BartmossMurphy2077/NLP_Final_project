"""Gradio app for comparing deployed sentiment model variants."""

from __future__ import annotations

from textwrap import dedent

import gradio as gr

from .inference import InferenceService


SERVICE = InferenceService()


def _status_markdown() -> str:
    lines = [
        "## Model availability",
        "",
        "| Variant | Status | Notes |",
        "| --- | --- | --- |",
    ]

    for variant in SERVICE.variants.values():
        status = "Available" if variant.available else "Unavailable"
        note = (
            f"`{variant.model_dir}`"
            if variant.available and variant.model_dir is not None
            else variant.reason
        )
        lines.append(f"| {variant.spec.title} | {status} | {note} |")

    return "\n".join(lines)


def _default_model_key() -> str | None:
    available = SERVICE.available_variants()
    if not available:
        return None
    return available[0].spec.key


def predict_for_gradio(text: str, model_key: str):
    result = SERVICE.predict(text, model_key)
    details = dedent(
        f"""
        **Prediction:** {result.predicted_label}

        **Detected slang profile:** {result.detected_slang_label}

        **Model path:** `{result.model_dir}`
        """
    ).strip()
    return result.scores, details, result.cleaned_text, result.prepared_text


def build_demo() -> gr.Blocks:
    available = SERVICE.available_variants()
    status_md = _status_markdown()

    with gr.Blocks(title="Sentiment Model Deployment Demo") as demo:
        gr.Markdown(
            dedent(
                """
                # Sentiment Model Deployment Demo

                Compare the exported sentiment models from `MODELS_FINAL` using one interface.
                The app applies the same URL/mention cleaning as training, then runs the
                variant-specific text preparation before inference.
                """
            ).strip()
        )
        gr.Markdown(status_md)

        if not available:
            gr.Markdown(
                "No deployable model exports were found. Add at least one `final_model` "
                "directory with weights under `MODELS_FINAL`."
            )
            return demo

        model_choices = [(variant.spec.title, variant.spec.key) for variant in available]

        with gr.Row():
            model_input = gr.Dropdown(
                choices=model_choices,
                value=_default_model_key(),
                label="Model variant",
            )
            text_input = gr.Textbox(
                label="Input text",
                lines=5,
                placeholder="Enter a tweet, headline, or short sentence...",
            )

        submit = gr.Button("Run prediction", variant="primary")
        label_output = gr.Label(label="Class probabilities")
        summary_output = gr.Markdown()
        cleaned_output = gr.Textbox(label="Cleaned text passed into variant builder")
        prepared_output = gr.Textbox(label="Final text sent to tokenizer")

        submit.click(
            fn=predict_for_gradio,
            inputs=[text_input, model_input],
            outputs=[label_output, summary_output, cleaned_output, prepared_output],
        )

        gr.Examples(
            examples=[
                ["This movie was lowkey fire, I loved it lol", _default_model_key()],
                ["The update is okay, nothing special, just neutral overall.", _default_model_key()],
                ["Bruh this app is trash and I hate the new UI!!!", _default_model_key()],
            ],
            inputs=[text_input, model_input],
        )

    return demo


demo = build_demo()

