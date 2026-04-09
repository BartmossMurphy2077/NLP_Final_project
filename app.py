"""Entry point for local runs and Hugging Face Spaces deployments."""

from src.deployment.gradio_app import demo


if __name__ == "__main__":
    demo.launch()

