import gradio as gr
from gradio_app.model_inference import predict_masks

with gr.Blocks() as FoodSeg_GUI:
    # Centered title and subtitle
    gr.Markdown(
        "# **<p align='center'>Fine-Tuned Mask2Former for FoodSeg103 Semantic Segmentation</p>**"
    )
    gr.Markdown("<p align='center'>By Nima Vahdat</p>")

    # Group for input and output
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1.5):
                input_image = gr.Image(
                    type="filepath", label="Choose your image or drag and drop here:"
                )
            with gr.Column(scale=1.5):
                output_image = gr.Image(label="Mask2Former Output:")

    # Group for the run button
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                start_run = gr.Button("Get the output")

    # Button click event
    start_run.click(predict_masks, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    FoodSeg_GUI.launch(share=True, debug=False)
