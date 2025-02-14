import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

controlnet = ControlNetModel.from_pretrained(
    "BASE_CONTROLNET_MODEL", torch_dtype=torch.float16
).to(device) # TODO: change this to the controlnet model you want to use

# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "BASE_MODEL", # TODO: change this to the base model you want to use
#     controlnet=controlnet,
#     safety_checker=None,
#     torch_dtype=torch.float16,
# ).to(device)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "BASE_MODEL", # TODO: change this to the base model you want to use
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)

pipe.enable_xformers_memory_efficient_attention()

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def create_code(content: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=16,
        border=0,
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # find smallest image size multiple of 256 that can fit qr
    offset_min = 8 * 16
    w, h = img.size
    w = (w + 255 + offset_min) // 256 * 256
    h = (h + 255 + offset_min) // 256 * 256
    if w > 1024:
        raise gr.Error("QR code is too large, please use a shorter content")
    bg = Image.new('L', (w, h), 128)

    # align on 16px grid
    coords = ((w - img.size[0]) // 2 // 16 * 16,
              (h - img.size[1]) // 2 // 16 * 16)
    bg.paste(img, coords)
    return bg

def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    image_path: Image.Image,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    seed: int = -1,
    sampler="Euler a",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qr_code_content is None or qr_code_content == "":
        raise gr.Error("QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    print("Generating QR Code from content")
    qrcode_image = create_code(qr_code_content)

    init_image = qrcode_image

    # out = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     image=qrcode_image,
    #     width=qrcode_image.width,
    #     height=qrcode_image.height,
    #     guidance_scale=float(guidance_scale),
    #     controlnet_conditioning_scale=float(controlnet_conditioning_scale),
    #     generator=generator,
    #     num_inference_steps=40,
    # )

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_path,
        control_image=qrcode_image,
        width=qrcode_image.width,
        height=qrcode_image.height,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        num_inference_steps=40,
    )

    return out.images[0]


if __name__ == "__main__":

  with gr.Blocks(theme="NoCrypt/miku") as blocks:
      with gr.Row():
          with gr.Column():
              with gr.Row():
                  with gr.Column():
                      qr_code_content = gr.Textbox(
                          label="QR Code Content or URL",
                          info="The text you want to encode into the QR code",
                          value="",
                      )
                      prompt = gr.Textbox(
                          label="Prompt",
                          info="Prompt that guides the generation towards",
                      )
                      negative_prompt = gr.Textbox(
                          label="Negative Prompt",
                          value="ugly, disfigured, low quality, blurry, nsfw",
                          info="Prompt that guides the generation away from",
                      )
                  with gr.Column():
                      input_image = gr.Image(
                          label="Input Image",
                          info="Upload an image to use as reference",
                          type="pil"
                      )

              with gr.Accordion(
                  label="Params: The generated QR Code functionality is largely influenced by the parameters detailed below",
                  open=False,
              ):
                  controlnet_conditioning_scale = gr.Slider(
                      minimum=0.5,
                      maximum=2.5,
                      step=0.01,
                      value=1.7,
                      label="Controlnet Conditioning Scale",
                      info="""Controls the readability/creativity of the QR code.
                      High values: The generated QR code will be more readable.
                      Low values: The generated QR code will be more creative.
                      """
                  )
                  guidance_scale = gr.Slider(
                      minimum=0.0,
                      maximum=25.0,
                      step=0.25,
                      value=7,
                      label="Guidance Scale",
                      info="Controls the amount of guidance the text prompt guides the image generation"
                  )
                  sampler = gr.Dropdown(choices=list(
                      SAMPLER_MAP.keys()), value="Euler a", label="Sampler")
                  seed = gr.Number(
                      minimum=-1,
                      maximum=9999999999,
                      step=1,
                      value=42,
                      label="Seed",
                      info="Seed for the random number generator. Set to -1 for a random seed"
                  )
              with gr.Row():
                  run_btn = gr.Button("Run")
          with gr.Column():
              result_image = gr.Image(label="Result Image", elem_id="result_image")
      run_btn.click(
          inference,
          inputs=[
              qr_code_content,
              prompt,
              negative_prompt,
              input_image,
              guidance_scale,
              controlnet_conditioning_scale,
              seed,
              sampler,
          ],
          outputs=[result_image],
      )

  blocks.launch(share=bool(os.environ.get("SHARE", True)), show_api=False)