"""Optional local Stable Diffusion example. Model weights are not included in the repository."""
from __future__ import annotations


def main() -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    model_id = "stabilityai/sdxl-turbo"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to("cuda")
    image = pipe(
        "clean abstract illustration of an offline data engineering pipeline, dark dashboard style",
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0]
    image.save("artifacts/reports/generated-pipeline-concept.png")


if __name__ == "__main__":
    main()
