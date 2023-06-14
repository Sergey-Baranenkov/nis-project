import torch
from diffusers.utils import load_image
import cv2
import numpy as np
from diffusers import ControlNetModel
import os
from lalala import StableDiffusionControlNetInpaintImg2ImgPipeline
from PIL import Image, ImageDraw



controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")
img2img_pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained("parlance/dreamlike-diffusion-1.0-inpainting",
                                                                               custom_pipeline="checkpoint_merger.py",
                                                                               torch_dtype=torch.float16).to("cuda")
img2img_pipe.enable_xformers_memory_efficient_attention()

class Generator():
    def __init__(self,
                 id,
                 diffusion_weights_prefix,
                 input_images_path,
                 output_images_path
                 ):
        self.id = id
        self.diffusion_weights_prefix = diffusion_weights_prefix
        self.input_images_path = input_images_path
        self.output_images_path = output_images_path


    def get_gcuda_generator(self, seed = 803465):
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(seed)
        return g_cuda

    def generate(self, prompt, negative_prompt):
        merged_pipe = img2img_pipe.merge(["parlance/dreamlike-diffusion-1.0-inpainting",
                                          self.diffusion_weights_prefix,
                                          "runwayml/stable-diffusion-v1-5"],
                                         controlnet=controlnet, torch_dtype=torch.float16,
                                         interp='add_diff', alpha=0.1, force=True)

        merged_pipe.safety_checker = lambda images, clip_input: (images, False)
        merged_pipe.to("cuda")

        self.transform_images(merged_pipe, prompt, negative_prompt)
        os.remove(f"{self.output_images_path}/1000.png")
        self.prettify_gfpgan()

    def prettify_gfpgan(self):
        os.system(f"""
            python inference_gfpgan.py -i "{self.output_images_path}" -o "{self.output_images_path}" -v 1.3 -s 2 --bg_upsampler realesrgan
        """)

    def transform_images(self, merged_pipe, prompt, negative_prompt):
        data_dir = self.input_images_path
        save_dir = self.output_images_path

        g_cuda = self.get_gcuda_generator()

        for i, filename in enumerate(sorted(os.listdir(data_dir))):
            img_path = os.path.join(data_dir, filename)
            image = load_image(img_path)
            image = image.resize((512, 512))
            image = np.array(image)

            low_threshold = 120
            high_threshold = 150

            canny_image = cv2.Canny(image, low_threshold, high_threshold)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_image = Image.fromarray(canny_image)
            width, height = canny_image.size

            if i == 0:
                latent_mask = Image.new("RGB", (width, height), "white")

                result = merged_pipe(prompt,
                                     negative_prompt=negative_prompt,
                                     num_inference_steps=30,
                                     generator=g_cuda,
                                     image=image,
                                     controlnet_conditioning_image=canny_image,
                                     controlnet_conditioning_scale=1.0,
                                     strength=1.0,
                                     mask_image=latent_mask,
                                     guidance_scale=20
                                     ).images[0].crop((0, 0, width, height))

                frame_1 = result.copy()
                canny_image_1 = canny_image.copy()

            elif i == 1:
                total_width = width * 2
                images = Image.new('RGB', (total_width, height))
                cannys = Image.new('RGB', (total_width, height))

                x_offset = 0
                for im in [frame_1,
                           Image.fromarray(np.array(image * 0.5 + np.array(frame_1) * 0.5).astype(np.uint8))
                           ]:
                    images.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                x_offset = 0
                for im in [canny_image_1, canny_image]:
                    cannys.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                latent_mask = Image.new("RGB", (width * 2, height), "black")
                latent_draw = ImageDraw.Draw(latent_mask)
                latent_draw.rectangle((width, 0, width * 2, height), fill="white")

                result = merged_pipe(prompt,
                                     negative_prompt=negative_prompt,
                                     num_inference_steps=50,
                                     generator=g_cuda,
                                     image=images,
                                     controlnet_conditioning_image=cannys,
                                     controlnet_conditioning_scale=0.9,
                                     strength=0.5,
                                     mask_image=latent_mask,
                                     guidance_scale=30
                                     ).images[0].crop((width, 0, width * 2, height))

                canny_image_2 = canny_image.copy()
                frame_2 = result.copy()
            else:
                total_width = width * 3
                images = Image.new('RGB', (total_width, height))
                cannys = Image.new('RGB', (total_width, height))

                x_offset = 0
                for im in [frame_2,
                           Image.fromarray(np.array(image * 0.5 + np.array(frame_1) * 0.5).astype(np.uint8)),
                           frame_1]:
                    images.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                x_offset = 0
                for im in [canny_image_2, canny_image, canny_image_1]:
                    cannys.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                latent_mask = Image.new("RGB", (width * 3, height), "black")
                latent_draw = ImageDraw.Draw(latent_mask)
                latent_draw.rectangle((width, 0, width * 2, height), fill="white")

                result = merged_pipe(prompt,
                                     negative_prompt=negative_prompt,
                                     num_inference_steps=50,
                                     generator=g_cuda,
                                     image=images,
                                     controlnet_conditioning_image=cannys,
                                     controlnet_conditioning_scale=0.9,
                                     strength=0.5,
                                     mask_image=latent_mask,
                                     guidance_scale=30
                                     ).images[0].crop((width, 0, width * 2, height))

                canny_image_2 = canny_image.copy()
                frame_2 = result.copy()

            result.save(os.path.join(save_dir, filename))