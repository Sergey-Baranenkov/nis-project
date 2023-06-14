import json
import os
from natsort import natsorted
from glob import glob


class Trainer():
    def __init__(self,
                 prefix,
                 id,
                 instance_data_dir,
                 class_data_dir,
                 class_prompt
                 ):
        # user_id
        self.id = id
        self.prefix = prefix
        self.concepts_list_path = f"{self.prefix}/concepts_list.json"
        self.weights_path = f"{self.prefix}/weights"

        self.instance_data_dir = instance_data_dir
        self.class_data_dir = class_data_dir
        self.class_prompt = class_prompt
        os.makedirs(f"{self.prefix}/weights", exist_ok=True)

    def train(self):
        self.save_concepts_list()
        self.train_dreambooth()
        self.convert_to_original_stable_diffusion()

    def save_concepts_list(self):
        concepts_list = [
            {
                "instance_prompt": self.id,
                "class_prompt": self.class_prompt,
                "instance_data_dir": self.instance_data_dir,
                "class_data_dir": self.class_data_dir
            },
        ]

        with open(self.concepts_list_path, "w") as f:
            json.dump(concepts_list, f, indent=4)

    def train_dreambooth(self):
        os.system(f"""
            python3 train_dreambooth.py   \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"   \
            --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse"   \
            --output_dir={self.weights_path}   \
            --revision="fp16"   \
            --with_prior_preservation  \
            --prior_loss_weight=1.0   \
            --seed=1337   \
            --resolution=512   \
            --train_batch_size=1   \
            --train_text_encoder   \
            --mixed_precision="fp16"   \
            --gradient_accumulation_steps=1   \
            --learning_rate=1e-6   \
            --use_8bit_adam \
            --lr_scheduler="constant"   \
            --lr_warmup_steps=64   \
            --num_class_images=90   \
            --sample_batch_size=1   \
            --max_train_steps=640   \
            --save_interval=700   \
            --concepts_list={self.concepts_list_path}   \
            --gradient_checkpointing   \
            --class_prompt=concepts_list['class_prompt']  \
            --not_cache_latents
        """)

    def convert_to_original_stable_diffusion(self):
        weights_path = natsorted(glob(self.weights_path + os.sep + "*"))[-1]
        ckpt_path = weights_path + "/model.ckpt"

        os.system(f"""
            python3 convert_diffusers_to_original_stable_diffusion.py --model_path "{weights_path}" --checkpoint_path "{ckpt_path}" --half
        """)