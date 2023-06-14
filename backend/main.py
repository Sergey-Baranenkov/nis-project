import os

from flask import Flask, send_file
from flask_json import FlaskJSON, json_response
from flask_cors import CORS
from natsort import natsorted
from glob import glob

from trainer import Trainer
from video_handler import VideoHandler
from generator import Generator

app = Flask(__name__)
json = FlaskJSON(app)
CORS(app)


id = 'test'
@app.route("/train", methods=['POST'])
def train():
    prefix = f'train_results/{id}'
    trainer_instance = Trainer(
        prefix=prefix,
        id=id,
        class_prompt='portrait photo of a woman',
        instance_data_dir='/home/araxal/nis/nis-project/data/lolli_poli',
        class_data_dir='/home/araxal/nis/nis-project/data/photosofwomen'
    )
    trainer_instance.train()
    return json_response(data={'status': 'ok'})

@app.route("/generate", methods=['POST'])
def generate():
    # TODO Переделать на /tmp
    generate_results_prefix = f'generate_results/{id}'
    train_results_prefix = f'train_results/{id}'
    input_images_path = f'{generate_results_prefix}/images'
    os.makedirs(input_images_path, exist_ok=True)

    video_handler_instance = VideoHandler()
    video_handler_instance.transform_video_to_images('/home/araxal/nis/nis-project/IMG_1608.mov', input_images_path)

    output_images_path = f'{generate_results_prefix}/result_images'
    os.makedirs(output_images_path, exist_ok=True)

    prompt = f'magical ({id}:1.4) midjourney style hyperrealism Cyberpunk+, city background'
    negative_prompt = "out of frame, duplicate, watermark, signature, text, ugly, blurry, deformed"

    weights_path = natsorted(glob(f"{train_results_prefix}/weights" + os.sep + "*"))[-1]

    generator_instance = Generator(
        id=id,
        diffusion_weights_prefix=weights_path,
        input_images_path=input_images_path,
        output_images_path=output_images_path
    )
    generator_instance.generate(prompt, negative_prompt)

    video_path = f'generate_results/{id}/video.mp4'
    video_handler_instance.transform_images_to_video(f"{output_images_path}/restored_imgs", video_path)
    os.remove(input_images_path)
    os.remove(output_images_path)
    return send_file(video_path)

if __name__ == "__main__":
    os.environ['LD_LIBRARY_PATH'] = "/opt/cuda/targets/x86_64-linux/lib"
    os.makedirs('train_results', exist_ok=True)
    os.makedirs('generate_results', exist_ok=True)
    app.run(host='localhost', port=3333)