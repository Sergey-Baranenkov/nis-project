import os

from flask import Flask, send_file, request, after_this_request
from flask_json import FlaskJSON, json_response
from flask_cors import CORS
from natsort import natsorted
from glob import glob
from shutil import rmtree
from trainer import Trainer
from video_handler import VideoHandler
from generator import Generator
import zipfile

app = Flask(__name__)
json = FlaskJSON(app)
CORS(app)

@app.route("/train", methods=['POST'])
def train():
    id = request.form['id']
    images_zip = request.files['file']

    file_like_object = images_zip.stream._file

    prefix = f'train_results/{id}'
    os.makedirs(prefix, exist_ok=True)

    images_path = f'{prefix}/images'

    with zipfile.ZipFile(file_like_object) as zip_ref:
        zip_ref.extractall(images_path)


    trainer_instance = Trainer(
        prefix=prefix,
        id=id,
        class_prompt='portrait photo of a woman',
        instance_data_dir=f'{images_path}',
        class_data_dir='./photosofwomen'
    )
    trainer_instance.train()
    return json_response(data={'status': 'ok'})

@app.route("/generate", methods=['POST'])
def generate():
    id = request.form['id']
    prompt_num = int(request.form['prompt'])
    video = request.files['file']
    if video.filename.split('.')[-1] not in ['mp4', 'mov']:
        return json_response(data={'status': 'incorrect format of file'}), 400

    generate_results_prefix = f'generate_results/{id}'
    os.makedirs(generate_results_prefix, exist_ok=True)

    input_video_path = f"{generate_results_prefix}/input_video.{video.filename.split('.')[-1]}"
    video.save(input_video_path)

    train_results_prefix = f'train_results/{id}'
    input_images_path = f'{generate_results_prefix}/images'
    os.makedirs(input_images_path, exist_ok=True)

    video_handler_instance = VideoHandler()
    video_handler_instance.transform_video_to_images(input_video_path, input_images_path)

    output_images_path = f'{generate_results_prefix}/result_images'
    os.makedirs(output_images_path, exist_ok=True)

    weights_path = natsorted(glob(f"{train_results_prefix}/weights" + os.sep + "*"))[-1]

    generator_instance = Generator(
        id=id,
        diffusion_weights_prefix=weights_path,
        input_images_path=input_images_path,
        output_images_path=output_images_path
    )
    generator_instance.generate(prompt_num)

    video_path = f'generate_results/{id}/output_video.mp4'
    video_handler_instance.transform_images_to_video(f"{output_images_path}/restored_imgs", video_path)

    @after_this_request
    def remove_files(response):
        try:
            rmtree(generate_results_prefix)
        except Exception as error:
            app.logger.error(f"Error removing dir {id}", error)
        return response

    return send_file(video_path)

if __name__ == "__main__":
    os.environ['LD_LIBRARY_PATH'] = "/opt/cuda/targets/x86_64-linux/lib"
    os.makedirs('train_results', exist_ok=True)
    os.makedirs('generate_results', exist_ok=True)
    app.run(host='localhost', port=3333)