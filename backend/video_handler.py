import os


class VideoHandler():
    def __init__(self):
        pass

    def transform_video_to_images(self, video_path, output_dir):
        os.system(f"""
            ffmpeg -i {video_path} -vf "fps=29.97" -start_number 1000 -q:v 1 {output_dir}/%04d.png
        """)

    def transform_images_to_video(self, images_path, output_path):
        os.system(f"""
            ffmpeg -framerate 30 -pattern_type glob -i "{images_path}/*.png" -c:v libx264 -pix_fmt yuv420p "{output_path}"
        """)