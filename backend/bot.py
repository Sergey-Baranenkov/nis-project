import io
import os
import telebot
import requests
import urllib.request
from PIL import Image
from pathlib import Path
import dotenv
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from zipfile import ZipFile

dotenv.load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
bot = telebot.TeleBot(API_TOKEN)
sessions = {}  # {user: {'photo': [urls], 'video': url}}
photo = "photo"
video = "video"
num_photos_required = 8
BASE_MEDIA_DIR = "media"
MODEL_URL = os.getenv('MODEL_URL')


def generate_url(file):
    file_id = file["file_id"]
    file_info = bot.get_file(file_id)
    return 'https://api.telegram.org/file/bot{0}/{1}'.format(API_TOKEN, file_info.file_path)


def train_model(user):
    if user not in sessions or photo not in sessions[user] or len(sessions[user][photo]) < num_photos_required:
        return

    zip_path = 'photos.zip'
    with ZipFile(zip_path, mode='w') as zf:
        for i, url in enumerate(sessions[user][photo]):
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
            new_image = image.resize((512, 512))

            img_byte_arr = io.BytesIO()
            new_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            zf.writestr(f'file{i}.jpeg', img_byte_arr)

    # Send photos to backend
    url = f"{MODEL_URL}/train"

    zip_obj = open(zip_path, 'rb')
    files = {
        'file': zip_obj
    }
    data = {
        'id': user
    }

    response = requests.post(url, files=files, data=data)

    os.remove(zip_path)

def generate_video(user, chat_id, prompt):
    if user not in sessions or video not in sessions[user] or not sessions[user][video]:
        return

    input_video = requests.get(sessions[user][video])
    ext = sessions[user][video].split('/')[-1].split('.')[-1]

    # Send video to backend
    url = f"{MODEL_URL}/generate"
    files = {'file': (f"file.{ext}", input_video.content)}
    data = {'id': user, 'prompt': prompt}

    response = requests.post(url, files=files, data=data)
    # Retrieve a video from request and send it to chat
    bot.send_video(chat_id, response.content)


def create_user_if_needed(user):
    if user not in sessions:
        sessions[user] = {photo: [], video: None}


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    json_message = message.json
    photos = json_message[photo]
    if not (photos or len(photos)):
        print("No photos in message :(")
        return

    user = message.from_user.username
    create_user_if_needed(user)

    # If all photos are uploaded
    if len(sessions[user][photo]) == num_photos_required:
        return

    sessions[user][photo].append(generate_url(photos[-1]))

    if len(sessions[user][photo]) == num_photos_required:
        train_model(user)
        bot.send_message(
            message.chat.id,
            text="Move on, tap on /generate"
        )


@bot.message_handler(content_types=['video'])
def handle_video(message):
    json_message = message.json
    video_file = json_message[video]
    if not video_file:
        print("No videos in message :(")
        return
    user = message.from_user.username
    create_user_if_needed(user)

    sessions[user][video] = generate_url(video_file)

    # Check if photos required to add
    if len(sessions[user][photo]) == num_photos_required:
        bot.send_message(
            message.chat.id,
            text=f"Move on, tap on /choose_style"
        )
    else:
        bot.send_message(
            message.chat.id,
            text=f"Please upload {num_photos_required - len(sessions[user][photo])} more photos"
        )


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(
        message.chat.id,
        text=f'Hello, {message.from_user.username}!\nI can generate cool Avatarify video:\n'
             f'Firstly  - /train\n'
             f'Secondly - /generate\n'
             f'Thirdly  - /choose_style'
    )


@bot.message_handler(commands=['train'])
def handle_train(message):
    bot.reply_to(message, "Let's start! Send me 8 selfies to train our model")


@bot.message_handler(commands=['generate'])
def handle_generate(message):
    bot.reply_to(message, "Great! Send me one short video (~3 sec)")


@bot.message_handler(commands=['choose_style'])
def choose_style(message):
    bot.send_message(message.chat.id, "Choose a style", reply_markup=gen_markup())
    # bot.reply_to(message, "Great, the model now training on your photos\nMove on, tap on /generate")


def gen_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(
        InlineKeyboardButton("Pen drawing", callback_data="1"),
        InlineKeyboardButton("Space", callback_data="2"),
        InlineKeyboardButton("Starry night by Van Gogh", callback_data="3"),
        InlineKeyboardButton("Pokemon", callback_data="4"),
        InlineKeyboardButton("Cyberpunk", callback_data="5")
    )
    return markup


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    generate_video(call.from_user.username, call.message.chat.id, call.data)
    bot.send_message(call.message.chat.id, text="Awesome!\nJust upload video to generate something new")


bot.infinity_polling()
