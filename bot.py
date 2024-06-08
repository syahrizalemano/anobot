import os
import logging
from telebot import TeleBot
from telebot.apihelper import ApiTelegramException
from telebot.types import Message
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up Telegram bot
TOKEN = os.environ['TOKEN']
bot = TeleBot(TOKEN)

# Set up model and processor
DEVICE = torch.device("cuda")
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=TOKEN,
)
MODEL = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE)
image_seq_len = MODEL.config.perceiver_config.resampler_n_latents
BOS_TOKEN = PROCESSOR.tokenizer.bos_token
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

# Define custom transform function
def convert_to_rgb(image):
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)
    x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
    x = PROCESSOR.image_processor.rescale(x, scale=1 / 255)
    x = PROCESSOR.image_processor.normalize(
        x,
        mean=PROCESSOR.image_processor.image_mean,
        std=PROCESSOR.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x

# Define bot commands
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Welcome to the bot!")

@bot.message_handler(commands=['generate'])
def generate(message):
    try:
        # Get the image from the user
        image = message.photo[-1].get_file()
        image_path = os.path.join('/tmp', image.file_path.split('/')[-1])
        image.download(image_path)

        # Convert the image to RGB
        image = Image.open(image_path)
        image = convert_to_rgb(image)

        # Process the image
        inputs = PROCESSOR.tokenizer(
            f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs["pixel_values"] = custom_transform(image)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate text
        generated_ids = MODEL.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_length=4096)
        generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Send the generated text to the user
        bot.send_message(message.chat.id, generated_text)

    except ApiTelegramException as e:
        bot.send_message(message.chat.id, f"Error: {e}")

if __name__ == "__main__":
    bot.polling()
