import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

'''
def generate_urls():
  image_urls = []
  static_dir = os.path.join(os.getcwd(), "static/images")
  print(static_dir)

  if os.path.isdir(static_dir):
    for filename in os.listdir(static_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_urls.append(filename)    
  print(image_urls)

  return image_urls
'''

def predict_step(image_paths):
  # predefinitions
  max_length = 16
  num_beams = 4
  gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

  # read the images
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  #pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]

  return preds


