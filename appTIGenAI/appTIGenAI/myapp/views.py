from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt

#from django.http import JsonResponse

import os
from PIL import Image

# Create your views here.
import os
from .forms import ImageForm
from .models import Image

from myapp.vision_transformer import vit

# importing python script
#from . import predictions

# load the home.html
def home(request):
    return render(request, 'home.html')

# upload the images in upload.html
@csrf_protect
@csrf_exempt
def upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        return render(request, 'upload.html')

    images = request.FILES.get("images")
    
    image_data = vit.predict(images)

    context = {
        "form": form,
        #"image_urls": image_urls,
        #"vect_captions": vect_captions
        "image_data": image_data
    } 
    return JsonResponse({images.name: image_data})

# # upload the images in upload.html
# def upload(request):
#     if request.method == 'POST':
#         form = ImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
            
#             #return HttpResponse("File uploaded successfully!")
#             #return render(request, "caption.html")
#     else:
#         form = ImageForm()
    
#     '''
#     ######## load the ViT-GPT-2 model ########
#     import torch
#     from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    
#     model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     ######### load all directories of images #########
#     image_urls = []

#     static_dir = os.path.join(os.getcwd(), "static/images")
#     if os.path.isdir(static_dir):
#         for filename in os.listdir(static_dir):
#             if filename.endswith(".jpg") or filename.endswith(".png"):                                
#                 image_urls.append(filename)    

#     ######## build an array data structure to store all images #########

#     # specify the path to images directory
#     image_path = os.path.join(os.getcwd(), "static", "images")

#     # vector of images
#     images = []

#     # make loop to open all the images
#     from PIL import Image

#     for filename in os.listdir(image_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):

#             # use the PIL to open image
#             img = Image.open(os.path.join(image_path, filename))

#             # add this image to the list of images
#             images.append(img)
#     '''

#     ######### generate the captions ###########

#     # vector of captions
#     vect_captions =[]

#     # predefinitions
#     max_length = 16
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#     # generate the caption
#     pixel_values = feature_extractor(images = images, return_tensors = "pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
#     preds = [pred.strip() for pred in preds]
#     vect_captions = preds

#     # concatenate vectors of url-images and captions
#     image_data = zip(image_urls, vect_captions)

#     context = {
#         "form": form,
#         #"image_urls": image_urls,
#         #"vect_captions": vect_captions
#         "image_data": image_data
#     }

#     return render(request, 'upload.html', context)

