import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

import logging

logger = logging.getLogger(__name__)

class ViTransformer:
    def __init__(self) -> None:
        logger.info("Start to load model.")
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        logger.info("Downloading ends.")

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model.to(self.device)
        logger.info("Load model ends.")

    
    def predict(self, images): 
        logger.info("Start to predict.")
        vect_captions = []
        images = Image.open(images)

        # generate the caption
        pixel_values = self.feature_extractor(images.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        preds = [pred.strip() for pred in preds]
        vect_captions = preds
        return "".join(vect_captions)

vit = ViTransformer()
