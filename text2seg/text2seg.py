"""

Text2seg project

"""

# import package
from PIL import Image
import requests
import numpy as np
import torch
import cv2
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import os, sys
import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# load GroundingDINO
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate

# CLIP Surgery
import CLIP_Surgery.clip as clips

# CLIP original
import clip

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry

from text2seg.utils import *

# set constant values

clip_prompt = {"building":["roof", 
               "building", 
               "construction", 
               "office", 
               "edifice", 
               "architecture", 
               "Establishment", 
               "Facility", 
               "House",
               "Mansion", 
               "Tower",
               "Skyscraper",
               "Monument",
               "Shrine",
               "Temple",
               "Palace",
               "iron roof",
               "cabin"
                          ],
               "road":["Street",
                       "road",
                       "Highway",
                       "Path",
                       "Route",
                       "Lane",
                       "Avenue",
                       "Boulevard",
                       "Way",
                       "Alley",
                       "Drive"
                      ],
               "water":["Liquid",
                       "water"
                      ],
               "barren":["barren",
                       "Desolate",
                       "Sterile",
                       "Unproductive",
                       "Infertile",
                       "Wasteland",
                      ],
               "forest":["Woodland",
                       "Jungle",
                       "Timberland",
                       "Wildwood",
                       "Bush"
                      ],
               "agricultural":["Farming",
                       "Agrarian",
                       "Ranching",
                       "Plantation",
                      ],
               "impervious surfaces":["Impermeable surfaces",
                                      "Non-porous surfaces",
                                      "Sealed surfaces",
                                      "Hard surfaces",
                                      "Concrete surfaces",
                                      "Pavement"
                                      
                      ],
               "low vegetation":["Ground cover",
                                 "Underbrush",
                                 "Shrubs",
                                 "Brush",
                                 "Herbs",
                                 "Grass",
                                 "Sod"
                      ],
               
               "tree":["Forest",
                       "Wood",
                       "tree",
                       "Timber",
                       "Grove",
                       "Sapling"
                      ],
               "car":["Automobile",
                       "Vehicle",
                      "Carriage",
                       "Sedan",
                       "Coupe",
                       "SUV",
                      "Truck",
                      ],
               "clutter":["background",
                       "clutter"
                      ],
               "human":["person",
                       "human"
                      ],
               "background":["background"],
               "dog":["puppy", "dog", "canine"],
               "puppy":["puppy", "dog", "canine"]
              }




def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]
    side = int(sm.shape[0] ** 0.5)
    sm = sm.reshape(1, 1, side, side)
    print(sm)
    print(sm.shape)
    # down sample to smooth results
    down_side = side // down_sample
    sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
    h, w = sm.shape
    sm = sm.reshape(-1)

    sm = (sm - sm.min()) / (sm.max() - sm.min())
    rank = sm.sort(0)[1]
    scale_h = float(shape[0]) / h
    scale_w = float(shape[1]) / w
    print(sm)
    num = min((sm >= t).sum(), sm.shape[0] // 2)
    print(num)
    labels = np.ones(num * 2).astype('uint8')
    print(labels)
    labels[num:] = 0
    points = []

    # positives
    for idx in rank[-num:]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1) # +0.5 to center
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    # negatives
    for idx in rank[:num]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1)
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    return points, labels  


class Text2Seg():
    def __init__(self):
        # set device if GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_sam()
        self.load_groundingDINO()
        self.load_CLIPS()

    def load_groundingDINO(self):
        # load GroundingDINO
        groundingDINO_chcheckpoint = "./Pretrained/groundingdino_swint_ogc.pth"
        self.groundingDINO = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", groundingDINO_chcheckpoint)
        self.groundingDINO.to(self.device)
        self.groundingDINO.eval()

    def load_sam(self):
        #load SAM
        sam_checkpoint = './Pretrained/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        sam_checkpoint = "./Pretrained/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=self.device)        
        self.sam = self.sam.to(self.device)
        self.sam.eval()
    
    def load_CLIPS(self):
        # CLIPS
        self.clip_model, self.clip_preprocess = clips.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
    
    @torch.no_grad()
    def retriev(self, elements, search_text):
        preprocessed_images = [self.clip_preprocess(image).to(self.device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(self.device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = self.clip_model.encode_image(stacked_images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100. * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)

    def get_indices_of_values_above_threshold(values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]

    def predict_dino(self, image_path, text_prompt):
        """
        Use groundingDINO to predict the bounding boxes of the text prompt.
        Then use the bounding boxes to predict the masks of the text prompt.
        """
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # predict
        boxes_lst = []
        for prompt in text_prompt:
            boxes, logits, phrases = predict(
                model=self.groundingDINO,
                image=image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25,
            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            boxes_lst.extend(boxes)
        
        boxes_lst = torch.stack(boxes_lst)
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        masks = ((masks.sum(dim=0)>0)[0]*1).cpu().numpy()
        annotated_frame_with_mask = draw_mask(masks, image_source)

        return masks, annotated_frame_with_mask, annotated_frame


    
    def predict_CLIPS(self, image_path, text_prompt):
        """
        Use the CLIPS model to predict the bounding boxes of the text prompt.
        Then use the bounding boxes to predict the masks of the text prompt.
        """
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # predict
        preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC),Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        cv2_img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        image = preprocess(image).unsqueeze(0).to(self.device)   
        self.sam_predictor.set_image(image_source)
        with torch.no_grad():
            # CLIP architecture surgery acts on the image encoder
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Prompt ensemble for text features with normalization
            text_features = clips.encode_text_with_prompt_ensemble(self.clip_model, text_prompt, self.device)

            # Extract redundant features from an empty string
            redundant_features = clips.encode_text_with_prompt_ensemble(self.clip_model, [""], self.device)

            # CLIP feature surgery with costum redundant features
            similarity = clips.clip_feature_surgery(image_features, text_features, redundant_features)[0]
            
            # Inference SAM with points from CLIP Surgery
            points, labels = clips.similarity_map_to_points(similarity[1:, 0], cv2_img.shape[:2], t=0.8)
            self.sam_predictor.set_image(image_source)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords = np.array(points),
                point_labels = labels,
                multimask_output = False,
            )
            masks = np.array(masks)[0, :, :]
            annotated_frame_with_mask = draw_mask(masks, image_source)
            return masks, annotated_frame_with_mask
        

    def predict_CLIP(self, image_path, text_prompt, points_per_side):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        self.generic_mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=points_per_side)
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # use SAM to generate masks
        segmented_frame_masks = self.generic_mask_generator.generate(image_source)

        # Cut out all masks
        cropped_boxes = []
        for mask in segmented_frame_masks:
            cropped_boxes.append(segment_image(image_source, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))  
        
        indices_lst = []
        for prompt in text_prompt:
            scores = self.retriev(cropped_boxes, prompt)
            indices = get_indices_of_values_above_threshold(scores, 0.05)
            indices_lst.extend(indices)      
            
        segmentation_masks = []
        for seg_idx in np.unique(indices_lst):
            segmentation_mask_image = segmented_frame_masks[seg_idx]["segmentation"]
            segmentation_masks.append(segmentation_mask_image)
        segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)

        annotated_frame_with_mask = draw_mask(segmentation_masks, image_source)

        return segmentation_masks, annotated_frame_with_mask
