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

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# CLIP Surgery
import CLIP_Surgery.clip as clips

# CLIP original
import clip

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry

from huggingface_hub import hf_hub_download

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  

# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.35, text_threshold = 0.25):
    boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
    return annotated_frame, boxes

def segment_DINO(image, sam_predictor, boxes):
    sam_predictor.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
    return masks.cpu()

def segment_CLIPS(image, image_source, sam_predictor, clip_model, text_prompts):
    all_texts = ['background', 
                 'building', 
                 'road', 
                 'water', 
                 'barren', 
                 'forest',
                 'agricultural',
                 'impervious surfaces', 
                 'low vegetation', 
                 'tree', 
                 'car', 
                 'clutter', 
                 'human',  
                 'dog',
                 'puppy',
                ]
    all_texts.append(text_prompts[0])
    # target_texts = ['Building', 'Impervious surfaces', 'Low vegetation', 'Tree', 'Car', 'Clutter/background']
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    cv2_img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
    sam_predictor.set_image(image_source)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # clip architecture surgery acts on the image encoder
        image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # prompt ensemble for text features with normalization
        text_features = clips.encode_text_with_prompt_ensemble(clip_model, all_texts, device)

        # apply feature surgery, no batch
        similarity = clips.clip_feature_surgery(image_features, text_features)[0]

        # inference SAM with points from CLIP Surgery
        index = all_texts.index(text_prompts[0])
        points, labels = clips.similarity_map_to_points(similarity[1:, index], cv2_img.shape[:2], t=0.8)
        transformed_points = sam_predictor.transform.apply_coords_torch(torch.FloatTensor(points).to(device), image.shape[:2])
        transformed_points = torch.unsqueeze(transformed_points, 0)
        transformed_labels = torch.unsqueeze(torch.FloatTensor(labels).to(device), 0)
        masks, _, _ = sam_predictor.predict_torch(
          point_coords = transformed_points,
          point_labels = transformed_labels,
          boxes = None,
          multimask_output = False,
          )
        return masks.cpu(), points
      
def save_anno(image, anns, points_per_side, dataset="None", method="SAM_Only"):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask_all = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        
        m = [m for x in range(3)]
        m = np.swapaxes(m,0,2)
        m = np.swapaxes(m,0,1)
        img = img*m
        # ax.imshow(np.dstack((img, m*0.35)))
        mask_all.append(img)
        # print(img.shape)
    mask_all = np.array(mask_all).sum(axis=0)
    save_img = image/2+mask_all*255
    save_img = Image.fromarray(save_img.astype(np.uint8))
    save_img.save(f"./{dataset}/{method}_{points_per_side}.png")
    
def show_anns(image, anns):
    plt.figure(figsize=(20,20))
    plt.imshow(image, alpha=0.5)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*1)))
    plt.axis('off')
    # plt.savefig(f"./{dataset}/SAM_only_{points_per_side}.png", bbox_inches='tight')
    plt.show()  
    
def draw_mask(mask, image, grounding_dino, CLIPS, CLIP, random_color=False):
    color_DINO = np.array([255/255, 51/255, 51/255, 0.6])
    color_CLIPS = np.array([0/255, 128/255, 255/255, 0.6])
    color_CLIP = np.array([255/255, 0/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 
    if grounding_dino and CLIPS and CLIP:
        mask_G_dino = 1*torch.logical_or(mask == 1, mask > 3)
        mask_CLIPS = 1*(mask == 2)
        mask_CLIP = 1*(mask == 3)
        
        mask_image_G_dino = mask_G_dino.reshape(h, w, 1) * color_DINO.reshape(1, 1, -1)
        mask_image_CLIPS = mask_CLIPS.reshape(h, w, 1) * color_CLIPS.reshape(1, 1, -1)
        mask_image_CLIP = mask_CLIP.reshape(h, w, 1) * color_CLIPS.reshape(1, 1, -1)
        
        mask_image_merged = mask_image_G_dino + mask_image_CLIPS + mask_image_CLIP
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image_merged.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    if grounding_dino and CLIPS:
        mask_G_dino = 1*torch.logical_or(mask == 1, mask == 3)
        mask_CLIPS = 1*(mask == 2)
        mask_image_G_dino = mask_G_dino.reshape(h, w, 1) * color_DINO.reshape(1, 1, -1)
        mask_image_CLIPS = mask_CLIPS.reshape(h, w, 1) * color_CLIPS.reshape(1, 1, -1)
        mask_image_merged = mask_image_G_dino + mask_image_CLIPS
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image_merged.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    elif grounding_dino:
        mask_G_dino = 1*(mask == 1)
        mask_image_G_dino = mask_G_dino.reshape(h, w, 1) * color_DINO.reshape(1, 1, -1)
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image_G_dino.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)), mask_G_dino    
    
    elif CLIPS:
        mask_CLIPS = 1*(mask == 2)
        mask_image_CLIPS = mask_CLIPS.reshape(h, w, 1) * color_CLIPS.reshape(1, 1, -1)
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image_CLIPS.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)) 
    
    elif CLIP:
        mask_CLIP = 1*(mask == 3)
        mask_image_CLIP = mask_CLIP.reshape(h, w, 1) * color_CLIPS.reshape(1, 1, -1)
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image_CLIP * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))   
    
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = image
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.shape[:2], (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

@torch.no_grad()
def retriev(elements, search_text):
    preprocessed_images = [clip_preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = clip_model.encode_image(stacked_images)
    text_features = clip_model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def segment_CLIP(masks, image, text_prompts):
    # Cut out all masks
    cropped_boxes = []
    for mask in masks:
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))  
    
    indices_lst = []
    for prompt in clip_prompt[text_prompts[0]]:
        scores = retriev(cropped_boxes, prompt)
        indices = get_indices_of_values_above_threshold(scores, 0.05)
        indices_lst.extend(indices)        
        
    segmentation_masks = []

    for seg_idx in np.unique(indices_lst):
        segmentation_mask_image = masks[seg_idx]["segmentation"]
        segmentation_masks.append(segmentation_mask_image)
    segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)
     
    return segmentation_masks

def VFM_Segmentation(text_prompts, image, image_source, grounding_dino=False, CLIPS=False, CLIP=False, points_per_side=16, Outputs="None"):
    # perform grounding dino for detection
    
    # try:
    #     os.mkdir(Outputs)
    # except OSError as error:
    #     print("Folder Exsist") 
    if grounding_dino:
        bounding_box_list = []
        for prompt in text_prompts:
            annotated_frame, detected_boxes = detect(image, image_source, text_prompt=prompt, model=groundingdino_model)
            # print(prompt) 
            # print(np.array(detected_boxes).shape)
            # print(np.array(detected_boxes))
            bounding_box_list.extend(np.array(detected_boxes))
        bounding_box_list = np.array(bounding_box_list)
        print(bounding_box_list)
        
        
    if grounding_dino and CLIPS and CLIP:
        print("Grounding DINO + CLIPS + SAM + CLIP")
        segmented_frame_masks_dino = segment_DINO(image_source, sam_predictor, boxes=torch.as_tensor(bounding_box_list))
        segmented_frame_masks_dino = (segmented_frame_masks_dino.sum(dim=0)>0)[0]*1
        annotated_frame_with_mask_dino = draw_mask(segmented_frame_masks_dino, image_source, grounding_dino, CLIPS, CLIP)
        
        segmented_frame_masks_CLIPS, _ = segment_CLIPS(image, image_source, sam_predictor, clip_model, text_prompts)
        segmented_frame_masks_CLIPS = segmented_frame_masks_CLIPS[0][0]*2
                                
        generic_mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
        segmented_frame_masks_CLIP = generic_mask_generator.generate(image_source)
        segmented_frame_masks_CLIP = segment_CLIP(segmented_frame_masks_CLIP, image_source, text_prompts)
        
        
        segmented_frame_masks_all = segmented_frame_masks_CLIPS + segmented_frame_masks_dino + segmented_frame_masks_CLIP
        annotated_frame_with_mask_all = draw_mask(segmented_frame_masks_all, image_source,grounding_dino, CLIPS, CLIP)
        img = Image.fromarray(annotated_frame_with_mask_all)
        img = img.resize((500, 500))
        image_filename = f"./{Outputs}/{text_prompts[0]}_Grounding_DINO_CLIPS_SAM_CLIP.png"
        img.save(image_filename)
        return segmented_frame_masks_all, annotated_frame_with_mask_all
    
    
    if grounding_dino and CLIPS:
        print("Grounding DINO + CLIPS + SAM")
        segmented_frame_masks_dino = segment_DINO(image_source, sam_predictor, boxes=torch.as_tensor(bounding_box_list))
        segmented_frame_masks_dino = (segmented_frame_masks_dino.sum(dim=0)>0)[0]*1
        annotated_frame_with_mask_dino = draw_mask(segmented_frame_masks_dino, image_source, grounding_dino, CLIPS, CLIP)
        
        segmented_frame_masks_CLIPS, _ = segment_CLIPS(image, image_source, sam_predictor, clip_model, text_prompts)
        segmented_frame_masks_CLIPS = segmented_frame_masks_CLIPS[0][0]*2
                                
        segmented_frame_masks_all = segmented_frame_masks_CLIPS + segmented_frame_masks_dino
        annotated_frame_with_mask_all = draw_mask(segmented_frame_masks_all, image_source,grounding_dino, CLIPS, CLIP)
        img = Image.fromarray(annotated_frame_with_mask_all)
        img = img.resize((500, 500))
        image_filename = f"./{Outputs}/{text_prompts[0]}_Grounding_DINO_CLIPS_SAM.png"
        img.save(image_filename)        
        return segmented_frame_masks_all, annotated_frame_with_mask_all

    elif grounding_dino: 
        print("Grounding DINO + SAM")
        if len(bounding_box_list)!=0:
            
            segmented_frame_masks = segment_DINO(image_source, sam_predictor, boxes=torch.as_tensor(bounding_box_list))
            segmented_frame_masks = (segmented_frame_masks.sum(dim=0)>0)[0]*1
            annotated_frame_with_mask, mask_image_pil = draw_mask(segmented_frame_masks, image_source,grounding_dino, CLIPS, CLIP)
            # img = Image.fromarray(annotated_frame_with_mask)
            # img = img.resize((500, 500))
            # image_filename = f"./{Outputs}/{text_prompts[0]}_Grounding_DINO_SAM.png"
            # img.save(image_filename)        
            return segmented_frame_masks, annotated_frame_with_mask, bounding_box_list
        else:
            return torch.zeros(image_source[:,:,0].shape), torch.zeros(image_source[:,:,0].shape)
    
    elif CLIPS:
        print("CLIP Surgery + SAM")
        segmented_frame_masks, _ = segment_CLIPS(image, image_source, sam_predictor, clip_model, text_prompts)
        annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0]*2, image_source, grounding_dino, CLIPS, CLIP)
        img = Image.fromarray(annotated_frame_with_mask)
        img = img.resize((500, 500))
        image_filename = f"./{Outputs}/{text_prompts[0]}_CLIPS_SAM.png"
        img.save(image_filename)        
        return segmented_frame_masks, annotated_frame_with_mask
    
    elif CLIP:
        print("SAM + CLIP")
        generic_mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
        segmented_frame_masks = generic_mask_generator.generate(image_source)
        segmented_frame_masks_clip = segment_CLIP(segmented_frame_masks, image_source, text_prompts)
        if segmented_frame_masks_clip.ndim>1:
            annotated_frame_with_mask = draw_mask(segmented_frame_masks_clip, image_source, grounding_dino, CLIPS, CLIP)
            img = Image.fromarray(annotated_frame_with_mask)
            img = img.resize((500, 500))
            image_filename = f"./{Outputs}/{text_prompts[0]}_SAM_CLIP.png"
            img.save(image_filename)        
            return segmented_frame_masks_clip, annotated_frame_with_mask
        else:
            print("CLIP find no item given the input prompt!")
            img = Image.fromarray(image_source)
            img = img.resize((500, 500))
            image_filename = f"./{Outputs}/{text_prompts[0]}_SAM_CLIP.png"
            img.save(image_filename)        
            return segmented_frame_masks_clip, segmented_frame_masks            
    
    else:
        print("Generic SAM using point sample")
        generic_mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
        segmented_frame_masks = generic_mask_generator.generate(image_source)
        
        return segmented_frame_masks
    
    
# DINO
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

#SAM
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)

# CLIPS
clip_model, clip_preprocess = clips.load("ViT-B/16", device=device)
clip_model.eval()