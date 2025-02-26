# Text2Seg v0.1


[Jielu Zhang](https://geography.uga.edu/directory/people/jielu-zhang), [Zhongliang Zhou](https://www.zhongliangzhou.com/), [Gengchen Mai](https://gengchenmai.github.io/), [Lan Mu](https://geography.uga.edu/directory/people/lan-mu), [Mengxuan Hu](https://www.linkedin.com/in/hu-mengxuan-823675263/), [Sheng Li](https://sheng-li.org/)

![Text2Seg design](framework.png?raw=true)

Text2Seg is a pipeline that combined multiple Vision Foundation Models to perform semantic segmentation.

## :fire: UPDATE:
- **`2023/06/07`**: Update the codebase to solve some known problems with GroundingDINO.

## Installation

1. Create an new conda environment 

```
conda create --name text2seg python==3.8
conda activate text2seg
pip install chardet ftfy regex tqdm
mkdir Pretrained
```

2. Install Pytorch version that fit you driver(tested on pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3).

3. Install Segment Anything and download weights:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
cd Pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../
```

4. Install Grounding DINO

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip3 install -q -e .
cd ..
cd Pretrained
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../
```

5. Download CLIP Surgery repository
```
git clone https://github.com/xmed-lab/CLIP_Surgery.git
```

6. Install CLIP repository
```
pip install git+https://github.com/openai/CLIP.git
```

## <a name="GettingStarted"></a>Getting Started

You can test the Text2Seg on demo.ipynb notebook. 


## Citing Text2Seg

If you find Text2Seg useful, please use the following BibTeX entry.

```
@inproceedings{zhang2024text2seg,
  title={Text2Seg: Zero-shot Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models},
  author={Zhang*, Jielu and Zhou*, Zhongliang and Mai, Gengchen and Hu, Mengxuan and Guan, Zihan and Li, Sheng and Mu, Lan},
  booktitle={Proceedings of the 7th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery},
  pages={63--66},
  year={2024}
}
```
