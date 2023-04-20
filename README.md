# Text2Seg


[Jielu Zhang](), [Zhongliang Zhou](), [Gengchen Mai](), [Lan Mu](), [Mengxuan Hu](), [Sheng Li]()

![Text2Seg design](framework.png?raw=true)

Text2Seg is a pipeline that combined multiple Vision Foundation Models to perform semantic segmentation.

## Installation

1. Create an new conda environment 

```
conda create --name text2seg python==3.8
conda activate text2seg
pip install chardet ftfy regex tqdm
```

2. Install Pytorch version that fit you driver(tested on pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3).

3. Install Segment Anything and download weights:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

4. Install Grounding DINO

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

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

```
