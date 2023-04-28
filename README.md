# Text2Seg


[Jielu Zhang](https://geography.uga.edu/directory/people/jielu-zhang), [Zhongliang Zhou](https://www.zhongliangzhou.com/), [Gengchen Mai](https://gengchenmai.github.io/), [Lan Mu](https://geography.uga.edu/directory/people/lan-mu), [Mengxuan Hu](https://www.linkedin.com/in/hu-mengxuan-823675263/), [Sheng Li](https://sheng-li.org/)

![Text2Seg design](framework.png?raw=true)

Text2Seg is a pipeline that combined multiple Vision Foundation Models to perform semantic segmentation.

## Installation

1. Create an new conda environment 

```
git clone https://github.com/Douglas2Code/Text2Seg.git
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

@article{zhang2023text2seg,
  title={Text2Seg: Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models},
  author={Zhang, Jielu and Zhou, Zhongliang and Mai, Gengchen and Mu, Lan and Hu, Mengxuan and Li, Sheng},
  journal={arXiv preprint arXiv:2304.10597},
  year={2023}
}
