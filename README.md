<div align="center">
<h1>Zero-Shot Referring Expression Segmentation based on CLIP</h1>
<h3>전이학습 (송지우, 여주안, 차순우) 팀</h3>
Zero-Shot Referring Expression Segmentation을 해결하기 위한 프로젝트 코드 Baseline
</div>

# 🚀 Quick Start
## 1. Project Import
```
> git clone https://github.com/juanyeo/Zeref-Segmentation.git
```
## 2. Create Anaconda Environment & Install Packages
```
# 환경 생성 및 활성화
> conda create -n zeref python=3.8 -y
> conda activate zeref

# Package 설치 (! 명령어 그대로 사용할 것)
> conda install pytorch=1.11.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
> conda install cython scipy shapely transformers h5py scikit-image matplotlib
> conda install -c conda-forge timm pycocotools
> pip install fvcore
> pip install ninja
> pip install opencv-python

# Detectron2 설치 1. git 프로젝트 import (! 프로젝트 홈 디렉토리에 할 것)
> git clone https://github.com/facebookresearch/detectron2.git
# Detectron2 설치 2. import 된 폴더로 이동
> cd detectron2
# Detectron2 설치 3. Detectron2 수동 빌드 (gcc, g++ 필요)
> python setup.py build_ext --inplace
# Detectron2 설치 4. 홈 디렉토리로 복귀
> cd ..
# Detectron2 설치 5. 패키지 설치
> python -m pip install -e detectron2
```
## 3. Dataset Import (3-5 시간 소요)
폴더 구조는 dataset/DATASET.md 참조
```
# 프로젝트 내 dataset 폴더로 이동
> cd dataset

# COCO 이미지 다운로드
> wget http://images.cocodataset.org/zips/train2014.zip
> unzip train2014.zip

# RefCOCO 데이터 다운로드
> wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
> unzip refcoco.zip

# RefCOCO+ 데이터 다운로드
> wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
> unzip refcoco+.zip

# RefCOCOg 데이터 다운로드
> wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
> unzip refcocog.zip

# gRefCOCO 데이터 다운로드 (수동 설치)
1. 아래 링크에서 grefs 다운로드 2. 압축 풀기 3. dataset 폴더 하위로 이동 4. grefs -> grefcoco로 rename
```
[gRefCOCO 다운로드 링크](https://entuedu-my.sharepoint.com/personal/liuc0058_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fliuc0058%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopensource%2FGRES%2Fdataset&ga=1)

## 4. Run Project
`data_loader` 폴더의 `register_refcoco.py` 파일의 95번 라인 `_root` 절대 경로 값을 실행 환경에 맞춰 변경
```
# 모델 실행
> python run.py
```
