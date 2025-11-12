
<mark style='background-color: #24292e'><font color= "white">  # HanYang AI+X : Deep-Learning  </font></mark>  

## Topic : Understanding Diffusion Models — From Noise to Creativity

## Team Member 
  * LG Electroncis Kang, OByoug (Prodct Planning)
  * LG Electronics Ryu, Seungduck (Product Plannning)
  * LG Electronics Park, Hyunsung (Quality Engineering)

## Table of Contents
I. Proposal  
II. Datasets  
III. Methodology  
IV. Evaluation & Analysis  
V. Related Work  
VI. Conclusion  

## I. Proposal
### Motivation
최근 생성형 AI(Generative AI)는 이미지, 텍스트, 오디오 등 다양한 형태의 데이터를 “무(無)”에서 만들어내는 방식을 완전히 바꿔놓았습니다.  
그 중심에는 Diffusion Model(확산 모델) 이 있습니다.  
Diffusion 모델은 기존의 GAN이나 VAE와는 다른 접근을 취하여, 고품질 이미지를 안정적으로 생성할 수 있습니다.  
또한 텍스트-이미지, 오디오, 3D 등 여러 모달리티로 확장 가능한 범용 생성 프레임워크로 각광받고 있습니다.  

### Goal
본 프로젝트의 목표는 다음과 같습니다  
*	Diffusion 모델의 이론적 원리(Forward / Reverse Diffusion) 이해
*	주요 변형 모델들(Latent Diffusion, DDIM, Score SDE 등) 비교
*	Conditioning / Guidance 기법(예: Classifier-Free Guidance) 분석
*	실험 구성 및 평가 지표(FID, IS, Sampling Speed 등) 정리

## II. Datasets
Diffusion 모델 연구에서는 데이터 품질이 모델 성능에 큰 영향을 미칩니다.  
이번 프로젝트에서는 다음과 같은 대표적인 학습 데이터셋을 참고하였습니다.  
* CIFAR-10: 32×32 크기의 소규모 이미지 (기초 실습용)
* CelebA / FFHQ: 고해상도 얼굴 이미지 생성 연구에 자주 사용
* ImageNet (일부 클래스): 대규모 이미지 분류 및 생성 실험용
* LAION-5B: Stable Diffusion이 학습된 공개 대규모 텍스트-이미지 페어 데이터셋 ( [![출처:]laion.ai])
데이터 전처리 과정에서는
•	텍스트-이미지 매칭 품질을 위한 CLIP 필터링,
•	해상도 정규화 및 민감/불법 이미지 필터링 등
윤리적/법적 측면을 고려한 데이터 관리가 필수적으로 수행됩니다.

## Code Sample

import torch  
from diffusers import StableDiffusionPipeline  
from PIL import Image  

#### 1. 모델 로드 (GPU 사용 설정)
#### 사용할 모델 ID 설정 (v1.5는 가장 표준적인 모델 중 하나)
model_id = "runwayml/stable-diffusion-v1-5"  
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  
pipe = pipe.to("cuda")  

#### 2. 프롬프트 설정
prompt = "A high-quality photo of a Mackerel running on the playground"  
negative_prompt = "blurry, low quality, bad art, (worst quality:1.4)"  

#### 3. 이미지 생성 실행
#### guidance_scale: 프롬프트를 얼마나 따를지 정하는 값 (7~8.5가 일반적)
#### num_inference_steps: 노이즈 제거 단계 수 (50 정도가 표준)
with torch.autocast("cuda"):  
    image = pipe(  
        prompt,   
        negative_prompt=negative_prompt,   
        guidance_scale=7.5,   
        num_inference_steps=50  
    ).images[0]    

#### 4. 이미지 저장
image.save("mackerel.png")  
print("이미지 생성 완료! 'Mackerel.png'로 저장되었습니다.")  

## 생성 Image
[![mackerel](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)

