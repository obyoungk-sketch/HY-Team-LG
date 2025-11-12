
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
Ⅶ. Code Sample  

### I. Proposal
#### Motivation
최근 생성형 AI(Generative AI)는 이미지, 텍스트, 오디오 등 다양한 형태의 데이터를 “무(無)”에서 만들어내는 방식을 완전히 바꿔놓았습니다.  
그 중심에는 Diffusion Model(확산 모델) 이 있습니다.  
Diffusion 모델은 기존의 GAN이나 VAE와는 다른 접근을 취하여, 고품질 이미지를 안정적으로 생성할 수 있습니다.  
또한 텍스트-이미지, 오디오, 3D 등 여러 모달리티로 확장 가능한 범용 생성 프레임워크로 각광받고 있습니다.  

#### Goal
본 프로젝트의 목표는 다음과 같습니다  
*	Diffusion 모델의 이론적 원리(Forward / Reverse Diffusion) 이해
*	주요 변형 모델들(Latent Diffusion, DDIM, Score SDE 등) 비교
*	Conditioning / Guidance 기법(예: Classifier-Free Guidance) 분석
*	실험 구성 및 평가 지표(FID, IS, Sampling Speed 등) 정리

### II. Datasets
Diffusion 모델 연구에서는 데이터 품질이 모델 성능에 큰 영향을 미칩니다.  
이번 프로젝트에서는 다음과 같은 대표적인 학습 데이터셋을 참고하였습니다.  
* CIFAR-10: 32×32 크기의 소규모 이미지 (기초 실습용)
* CelebA / FFHQ: 고해상도 얼굴 이미지 생성 연구에 자주 사용
* ImageNet (일부 클래스): 대규모 이미지 분류 및 생성 실험용
* LAION-5B: Stable Diffusion이 학습된 공개 대규모 텍스트-이미지 페어 데이터셋  [출처:laion.ai](https://laion.ai/?utm_source=chatgpt.com)   
데이터 전처리 과정에서는  
* 텍스트-이미지 매칭 품질을 위한 CLIP 필터링
* 해상도 정규화 및 민감/불법 이미지 필터링 등
윤리적/법적 측면을 고려한 데이터 관리가 필수적으로 수행됩니다.  

### III. Methodology
#### A. 직관적 개요
Diffusion 모델의 핵심 아이디어는 데이터를 점진적으로 파괴(노이즈 추가) 했다가,  
그 과정을 역으로 복원(노이즈 제거) 하는 확률적 생성 과정에 있습니다.  
즉, “데이터를 망가뜨리는 법을 배우고, 그 과정을 거꾸로 되돌리며 새로운 데이터를 만드는 모델” 입니다.
________________________________________  
🔹 Step 1: Forward Diffusion (노이즈 추가 과정)  
원본 데이터 x0x_0x0 에 시간 단계 ttt 를 따라 점차적으로 노이즈를 추가합니다.  
xt=1−βt⋅xt−1+βt⋅ϵ,ϵ∼N(0,I)x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)xt=1−βt⋅xt−1+βt⋅ϵ,ϵ∼N(0,I)   
ttt 가 증가할수록 데이터는 점점 더 무작위적이 되어,최종적으로는 완전한 노이즈 xTx_TxT 가 됩니다.  
________________________________________  
🔹 Step 2: Reverse Diffusion (노이즈 제거 과정)  
모델은 이 과정을 거꾸로 수행하여 노이즈로부터 원본 이미지를 복원합니다.  
즉, 완전히 무작위한 입력 xTx_TxT 에서 점진적으로 노이즈를 제거하며  
점점 더 사실적인 이미지를 만들어내는 구조입니다.  
________________________________________  
🔹 Step 3: Learning Objective  
Diffusion 모델은 각 단계에서 예측한 노이즈 ϵθ(xt,t)\epsilon_\theta(x_t, t)ϵθ(xt,t) 가  
실제 노이즈 ϵ\epsilonϵ 과 얼마나 가까운지를 최소화하는 손실 함수를 사용합니다.  
L=Ext,t,ϵ[∥ϵ−ϵθ(xt,t)∥2]L = \mathbb{E}_{x_t, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]L=Ext,t,ϵ[∥ϵ−ϵθ(xt,t)∥2]   
이 단순한 MSE 기반 손실 덕분에 학습이 안정적이고 효율적으로 수행됩니다.  
________________________________________  
🔹 Step 4: Latent Diffusion Model (LDM)  
Stable Diffusion은 픽셀 공간 대신 잠재 공간(latent space) 에서 확산 과정을 수행합니다.  
이를 통해  
* 연산량을 크게 줄이고,
* 빠른 이미지 생성을 가능하게 하며,
* 고해상도 이미지까지 효율적으로 다룰 수 있습니다

<img width="413" height="553" alt="image" src="https://github.com/user-attachments/assets/bee17973-6e60-40c4-9993-95247c537b04" />

### IV. Evaluation & Analysis
Diffusion Model의 성능 평가는 주로 다음 지표를 사용합니다.  
<img width="482" height="78" alt="image" src="https://github.com/user-attachments/assets/0c628311-92b2-4b16-b1b5-5295a89760ac" />

🔹 비교 분석 요약  
Diffusion 모델은 안정성과 품질 면에서 GAN을 능가하지만, 여전히 샘플링 속도와 연산 효율성은 주요 개선 과제입니다  
<img width="539" height="112" alt="image" src="https://github.com/user-attachments/assets/7948ea14-b7e5-4c0c-b612-738bb4f68363" />

### V. Related Work (핵심 논문 및 리소스)
•	Ho et al. (2020) — Denoising Diffusion Probabilistic Models (DDPM)  
https://arxiv.org/abs/2006.11239  
•	Song et al. (2020) — DDIM: Denoising Diffusion Implicit Models  
https://arxiv.org/abs/2010.02502  
•	Song et al. (2021) — Score-Based Generative Modeling through SDEs  
https://arxiv.org/abs/2011.13456  
•	Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion의 기반 논문)  
https://arxiv.org/abs/2112.10752  
🔹 오픈소스 및 실습 리소스  
•	CompVis / Stable Diffusion (GitHub): https://github.com/CompVis/stable-diffusion  
•	Hugging Face Diffusers: https://huggingface.co/docs/diffusers  
•	PyTorch / TensorFlow 기반 Diffusion 구현 예제  

### VI. Conclusion
Diffusion Models는 단순히 이미지를 생성하는 기술을 넘어,  
AI가 데이터를 이해하고 재구성하는 방식 자체를 혁신적으로 바꾼 모델입니다.  
GAN의 불안정한 학습 구조를 극복하고,더 자연스럽고 사실적인 이미지를 만들어내며,  
텍스트, 오디오, 3D 생성 등으로 확장 가능한 범용적 생성 구조로 자리잡았습니다.  

향후 연구 방향  
* Efficient Sampling: 생성 속도를 개선하기 위한 최적화 연구 (예: DDIM, PNDM)
* Multi-modal Diffusion: 텍스트·오디오·3D 등 다양한 입력 조합 확장
* Controllable Generation: ControlNet 등 사용자 의도 기반 제어 가능 모델  
“Diffusion Model은 노이즈에서 시작해 현실을 그려내는, 현대 AI의 가장 창의적이고 시각적인 알고리즘이다.”

### Ⅶ. Code Sample

import torch  
from diffusers import StableDiffusionPipeline  
from PIL import Image  

##### 1. 모델 로드 (GPU 사용 설정)
##### 사용할 모델 ID 설정 (v1.5는 가장 표준적인 모델 중 하나)
model_id = "runwayml/stable-diffusion-v1-5"  
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  
pipe = pipe.to("cuda")  

##### 2. 프롬프트 설정
prompt = "A high-quality photo of a Mackerel running on the playground"  
negative_prompt = "blurry, low quality, bad art, (worst quality:1.4)"  

##### 3. 이미지 생성 실행
##### guidance_scale: 프롬프트를 얼마나 따를지 정하는 값 (7~8.5가 일반적)  
##### num_inference_steps: 노이즈 제거 단계 수 (50 정도가 표준)  
with torch.autocast("cuda"):  
    image = pipe(  
        prompt,   
        negative_prompt=negative_prompt,   
        guidance_scale=7.5,   
        num_inference_steps=50  
    ).images[0]    

##### 4. 이미지 저장
image.save("mackerel.png")  
print("이미지 생성 완료! 'Mackerel.png'로 저장되었습니다.")  

### 생성 Image
[![mackerel](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)

