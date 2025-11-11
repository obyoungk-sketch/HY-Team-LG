# HanYang AI+X : Deep-Learning

## Topic : Stable Diffusion Model

## Team Member 
  * LG Electroncis Kang, OByoug (Prodct Planning)
  * LG Electronics Ryu, Seungduck (Product Plannning)
  * LG Electronics Park, Hyunsung (Quality Engineering)


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

### Image
[![mackerel](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)](https://github.com/obyoungk-sketch/HY-Team-LG/blob/master/assets/mackerel.png)

