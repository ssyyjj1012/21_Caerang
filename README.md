# 2021년도 씨애랑 소프트웨어전시회 작품 

20195178 서영재



## 프로젝트 개요

딥러닝 기술 활용에 있어 가장 필수적인 해결 과제는 학습에 필요한 데이터를 충분히 확보하는 것입니다. 

데이터를 확보하는 과정에서 개인정보 보호 문제나 사건 발생률이 높지 않아 학습에 충분한 데이터 확보가 어렵습니다. 

GAN을 통해 데이터를 증강함으로써 데이터 부족 문제를 해결하고, 해결하고자 하는 문제의 기계학습 성능을 향상시키고자 합니다. 

의료 데이터뿐만 아니라 상대적으로 데이터가 적어 학습이 어려운 산업까지 확장함으로써 인공지능산업에 선점을 기여할 수 있을 것입니다.
_________________________

## 프로젝트 목표 및 내용

딥러닝 학습에서 이미지 데이터가 부족한 경우 원본 데이터를 회전 확대 후 자르기 상하좌우 반전 등 여러 선형적인 방법을 통해 데이터를 증강시키는 방법이 활용되고 있습니다. 

하지만 이 방법은 추가적인 정보를 얻기에는 한계가 있습니다. 

이에 Generative adversarial networks(GAN) 기반 데이터 증강 기법을 통해 가상의 데이터를 생성하여 궁극적으로 해결하고자 하는 문제의 기계학습 성능 향상이 하려 합니다.

____________________________

## DATASET

캐글 대회에서 제공한 데이터

[**MOAI 2021 Body Morphometry AI Segmentation Online Challenge**](https://www.kaggle.com/c/body-morphometry-kidney-and-tumor/data)

________________________________
## Model 

1.FPN (Pretrained)

2.U-net (our Model)


** Augumentation의 유무에 대해서도 성능 분석
______________________________
## GAN

- Pix2Pix

     make new data to improve CNN acc.
     
<img src = "https://user-images.githubusercontent.com/52689953/143183478-dd315705-b8ee-4279-a13d-05401d7db547.png" width="200" height="200"/>

^ 실제데이터

<img src = "https://user-images.githubusercontent.com/52689953/143183525-63094f26-bba2-4b24-a341-d38df96e7115.png" width="200" height="200"/><img src = "https://user-images.githubusercontent.com/52689953/143183542-e785ab54-6f2d-458d-8112-18764645581a.png" width="200" height="200"/>


^ GAN의 데이터 증강기법을 통해 생성한 인공데이터

____________________

## 기대효과 및 활용 방안

부족한 데이터를 GAN을 통해 증강시키는 것 자체가 유의미한 과정이며, 이를 통해 인공지능 성능들을 높일 수 있습니다. 

또한 조금만 변형한다면 의료 데이터뿐만 아니라 다양한 데이터 분야에서의 데이터 증강이 가능할 것입니다. 

그렇다면 상대적으로 데이터가 적어 학습이 어려운 산업까지 확장함으로써 인공지능산업에 선점을 기여할 수 있습니다.


_______________________
## 코드
[model ](https://github.com/ssyyjj1012/21_Caerang/tree/main/model "코드 링크")

![image](https://user-images.githubusercontent.com/52689953/142764573-33a35d68-5d6d-47d4-8e25-2b3b4936779c.png)

본래 신장 위치 / 본래 신장암 위치 / 신장 위치 예측 / 신장암 위치 예측 / 신장 및 신장암 위치에 대한 정답


### GAN으로 가짜 이미지 생성 후 모델 학습 결과 

![image](https://user-images.githubusercontent.com/52689953/142764649-94091623-2acf-476d-b85c-e5087da1ef5d.png)


___________________

## 향후 개선점
- 성능 지표 
