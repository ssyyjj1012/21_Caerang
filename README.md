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

____________________

## 기대효과 및 활용 방안

부족한 데이터를 GAN을 통해 증강시키는 것 자체가 유의미한 과정이며, 이를 통해 인공지능 성능들을 높일 수 있습니다. 

또한 조금만 변형한다면 의료 데이터뿐만 아니라 다양한 데이터 분야에서의 데이터 증강이 가능할 것입니다. 

그렇다면 상대적으로 데이터가 적어 학습이 어려운 산업까지 확장함으로써 인공지능산업에 선점을 기여할 수 있습니다.


_______________________
## 코드
[링크](https://github.com/ssyyjj1012/code/blob/main/code/check_code.md "코드 링크")
![image](https://user-images.githubusercontent.com/52689953/142764573-33a35d68-5d6d-47d4-8e25-2b3b4936779c.png)

본래 콩팥 위치 / 본래 종양 위치 / 콩팥 예측 / 종양 예측 / 콩팥 밑 종양 위치 정답

#### GAN으로 가짜 이미지 생성 후 모델 학습 결과 

![image](https://user-images.githubusercontent.com/52689953/142764649-94091623-2acf-476d-b85c-e5087da1ef5d.png)

좋은 성능으로 예측했음을 알 수 있었습니다.
___________________

## 향후 개선점
- test iou 부분 수정
- testset에 알뷰 넣어서 실험
- pretrain 모델이 아닌 u-net 구조로 모델 구현
