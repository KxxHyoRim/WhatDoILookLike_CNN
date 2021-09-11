# WhatDoILookLike_CNN

#### Android 어플리케이션 '나의 얼굴은'의 동물상 판별 딥러닝 모델입니다. </br>
#### 여러 딥러닝 모델로 각각 여러번의 시도 결과 가장 성능이 좋게 나온 CNN모델의 정확도는 72%입니다.

<img src = "https://user-images.githubusercontent.com/59546818/132958071-c046cd15-8dd3-492e-abdb-846f3f6edd2d.png" width="80%" height="80%">
<img src = "https://user-images.githubusercontent.com/59546818/132957909-d2693043-e9aa-47de-bb7c-105fba082458.png" width="80%" height="80%">

</br></br>

#### 1. 이미지 크롤링
연예인 이미지를 크롤링 해옵니다.</br></br>

#### 2. 이미지 전처리
정확한 얼굴상 판별을 위해 얼굴부분의 사진을 CROP하여 저장합니다.</br>
얼굴 CROP을 위한 기술은 Google Vison API를 활용했습니다.</br></br>

#### 3. 모델 설계
모델 설계는 Tensorflow/Keras를 활용했으며 CNN모델을 구축하였습니다.

<img src = "https://user-images.githubusercontent.com/59546818/132957916-a06c6253-a1c1-46b2-9765-383779f9af21.png" >
<img src = "https://user-images.githubusercontent.com/59546818/132957921-26e0d379-ff30-4cc2-a62d-915b7780efe5.png" >
</br></br>

#### 4. 모델 변환
해당 딥러닝 모델을 안드로이드에서 활용하기 위해서는 Tesorflow-Lite 모델 형식을 사용해야 합니다(.tflite).</br>
'WhatDoILookLike_CNN.py'에서 keras로 생성된 모델은 Tensorflow 모델(.h5)이기 떄문에 변환과정을 거져 tflite를 생성합니다.</br></br>

<img src = "https://user-images.githubusercontent.com/59546818/132957923-f2c4ce57-a58a-4ab5-8249-696061e6ab56.png">
