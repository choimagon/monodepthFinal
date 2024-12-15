## 단안 카메라로 3D 재구성 프로젝트

### 데이터셋 
데이터셋 : https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
![image](https://github.com/user-attachments/assets/826f103d-4347-4dc4-b2e9-66f3009636a0)

### 폴더 구조
```
MONODEPTHFINAL/
│
├── FinalTrain/
│   ├── data/             -> 데이터셋 다운로드한 후 여기에서 배치
│   ├── saved_modelsV2/   -> 벨리데이션이 낮아질수록 모델 저장
│   ├── V2/               -> 각 에초치마다 벨리데이션 시각화
│   ├── image_processor.py  -> 개량 DoG 코드
│   └── runTrain.py         -> train 코드
│
├── FinalWeb/
│   ├── modelpth/    -> 학습 후 가중치는 여기 둘것
│   ├── static/          
│   ├── templates/
│   ├── app.py          -> 모델 가중치 여기서 바꿀수있음
│   ├── image_processor.py -> 개량 DoG 코드
│   ├── model.py           -> 모델 코드 
│   ├── process_image.py 
│   └── visualizer.py
│
└── .gitignore
```

### 사용방법
> #### 환경설정
> 기본 환경 cuda 11.8 / 리눅스 20.04 
> 1. ```conda create -n myenv python=3.8 -y```
> 2. ```conda activate myenv```
> 3. ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
> 4. ```pip install pandas opencv-python numpy tqdm matplotlib```
> 5. ```pip install flask```
> 6. ```git clone https://github.com/choimagon/monodepthFinal.git```
<br>
> #### 모델 학습하기
> 1. FinalTrain 폴더를 들어간다
> 2. data 파일을 위 데이터셋 주소에서 다운받아서 위 폴더구조에 맞게 배치한다.
> 3. ```python runTrain.py```를 실행한다.
<br> vram 31기가 잡아먹음.

> #### 웹사이트 구동
> 학습된 가중치 : https://drive.google.com/file/d/1DCw0LiPB0MNm969YM7eKAeyzdPZcU7_G/view?usp=drive_link
> 1. FinalWeb 폴더를 들어간다
> 2. 학습된 가중치 modelpth파일에 넣기
> 3. ```python app.py``` 를 실행한다.
<br>이미지를 업로드 한 후 생성 버튼을 누른다.
