## 단안 카메라로 3D 재구성 프로젝트

### 데이터셋 
데이터셋 : https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
![image](https://github.com/user-attachments/assets/826f103d-4347-4dc4-b2e9-66f3009636a0)

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
│   ├── modelpth/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   ├── image_processor.py -> 개량 DoG 코드
│   ├── model.py           -> 모델 코드
│   ├── process_image.py 
│   └── visualizer.py
│
└── .gitignore
```
