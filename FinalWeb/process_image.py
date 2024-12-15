from model_module import load_model, predict_image, save_prediction

if __name__ == '__main__':
    model_path = 'modelpth/best_model_epoch_50.pth'
    image_path = 'data/nyu2_test/00000_colors.png'
    output_path = 'output/predicted_depth.png'

    # 모델 로드
    model, device = load_model(model_path)

    # 이미지 처리
    predicted_output = predict_image(model, device, image_path)

    # 결과 저장
    save_prediction(predicted_output, output_path)

    print(f"Prediction saved at {output_path}")
