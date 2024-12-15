from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from model import ModelHandler  # model.py에서 ModelHandler 클래스 임포트
from visualizer import Visualizer

import time

# Flask 애플리케이션 설정
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
GENERATED_FOLDER = './static/generated'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# 모델 로드
MODEL_PATH = 'modelpth/best_model_epoch_50.pth'
model_handler = ModelHandler(MODEL_PATH)  # ModelHandler 인스턴스 생성

@app.route('/')
def index():
    """홈 페이지 렌더링"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
    except FileNotFoundError:
        files = []
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """이미지 업로드 처리"""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('index'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    """
    업로드된 파일 및 생성된 파일 삭제
    """
    try:
        # 업로드된 파일 경로
        upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # 생성된 파일 경로
        prediction_output_filename = f"predicted_{filename}"
        animation_output_filename = f"animated_{filename.split('.')[0]}.mp4"
        prediction_filepath = os.path.join(app.config['GENERATED_FOLDER'], prediction_output_filename)
        animation_filepath = os.path.join(app.config['GENERATED_FOLDER'], animation_output_filename)

        # 업로드된 파일 삭제
        if os.path.exists(upload_filepath):
            os.remove(upload_filepath)

        # 생성된 예측 결과 삭제
        if os.path.exists(prediction_filepath):
            os.remove(prediction_filepath)

        # 생성된 애니메이션 삭제
        if os.path.exists(animation_filepath):
            os.remove(animation_filepath)

        return jsonify({'success': True}), 200

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/generate/<filename>', methods=['POST'])
def generate_image(filename):
    """
    업로드된 이미지를 처리하고 애니메이션을 생성.
    Visualizer 클래스를 사용하여 3D 애니메이션을 생성.
    """
    try:
        # 업로드된 이미지 경로
        input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        app.logger.info(f"1 - {input_image_path}")
        # 결과 저장 경로
        prediction_output_filename = f"predicted_{filename}"
        prediction_output_path = os.path.join(app.config['GENERATED_FOLDER'], prediction_output_filename)
        animation_output_filename = f"animated_{filename.split('.')[0]}.mp4"
        animation_output_path = os.path.join(app.config['GENERATED_FOLDER'], animation_output_filename)

        # 모델을 사용하여 예측 생성 및 JPG 저장
        app.logger.info(f"2 - {input_image_path}")
        model_handler.predict(input_image_path, prediction_output_path)

        # Visualizer를 사용하여 애니메이션 생성
        visualizer = Visualizer(input_image_path, prediction_output_path)
        visualizer.load_data()
        visualizer.visualize(save_animation=True, output_path=animation_output_path)

        # 결과 반환
        return jsonify({
            'success': True,
            'image_url': url_for('static', filename=f'generated/{prediction_output_filename}'),
            'animation_url': url_for('static', filename=f'generated/{animation_output_filename}')
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
