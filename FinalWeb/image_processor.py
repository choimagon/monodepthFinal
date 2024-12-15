import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, grid_size=6):
        self.grid_size = grid_size
    
    def calculate_blur(self, img, x, y, radius=10):
        h, w = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        local_region = cv2.bitwise_and(img, img, mask=mask)
        laplacian_var = cv2.Laplacian(local_region, cv2.CV_64F).var()
        return laplacian_var

    def find_focus_points(self, gray_img):
        h, w = gray_img.shape
        points_x = np.linspace(0, w - 1, self.grid_size, dtype=int)
        points_y = np.linspace(0, h - 1, self.grid_size, dtype=int)

        center_x = points_x[1:-1]
        center_y = points_y[1:-1]

        focus_scores = []

        for y in center_y:
            for x in center_x:
                blur_score = self.calculate_blur(gray_img, x, y)
                focus_scores.append(((x, y), blur_score))

        focus_scores.sort(key=lambda x: x[1], reverse=True)
        top_1 = focus_scores[0]
        top_2 = focus_scores[1] if len(focus_scores) > 1 else top_1
        return top_1[0], top_2[0], top_1[1] - top_2[1]

    def apply_gaussian_blur(self, gray_img, point, sigma):
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError(f"Point {point} is not a valid (x, y) tuple.")
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.circle(mask, point, 1000, 255, -1)
        blurred1 = cv2.GaussianBlur(gray_img, (0, 0), sigma / 10)
        blurred1 = np.where(mask == 255, blurred1, gray_img)
        blurred2 = cv2.GaussianBlur(gray_img, (0, 0), sigma)
        blurred2 = np.where(mask == 255, blurred2, gray_img)
        result = cv2.subtract(blurred2, blurred1)
        return result

    def process_image(self, image):
    # 입력 이미지를 3채널로 변환 (Grayscale 이미지가 들어오는 경우 처리)
        if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Grayscale 변환
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 초점 포인트와 블러 적용
        focus_point_1, focus_point_2, sigma = self.find_focus_points(gray_img)
        sigma = max(sigma, 1.0)
        if not (isinstance(focus_point_1, tuple) and isinstance(focus_point_2, tuple)):
            raise ValueError("Focus points must be tuples of (x, y).")
        
        blurred_img = self.apply_gaussian_blur(gray_img, focus_point_1, sigma)
        return blurred_img


def visualize_image_processing(image_path, processor):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Matplotlib를 위해 RGB로 변환

    # 처리 시간 측정 시작
    start_time = time.time()
    
    # 이미지 처리
    blurred_img = processor.process_image(image)
    
    # 처리 시간 측정 종료
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Blurred image processing time: {processing_time:.4f} seconds")

    # 관심 영역 표시를 위해 그레이스케일 이미지와 초점 포인트 추출
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    focus_point_1, focus_point_2, _ = processor.find_focus_points(gray_img)

    # 관심 영역 표시
    image_with_points = image.copy()
    cv2.circle(image_with_points, focus_point_1, 10, (255, 0, 0), -1)  # 빨간색 점
    cv2.circle(image_with_points, focus_point_2, 10, (0, 255, 0), -1)  # 초록색 점

    # 원본 이미지와 블러 처리된 이미지 합성
    blended_img = cv2.addWeighted(image.copy(), 0.8, cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2RGB), 0.2, 0)
    # 합성 이미지 저장 (RGB -> BGR로 변환 후 저장)
    blended_img_bgr = cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR)
    output_path = "blended_image.jpg"
    cv2.imwrite(output_path, blended_img_bgr)
    print(f"Blended image saved to: {output_path}")
    # 결과 시각화
    plt.figure(figsize=(20, 5))
    
    # 원본 이미지
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title(f"Original Image {image.shape}")
    plt.axis("off")
    
    # 초점 포인트 표시 이미지
    plt.subplot(1, 5, 2)
    plt.imshow(image_with_points)
    plt.title(f"Focus Points {image_with_points.shape}")
    plt.axis("off")
    
    # 블러 처리 결과
    plt.subplot(1, 5, 3)
    plt.imshow(blurred_img, cmap='gray')
    plt.title(f"Blurred Image {blurred_img.shape}")
    plt.axis("off")

    # 원본과 블러 합성 이미지
    plt.subplot(1, 5, 4)
    plt.imshow(blended_img)
    plt.title("Blended Image")
    plt.axis("off")

    # Ground Truth 이미지 표시
    gtimg = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 5, 5)
    plt.imshow(gtimg, cmap='gray')
    plt.title(f"GT {gtimg.shape}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


# 예제 실행
if __name__ == "__main__":
    image_path = "2.jpg"  # 이미지 파일 경로 설정
    processor = ImageProcessor(grid_size=6)
    visualize_image_processing(image_path, processor)
