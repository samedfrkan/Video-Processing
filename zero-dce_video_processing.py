import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import sys

# FURKAN YILDIZ 152120191002
# SAMED FURKAN DEMİR 152120201070

# Model 
sys.path.append("C:/Users/furka/Desktop/Zero-DCE/Zero-DCE_code")
from model import enhance_net_nopool

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    blur = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)

def zero_dce_enhance(image_bgr, model, device):
    # Model kullanarak aydınlat 
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, enhanced_image, _ = model(input_tensor)

    output = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    output = np.clip(output, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Hafif Gürültü azaltma
    denoised = cv2.fastNlMeansDenoisingColored(output_bgr, None, 5, 5, 7, 15)

    # Hafif CLAHE 
    lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    output_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    return output_bgr

def fix_bright_image(image_bgr):
    # Gamma düzeltmesi (parlaklığı azaltır)
    gamma = 0.6
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    darkened = cv2.LUT(image_bgr, table)

    # Kontrast iyileştirme (CLAHE)
    lab = cv2.cvtColor(darkened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return result

video_path = r"C:\Users\furka\Desktop\Image_processing\Segment_001.avi"
processed_dir = r"C:\Users\furka\Desktop\Image_processing\dce_001_processed_frames"
comparison_dir = r"C:\Users\furka\Desktop\Image_processing\dce_001_comparison_frames"
output_video_path = r"C:\Users\furka\Desktop\Image_processing\dce_001_enhanced_video.avi"
comparison_video_path = r"C:\Users\furka\Desktop\Image_processing\dce_001_comparison_video.avi"
model_path = r"C:\Users\furka\Desktop\Zero-DCE\Zero-DCE_code\snapshots\Epoch99.pth"

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# Modeli hazırlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = enhance_net_nopool().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Video işleme
cap = cv2.VideoCapture(video_path)
frame_num = 0
brightness_threshold = 60  # Karanlık eşik

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < brightness_threshold:
        frame = zero_dce_enhance(frame, model, device)
    elif brightness > brightness_threshold:
        frame = fix_bright_image(frame)

    # Kayıt
    processed_path = os.path.join(processed_dir, f"frame_{frame_num:04d}.jpg")
    cv2.imwrite(processed_path, frame)

    # Karşılaştırma görseli
    comparison = np.hstack((original, frame))
    comparison_path = os.path.join(comparison_dir, f"comp_{frame_num:04d}.jpg")
    cv2.imwrite(comparison_path, comparison)

    frame_num += 1

cap.release()
print(f"{frame_num} kare işlendi ve kayıt edildi.")

# Yeni video oluşturma
frame_example = cv2.imread(os.path.join(processed_dir, "frame_0000.jpg"))
height, width, _ = frame_example.shape
fps = 10  

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
for i in range(frame_num):
    frame = cv2.imread(os.path.join(processed_dir, f"frame_{i:04d}.jpg"))
    out.write(frame)
out.release()
print("Aydınlatılmış video oluşturuldu.")

# Karşılaştırmalı video oluşturma
comp_example = cv2.imread(os.path.join(comparison_dir, "comp_0000.jpg"))
h_c, w_c, _ = comp_example.shape

out_comp = cv2.VideoWriter(comparison_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_c, h_c))
for i in range(frame_num):
    frame = cv2.imread(os.path.join(comparison_dir, f"comp_{i:04d}.jpg"))
    out_comp.write(frame)
out_comp.release()
print("Karşılaştırmalı video oluşturuldu.")
