# Zero-DCE Video Processing

Bu proje, Zero-DCE (Zero-Reference Deep Curve Estimation) modelini kullanarak video karelerindeki düşük ışık koşullarını iyileştiren ve aşırı parlak görüntüleri düzelten bir video işleme sistemidir.

## Geliştirici Bilgileri

- **Furkan Yıldız** 
- **Samed Furkan Demir** 

## Özellikler

- **Adaptif Parlaklık Tespiti**: Her kareyi otomatik olarak analiz eder ve parlaklık seviyesine göre uygun işlem uygular
- **Düşük Işık İyileştirme**: Zero-DCE modelini kullanarak karanlık kareleri aydınlatır
- **Aşırı Parlaklık Düzeltme**: Gamma düzeltmesi ile parlak kareleri optimize eder
- **Gelişmiş Görüntü İşleme**: 
  - Gürültü azaltma (Non-local means denoising)
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Unsharp masking desteği
- **Karşılaştırmalı Çıktı**: Orijinal ve işlenmiş kareleri yan yana gösterir

## Gereksinimler

### Python Kütüphaneleri
```bash
pip install opencv-python
pip install torch torchvision
pip install pillow
pip install numpy
```

### Donanım Gereksinimleri
- CUDA destekli GPU (önerilir, CPU ile de çalışır)
- Yeterli RAM (video boyutuna bağlı)

## Kurulum

1. **Zero-DCE modelini indirin**:
   ```bash
   git clone https://github.com/Li-Chongyi/Zero-DCE.git
   ```

2. **Eğitilmiş model dosyasını indirin**:
   - `Epoch99.pth` dosyasını Zero-DCE proje dizinine yerleştirin

3. **Dosya yollarını güncelleyin**:
   ```python
   # Aşağıdaki yolları kendi sistem yapınıza göre değiştirin
   video_path = "path/to/your/video.avi"
   model_path = "path/to/Zero-DCE/snapshots/Epoch99.pth"
   ```

## Kullanım

### Temel Kullanım
```python
python zero-dce_video_processing.py
```

### Parametreler

#### Parlaklık Eşiği
```python
brightness_threshold = 60  # Karanlık/aydınlık ayrım eşiği
```

#### Gamma Düzeltmesi
```python
gamma = 0.6  # Parlak görüntüler için gamma değeri
```

#### CLAHE Parametreleri
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

## Çıktı Dosyaları

### Dizin Yapısı
```
project_folder/
├── dce_001_processed_frames/     # İşlenmiş kareler
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── dce_001_comparison_frames/    # Karşılaştırma kareleri
│   ├── comp_0000.jpg
│   ├── comp_0001.jpg
│   └── ...
├── dce_001_enhanced_video.avi    # İşlenmiş video
└── dce_001_comparison_video.avi  # Karşılaştırma videosu
```

### Video Çıktıları
- **Enhanced Video**: Sadece işlenmiş kareler
- **Comparison Video**: Orijinal ve işlenmiş kareleri yan yana gösterir

## Algoritma Detayları

### 1. Parlaklık Analizi
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)
```

### 2. Düşük Işık İyileştirme
- Zero-DCE modeli ile curve estimation
- Non-local means denoising
- CLAHE kontras iyileştirme

### 3. Aşırı Parlaklık Düzeltme
- Gamma düzeltmesi (γ = 0.6)
- LAB color space'de CLAHE uygulaması

## Performans Optimizasyonu

### GPU Kullanımı
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Bellek Yönetimi
```python
with torch.no_grad():
    _, enhanced_image, _ = model(input_tensor)
```

## Sorun Giderme

### Yaygın Hatalar

1. **CUDA Hatası**: GPU sürücülerini kontrol edin
2. **Model Yükleme Hatası**: Model dosya yolunu doğrulayın
3. **Video Codec Hatası**: XVID codec'in yüklü olduğundan emin olun

### Performans İyileştirme

- Büyük videolar için kare örnekleme kullanın
- Batch processing için kodu modifiye edin
- Daha düşük çözünürlükte test edin

## Lisans

Bu proje akademik çalışma amaçlıdır. Zero-DCE orijinal paper'ına referans verin:

```bibtex
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1780--1789},
  year={2020}
}
```

## Katkı Sağlama

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## Teşekkürler

- Zero-DCE orijinal yazarlarına
- OpenCV ve PyTorch topluluklarına
