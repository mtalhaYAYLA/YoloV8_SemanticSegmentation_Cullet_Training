# YoloV8_SemanticSegmentation_Cullet_Training

# 🔍 YOLOv8 ile Kırık Cam Segmentasyonu

Bu proje, endüstriyel cam yüzeylerinde oluşan **kırık bölgeleri (cullet)** tespit etmek ve bu bölgeleri hassas bir şekilde segmentlemek amacıyla geliştirilmiştir. Derin öğrenme tabanlı **YOLOv8x-seg** modeli kullanılmıştır.

## 🎯 Proje Amacı

- Üretim hattındaki cam yüzeylerdeki kırıkların otomatik tespiti
- Yüksek doğrulukta segmentasyon maskeleri üretimi
- Kalite kontrol süreçlerini hızlandırmak ve otomatikleştirmek

## 📁 Veri Seti Yapısı

Görüntüler ve maskeler aşağıdaki formatta yapılandırılmıştır:


dataset/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
├── test/
│ ├── images/
│ └── labels/



- `images/` klasöründe .jpg formatında renkli cam görüntüleri bulunur.
- `labels/` klasöründe her görüntüye ait segmentasyon maskesinin YOLO formatında `.txt` karşılığı yer alır.

## ⚙️ Eğitim Süreci

- Model: `yolov8x-seg.pt`
- Görüntü boyutu: `640x640`
- Epoch: `100`
- Batch size: `8`
- İzleme: TensorBoard üzerinden

> Eğitim sırasında `.pt` model dosyası büyük boyut nedeniyle bu repoya eklenmemiştir.

## 📊 Başarım Metrikleri

| Metrik               | Değer  |
|----------------------|--------|
| Mask mAP@0.50        | %96    |
| Mask mAP@0.50-0.95   | %73    |
| mIoU (Ortalama IoU)  | Yüksek |
| Dice / F1 Skoru      | Başarılı |
| Recall (İnce Kırık)  | Güçlü tespit |

## 🚀 Kullanılan Scriptler

- `convert_json_to_yolo_txt.py` — JSON maskelerini YOLOv8 formatına dönüştürür.
- `split_train_val.py` — Eğitim/val verisini böler.
- `trainWithCustomP.py` — Eğitim sürecini başlatır.
- `test_model_on_dataset.py` — Test veri setinde inference yapar.
- `plot_training_metrics.py` — Eğitim loglarını görselleştirir.

## ⚠️ Notlar

- `.pt` modeli bu repoda yer almamaktadır. Kullanmak isterseniz eğitim script'ini çalıştırarak yeniden üretebilirsiniz.
- Segmentasyon verileri dışındaki büyük klasörler `.gitignore` ile hariç bırakılmıştır.

---

## 👤 Geliştirici

**Hasan Özgür Doğan**  
GitHub: [@mtalhaYAYLA](https://github.com/mtalhaYAYLA)  
AISOFT

---

