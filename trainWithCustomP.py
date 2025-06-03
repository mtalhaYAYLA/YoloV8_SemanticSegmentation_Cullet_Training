from ultralytics import YOLO
import os
import torch
import traceback

def train_yolov8x_segmentation_gpu1():
    dataset_yaml_path = r'C:/Users/beu/Desktop/SAKARYA/esref/cullet_count/datasets/fercam_segment_model/yolov8_prepared_dataset_final_detaylıParametre/dataset.yaml'
    script_base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_filename = 'yolov8x-seg.pt'
    model_path = os.path.join(script_base_dir, local_model_filename)

    epochs = 100
    imgsz = 640
    batch_size = 8  # Burada batch size 8 yapıldı
    TARGET_GPU_ID = 1
    run_name = f'cullet_seg_yolov8x_gpu{TARGET_GPU_ID}_detayli_run'
    patience_epochs = 15
    workers_count = 4
    optimizer = 'AdamW'       # optimizer AdamW olarak kalacak
    lr0 = 0.01
    lrf = 0.5
    cos_lr = True
    warmup_epochs = 3
    warmup_bias_lr = 0.05

    print("--- GPU Kontrolü Başlatılıyor ---")
    if not torch.cuda.is_available():
        print("HATA: CUDA destekli GPU bulunamadı! Çıkılıyor.")
        return
    gpu_count = torch.cuda.device_count()
    print(f"Toplam GPU Sayısı: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if TARGET_GPU_ID >= gpu_count or TARGET_GPU_ID < 0:
        print(f"Hedef GPU ID ({TARGET_GPU_ID}) geçersiz. Çıkılıyor.")
        return
    device_to_use_str = str(TARGET_GPU_ID)
    print(f"Eğitim GPU'su: {torch.cuda.get_device_name(TARGET_GPU_ID)} (ID: {TARGET_GPU_ID})")
    print("---------------------------------")

    if not os.path.exists(dataset_yaml_path):
        print(f"YAML dosyası bulunamadı: {dataset_yaml_path}")
        return
    if not os.path.exists(model_path):
        print(f"Model ağırlık dosyası bulunamadı: {model_path}")
        return

    try:
        model = YOLO(model_path)
        print(f"Model yüklendi: {model_path}")

        print(f"\nEğitim başlatılıyor...")
        print(f"Epochs: {epochs}, Batch: {batch_size}, Image Size: {imgsz}x{imgsz}")

        train_args = {
            'data': dataset_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device_to_use_str,
            'name': run_name,
            'patience': patience_epochs,
            'workers': workers_count,
            'optimizer': optimizer,
            'lr0': lr0,
            'lrf': lrf,
            'cos_lr': cos_lr,
            'warmup_epochs': warmup_epochs,
            'warmup_bias_lr': warmup_bias_lr,
            'project': 'runs/custom_segmentations',
            'verbose': True,
            'half': True  # Mixed precision ile bellek optimizasyonu
        }

        results = model.train(**train_args)

        print("\nEğitim tamamlandı!")
        print(f"Kayıt dizini: {model.trainer.save_dir}")
        print(f"En iyi model: {model.trainer.best}")

    except Exception as e:
        print(f"Eğitim hatası: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    train_yolov8x_segmentation_gpu1()
