import os
import random
import shutil

def split_dataset(base_dataset_dir, train_subdir="train", val_subdir="val", val_split_ratio=0.15, seed=42):
    """
    Verilen bir veri setinin train alt klasöründeki .jpg ve eşleşen .json dosyalarını
    rastgele olarak train ve val setlerine böler.

    Args:
        base_dataset_dir (str): Ana veri kümesi klasörü (örn: 'yolov8_prepared_dataset_final').
        train_subdir (str): Mevcut train verilerinin olduğu alt klasör adı.
        val_subdir (str): Oluşturulacak val verilerinin konulacağı alt klasör adı.
        val_split_ratio (float): Val setine ayrılacak veri oranı (örn: 0.15 = %15).
        seed (int): Rastgelelik için tohum değeri.
    """
    random.seed(seed)

    source_train_dir = os.path.join(base_dataset_dir, train_subdir)
    target_val_dir = os.path.join(base_dataset_dir, val_subdir)

    if not os.path.isdir(source_train_dir):
        print(f"HATA: Kaynak train klasörü bulunamadı: {source_train_dir}")
        return

    os.makedirs(target_val_dir, exist_ok=True)
    print(f"Val klasörü oluşturuldu/kontrol edildi: {target_val_dir}")

    # Sadece .jpg dosyalarını listele (bunlar ana dosyalarımız)
    image_files = sorted([f for f in os.listdir(source_train_dir) if f.lower().endswith(".jpg")])
    
    if not image_files:
        print(f"UYARI: '{source_train_dir}' içinde .jpg dosyası bulunamadı.")
        return

    num_images = len(image_files)
    num_val_images = int(num_images * val_split_ratio)

    if num_val_images == 0 and num_images > 0 : # En az 1 tane val resmi olsun (eğer yeterli veri varsa)
        num_val_images = 1 if val_split_ratio > 0 else 0
    
    if num_val_images == 0:
        print("UYARI: Val seti için yeterli sayıda görüntü ayrılamadı. Oranı artırın veya daha fazla veri ekleyin.")
        return

    print(f"Toplam {num_images} görüntüden {num_val_images} tanesi val setine ayrılacak.")

    # Rastgele olarak val için resimleri seç
    random.shuffle(image_files)
    val_image_filenames = image_files[:num_val_images]

    moved_jpg_count = 0
    moved_json_count = 0

    for jpg_filename in val_image_filenames:
        base_name = os.path.splitext(jpg_filename)[0]
        json_filename = base_name + ".json" # Eşleşen JSON dosyasının adı

        source_jpg_path = os.path.join(source_train_dir, jpg_filename)
        target_jpg_path = os.path.join(target_val_dir, jpg_filename)

        source_json_path = os.path.join(source_train_dir, json_filename)
        target_json_path = os.path.join(target_val_dir, json_filename)

        # JPG dosyasını taşı
        if os.path.exists(source_jpg_path):
            try:
                shutil.move(source_jpg_path, target_jpg_path)
                moved_jpg_count += 1
            except Exception as e:
                print(f"  HATA: '{jpg_filename}' taşınırken sorun: {e}")
        else:
            print(f"  UYARI: Kaynak JPG '{source_jpg_path}' bulunamadı.")

        # JSON dosyasını taşı
        if os.path.exists(source_json_path):
            try:
                shutil.move(source_json_path, target_json_path)
                moved_json_count += 1
            except Exception as e:
                print(f"  HATA: '{json_filename}' taşınırken sorun: {e}")
        else:
            print(f"  UYARI: Kaynak JSON '{source_json_path}' bulunamadı (bu beklenen bir durum olabilir eğer bazı JPG'lerin JSON'u yoksa).")
    
    print(f"\n'{val_subdir}' seti oluşturuldu:")
    print(f"  Taşınan JPG dosyası sayısı: {moved_jpg_count}")
    print(f"  Taşınan JSON dosyası sayısı: {moved_json_count}")
    print(f"Kalan JPG dosyası sayısı ('{train_subdir}' içinde): {num_images - moved_jpg_count}")


if __name__ == "__main__":
    # --- KULLANICI AYARLARI ---
    # Scriptinizin çalıştığı dizindeki veri kümesi ana klasörünün adı
    dataset_main_folder = "yolov8_prepared_dataset_final_detaylıParametre" 
    
    # Scriptin çalıştığı dizini al
    current_script_dir = os.getcwd() # Veya os.path.dirname(os.path.abspath(__file__))
    base_prepared_dataset_path = os.path.join(current_script_dir, dataset_main_folder)

    validation_split_percentage = 0.15 # Train setinin %15'i val için ayrılacak

    # --- KULLANICI AYARLARI SONU ---

    if not os.path.isdir(base_prepared_dataset_path):
        print(f"HATA: Hazırlanmış veri kümesi ana klasörü bulunamadı: {base_prepared_dataset_path}")
        print("Lütfen bir önceki scripti çalıştırarak 'yolov8_prepared_dataset_final' klasörünü oluşturduğunuzdan emin olun.")
    else:
        split_dataset(
            base_dataset_dir=base_prepared_dataset_path,
            train_subdir="train", # Kaynak train klasörü
            val_subdir="val",     # Hedef val klasörü
            val_split_ratio=validation_split_percentage
        )