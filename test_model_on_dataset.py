import os
import random # <-- random modülünü ekledik
import shutil # <-- shutil modülünü ekledik (kopyalama için)
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import time

def test_segmentation_on_single_image(
    weights_path,                   # Eğitilmiş modelin .pt dosyasının yolu
    single_image_path,              # Test edilecek tek görüntünün yolu
    output_results_dir=None,        # (Opsiyonel) Sonucun kaydedileceği klasör
    confidence_threshold=0.25,
    iou_threshold=0.7,
    show_result_live=False # Genellikle toplu testte canlı göstermeyiz, False yaptık varsayılanı
    ):
    """
    Eğitilmiş YOLOv8 segmentasyon modelini TEK BİR görüntü üzerinde çalıştırır.
    Çıktı klasörü belirtilmişse orijinal görseli ve segmentasyon sonucunu kaydeder.
    """
    print(f"\n>>> İşleniyor: {os.path.basename(single_image_path)}")
    print(f">>> Model yolu: {weights_path}")
    if output_results_dir:
        print(f">>> Çıktı klasörü: {output_results_dir}")
    print("-------------------------------------")

    # Modeli yükle (Modeli her seferinde yüklemek yerine dışarıda yükleyip fonksiyona pass edebiliriz
    # ama bu haliyle de çalışır, sadece biraz daha yavaş olabilir ilk çağrılarda)
    try:
        # Modelin zaten yüklü olduğunu varsayarak (ana blokta yükleyeceğiz) bu kısmı yorum satırı yapıyorum
        # Ancak eğer her görsel için yeniden yüklemek isterseniz bu satırları kullanabilirsiniz.
        # print(f">>> Model yükleniyor...")
        # model = YOLO(weights_path)
        # print(f">>> MODEL BAŞARIYLA YÜKLENDİ")
        # Not: Model nesnesini dışarıdan alacak şekilde fonksiyon imzasını değiştirmek daha verimli olur.
        # Şimdilik içeride yüklüyorum, siz dilerseniz dışarı alabilirsiniz.
        model = YOLO(weights_path)


    except Exception as e:
        print(f"HATA: Model yüklenirken sorun oluştu: {e}")
        traceback.print_exc()
        return

    # Görüntü dosyasının varlığını kontrol et
    if not os.path.exists(single_image_path):
        print(f"HATA: Test görüntüsü bulunamadı: {single_image_path}")
        return

    # Eğer sonuçlar kaydedilecekse çıktı klasörünü oluştur ve orijinali kopyala
    if output_results_dir:
        # Klasör zaten ana blokta oluşturulacak, burada sadece varlığından emin oluyoruz
        os.makedirs(output_results_dir, exist_ok=True)

        # --- Orijinal görseli kopyala ---
        print(f">>> Orijinal görsel kopyalanıyor: {os.path.basename(single_image_path)}")
        try:
            # Dosya adının başına "original_" ekleyerek kopyala
            original_copied_path = os.path.join(output_results_dir, f"original_{os.path.basename(single_image_path)}")
            shutil.copy2(single_image_path, original_copied_path)
            print(f">>> Orijinal görsel başarıyla kopyalandı: {original_copied_path}")
        except Exception as e:
            print(f"HATA: Orijinal görsel kopyalanırken sorun oluştu: {e}")
            traceback.print_exc()
        # -----------------------------

    try:
        print(">>> Model ile tahmin işlemi başlatılıyor...")
        start_time = time.time()
        # predictions yaparken verbose=False ekleyerek konsol çıktısını azaltabilirsiniz
        results = model.predict(
            source=single_image_path,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=640, # Eğitimde kullanılan boyutu burada da belirtmek faydalı olabilir
            verbose=False # Konsol çıktısını azaltır
            )
        end_time = time.time()
        inference_time = end_time - start_time
        print(f">>> Çıkarım süresi: {inference_time:.4f} saniye")

        if results and len(results) > 0:
            result = results[0]
            # Annotate image: sadece segmentasyon maskeleri çizilsin (boxes ve labels kapalı)
            annotated_frame = result.plot(boxes=False, labels=False)
            print(">>> Tahmin sonuçları başarıyla alındı. (Sadece segmentasyon maskeleri çizildi)")

            if show_result_live:
                print(">>> Sonuç görselleştiriliyor...")
                cv2.imshow(f"Segmentasyon Sonucu - {os.path.basename(single_image_path)}", annotated_frame)
                print(">>> Sonucu kapatmak için herhangi bir tuşa basın...")
                cv2.waitKey(0) # Pencereyi kapatmak için bir tuşa basılmasını bekle
                cv2.destroyAllWindows()

            if output_results_dir:
                # Segmentasyon sonucunu "segmented_" ön ekiyle kaydet
                output_filename = os.path.join(output_results_dir, f"segmented_{os.path.basename(single_image_path)}")
                print(f">>> Tahmin sonucu kaydediliyor: {output_filename}")
                cv2.imwrite(output_filename, annotated_frame)
                print(f">>> Tahmin sonucu başarıyla kaydedildi: {output_filename}")
        else:
            print(">>> Bu görüntü için sonuç bulunamadı.")

    except Exception as e:
        print(f"HATA: {os.path.basename(single_image_path)} işlenirken sorun oluştu: {e}")
        traceback.print_exc()

    print("-------------------------------------")


if __name__ == '__main__':
    # --- KULLANICI AYARLARI ---
    # 1. Eğitilmiş en iyi modelin .pt dosyasının yolu
    MODEL_WEIGHTS_PATH = r'C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_segment_model\yolov8_fercam_segment_model\runs\custom_segmentations\cullet_seg_yolov8x_gpu1_detayli_run2\weights\best.pt'

    # 2. TEST EDİLECEK GÖRSELLERİN BULUNDUĞU KLASÖRÜN TAM YOLU
    TEST_IMAGES_DIR = r'C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_segment_model\yolov8_fercam_segment_model\yolov8_prepared_dataset_final_detaylıParametre\test\images'

    # 3. Sonucun (orijinal kopya ve segmentasyon çizilmiş görsel) kaydedileceği ana klasör
    #    Scriptin çalıştığı dizinde "rapor" adında bir alt klasör oluşturulacak.
    OUTPUT_REPORT_DIR = os.path.join(os.getcwd(), "rapor")

    # 4. Rastgele kaç adet görsel test edilecek?
    NUM_RANDOM_IMAGES_TO_TEST = 5

    # 5. Tahmin parametreleri
    CONF_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.3

    # 6. Her görselin sonucunu işlenirken canlı ekranda gösterilsin mi?
    SHOW_LIVE = False # Genellikle toplu testte False yapılır
    # --- KULLANICI AYARLARI SONU ---

    print(">>> Script başlatıldı: Rastgele test görselleri üzerinde segmentasyon testi.")

    # Model ağırlık dosyasının varlığını kontrol et
    if not os.path.isabs(MODEL_WEIGHTS_PATH):
        # Göreceli yol verilmişse, scriptin çalıştığı dizine göre tam yolu bulmaya çalış
        script_dir_for_model = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        potential_model_path = os.path.join(script_dir_for_model, MODEL_WEIGHTS_PATH)
        if not os.path.exists(potential_model_path) and os.path.exists(MODEL_WEIGHTS_PATH):
             # MODEL_WEIGHTS_PATH zaten geçerli bir göreceli yol olabilir
             pass
        elif os.path.exists(potential_model_path):
            MODEL_WEIGHTS_PATH = potential_model_path # Göreceli yolu mutlak yola çevir
        else:
             print(f"HATA: Model ağırlık dosyası bulunamadı veya yolu çözümlenemedi: {MODEL_WEIGHTS_PATH}")
             exit() # Bulunamazsa çıkış yap

    elif not os.path.exists(MODEL_WEIGHTS_PATH):
         # Mutlak yol verilmiş ve dosya yoksa çıkış yap
         print(f"HATA: Model ağırlık dosyası bulunamadı: {MODEL_WEIGHTS_PATH}")
         exit()

    # Test görselleri klasörünü ve yeterli görsel olup olmadığını kontrol et
    print(f">>> Test görselleri klasörü kontrol ediliyor: {TEST_IMAGES_DIR}")
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"HATA: Test görselleri klasörü bulunamadı: {TEST_IMAGES_DIR}")
        exit()

    # Klasördeki tüm resim dosyalarını listele (basit uzantı kontrolü)
    supported_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    all_image_files = []
    for f in os.listdir(TEST_IMAGES_DIR):
        f_path = os.path.join(TEST_IMAGES_DIR, f)
        if os.path.isfile(f_path) and os.path.splitext(f)[1].lower() in supported_image_extensions:
            all_image_files.append(f_path)

    print(f">>> Klasörde bulunan desteklenen resim sayısı: {len(all_image_files)}")

    if len(all_image_files) < NUM_RANDOM_IMAGES_TO_TEST:
        print(f"HATA: Test klasöründe yeterli sayıda ({NUM_RANDOM_IMAGES_TO_TEST}) resim bulunmuyor.")
        print(f"Bulunan resim sayısı: {len(all_image_files)}")
        exit()

    # Rastgele belirtilen sayıda görsel seç
    print(f">>> Rastgele {NUM_RANDOM_IMAGES_TO_TEST} görsel seçiliyor...")
    try:
        selected_images = random.sample(all_image_files, NUM_RANDOM_IMAGES_TO_TEST)
        print(">>> Seçilen görseller:")
        for img_path in selected_images:
             print(f"  - {os.path.basename(img_path)}")
    except ValueError as e:
         # Bu hata normalde yeterli görsel varsa oluşmaz ama kontrol etmek iyi
         print(f"HATA: Rastgele seçim sırasında sorun oluştu: {e}")
         traceback.print_exc()
         exit()


    # Sonuç klasörünü oluştur (eğer yoksa)
    print(f">>> Sonuç klasörü oluşturuluyor (veya kontrol ediliyor): {OUTPUT_REPORT_DIR}")
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    print(f">>> Sonuç klasörü hazır: {os.path.abspath(OUTPUT_REPORT_DIR)}")

    # Model nesnesini bir kere yükle (Daha verimli yaklaşım)
    try:
        print(">>> Model yükleniyor...")
        model_instance = YOLO(MODEL_WEIGHTS_PATH)
        print(">>> Model başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Model yüklenirken sorun oluştu: {e}")
        traceback.print_exc()
        exit() # Model yüklenemezse devam etmenin anlamı yok

    # Seçilen her görsel için testi çalıştır
    print("\n>>> Seçilen görseller üzerinde testler başlatılıyor...")
    for i, image_path in enumerate(selected_images):
        # Not: Fonksiyonun imzasını model nesnesini alacak şekilde değiştirmedim.
        # Şimdilik fonksiyon her çağrıldığında modeli yeniden yüklüyor.
        # Daha verimli hale getirmek için test_segmentation_on_single_image fonksiyonunu
        # 'model' parametresi alacak şekilde güncelleyebilir ve yukarıda yüklenen 'model_instance'ı pass edebilirsiniz.
        # Ancak mevcut fonksiyon yapısını koruyarak da çalışır.

        test_segmentation_on_single_image(
            weights_path=MODEL_WEIGHTS_PATH, # Model yolu (fonksiyon içinde tekrar yüklenecek)
            single_image_path=image_path,      # Şu an işlenen görselin yolu
            output_results_dir=OUTPUT_REPORT_DIR, # Tüm sonuçlar aynı klasöre gidecek
            confidence_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            show_result_live=SHOW_LIVE         # Canlı gösterme ayarı
        )
        # Her görselin işlenmesi arasında kısa bir duraklama (isteğe bağlı)
        # time.sleep(0.5)

    print("\n>>> Tüm Rastgele Görsel Testleri Tamamlandı <<<")
    print(f"Test sonuçları (orijinal kopyalar ve segmentlenmiş çıktılar) şuraya kaydedildi: {os.path.abspath(OUTPUT_REPORT_DIR)}")