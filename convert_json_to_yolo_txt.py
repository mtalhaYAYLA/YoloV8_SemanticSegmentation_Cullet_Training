import json
import os
import shutil # shutil'i ekledik

def convert_json_labels_to_yolo_txt(json_labels_source_dir, images_target_dir, yolo_txt_output_dir):
    """
    Bir klasördeki JSON etiket dosyalarını okur ve YOLO .txt formatına dönüştürür.
    Aynı zamanda eşleşen .jpg dosyalarını images_target_dir'e taşır/kopyalar.
    """
    os.makedirs(images_target_dir, exist_ok=True)
    os.makedirs(yolo_txt_output_dir, exist_ok=True)
    
    json_files_processed = 0
    txt_files_created = 0
    jpg_files_moved = 0

    for filename in os.listdir(json_labels_source_dir):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(json_labels_source_dir, filename)
            json_basename = os.path.splitext(filename)[0] # örn: CLAHE_1

            # Eşleşen .jpg dosyasını bul ve taşı/kopyala
            original_jpg_filename = json_basename + ".jpg"
            original_jpg_source_path = os.path.join(json_labels_source_dir, original_jpg_filename)
            target_jpg_path = os.path.join(images_target_dir, original_jpg_filename)

            if os.path.exists(original_jpg_source_path):
                if not os.path.exists(target_jpg_path): # Sadece hedefte yoksa taşı
                    try:
                        shutil.move(original_jpg_source_path, target_jpg_path) # VEYA shutil.copy2
                        jpg_files_moved +=1
                    except Exception as e:
                        print(f"  HATA: '{original_jpg_filename}' taşınırken/kopyalanırken: {e}")
                        continue # Bu çifti işlemeyi durdur
            # else:
                # print(f"  UYARI: '{original_jpg_filename}' için kaynak .jpg bulunamadı. Sadece JSON işlenecek.")
                # Bu durumda JSON'daki image_filename'e güvenmemiz gerekebilir veya hata verebiliriz.
                # Şimdilik, .jpg olmadan da .txt oluşturmaya çalışalım.


            json_files_processed +=1
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                txt_filename = json_basename + ".txt"
                txt_path = os.path.join(yolo_txt_output_dir, txt_filename)
                
                yolo_lines = []
                if "annotations" in data and isinstance(data["annotations"], list):
                    for ann in data["annotations"]:
                        class_id = ann.get("class_id", 0)
                        if "segmentation_normalized" in ann and \
                           isinstance(ann["segmentation_normalized"], list) and \
                           len(ann["segmentation_normalized"]) > 0 and \
                           isinstance(ann["segmentation_normalized"][0], list):
                            points = ann["segmentation_normalized"][0]
                            if len(points) >= 6 :
                                line = f"{class_id} " + " ".join(map(str, points))
                                yolo_lines.append(line)
                
                if yolo_lines:
                    with open(txt_path, 'w') as f_txt:
                        f_txt.write("\n".join(yolo_lines))
                    txt_files_created +=1
                # else: # Anlamlı segmentasyon yoksa boş .txt dosyası YOLO için daha iyidir.
                #     open(txt_path, 'w').close() 
                #     txt_files_created +=1 # Boş dosya da oluşturuldu sayılır


            except Exception as e:
                print(f"HATA: {filename} işlenirken sorun oluştu: {e}")
    
    print(f"  İşlenen JSON dosyası: {json_files_processed}")
    print(f"  Oluşturulan .txt dosyası: {txt_files_created}")
    print(f"  Images klasörüne taşınan/kopyalanan .jpg: {jpg_files_moved}")


if __name__ == '__main__':
    prepared_dataset_base_dir = r"C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_segment_model\yolov8_prepared_dataset_final_detaylıParametre"
    sets_to_convert = ["train", "val", "test"] # İşlenecek setler

    for s in sets_to_convert:
        print(f"\n'{s}' seti için JSON'dan TXT'ye dönüştürme ve dosya düzenleme başlıyor...")
        
        source_set_dir_containing_json_and_jpg = os.path.join(prepared_dataset_base_dir, s) # örn: ../yolov8_prepared_dataset_final/train/
        
        target_images_dir_for_set = os.path.join(source_set_dir_containing_json_and_jpg, "images")
        target_labels_dir_for_set = os.path.join(source_set_dir_containing_json_and_jpg, "labels") # .txt dosyaları buraya
        
        if not os.path.isdir(source_set_dir_containing_json_and_jpg):
            print(f"UYARI: Kaynak set klasörü '{source_set_dir_containing_json_and_jpg}' bulunamadı. Atlanıyor.")
            continue

        # JSON'ları .txt'ye dönüştür, .jpg'leri images'a taşı
        convert_json_labels_to_yolo_txt(
            json_labels_source_dir=source_set_dir_containing_json_and_jpg, # JSON ve JPG'lerin olduğu yer
            images_target_dir=target_images_dir_for_set,  # JPG'lerin taşınacağı/kopyalanacağı yer
            yolo_txt_output_dir=target_labels_dir_for_set  # .txt'lerin kaydedileceği yer
        )
        print(f"'{s}' seti için işlem tamamlandı.")

    print("\nTüm setler için JSON'dan TXT'ye dönüştürme ve dosya düzenleme işlemi bitti.")
    print("Lütfen `dataset.yaml` dosyanızdaki yolların `images` alt klasörlerini gösterdiğinden emin olun.")