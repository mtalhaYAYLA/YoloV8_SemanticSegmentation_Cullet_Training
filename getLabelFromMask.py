import cv2
import numpy as np
import os
import json
import traceback
import shutil # Dosya kopyalama için

# --- JSON Oluşturma Fonksiyonu (Değişiklik Yok) ---
def convert_mask_to_yolov8_json(
    mask_path,
    original_jpg_filename_for_json,
    output_json_path,
    class_id=0,
    class_name="Cullet",
    min_contour_area=30,
    apply_morph_operations=True,
    morph_open_kernel_size=(3,3),
    approx_poly_epsilon_factor=0.002
    ):
    try:
        mask_image_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image_gray is None:
            print(f"  HATA: PNG Maskesi yüklenemedi: {mask_path}")
            return False
        h, w = mask_image_gray.shape
        if h == 0 or w == 0:
            print(f"  HATA: PNG Maskesi geçersiz boyutlarda: {mask_path}")
            return False
        binary_mask = np.where(mask_image_gray >= 1, 255, 0).astype(np.uint8)
        current_processed_mask = binary_mask.copy()
        if apply_morph_operations:
            if morph_open_kernel_size:
                kernel = np.ones(morph_open_kernel_size, np.uint8)
                current_processed_mask = cv2.morphologyEx(current_processed_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(current_processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        json_annotations_list = []
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            epsilon = approx_poly_epsilon_factor * cv2.arcLength(contour, True)
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(approximated_contour) >= 3:
                normalized_segment_points = []
                for point in approximated_contour:
                    px, py = point[0]
                    normalized_px = round(px / w, 6)
                    normalized_py = round(py / h, 6)
                    normalized_segment_points.extend([normalized_px, normalized_py])
                if normalized_segment_points:
                    json_annotations_list.append({
                        "class_id": class_id, "class_name": class_name,
                        "segmentation_normalized": [normalized_segment_points]
                    })
        if json_annotations_list:
            json_output_data = {
                "image_filename": original_jpg_filename_for_json,
                "image_height": h, "image_width": w,
                "annotations": json_annotations_list
            }
            with open(output_json_path, 'w') as f:
                json.dump(json_output_data, f, indent=2)
            return True
        else:
            return False
    except Exception as e:
        print(f"  HATA ({mask_path}): {e}\n{traceback.format_exc()}")
        return False

# --- Belirli Bir Seti İşleyen Fonksiyon ---
def process_dataset_set(
    set_name,                           # "train", "test", "val" gibi
    source_base_dir,                    # Kaynak verilerin ana klasörü (örn: .../fercam_data_ock_08_25)
    target_dataset_base_dir,            # Hazırlanmış veri setinin kaydedileceği ana klasör
    min_contour_area_param,
    apply_morph_op_param,
    morph_open_kernel_param,
    approx_poly_eps_factor_param
    ):
    """
    Belirtilen bir veri seti alt klasörünü (train, test vb.) işler,
    JPG'leri kopyalar ve PNG maskelerinden JSON etiketleri oluşturur.
    """
    source_set_dir = os.path.join(source_base_dir, set_name)
    target_set_output_dir = os.path.join(target_dataset_base_dir, set_name)

    os.makedirs(target_set_output_dir, exist_ok=True)

    print(f"\n--- '{set_name}' Seti İşleniyor ---")
    print(f"Kaynak Klasör: {source_set_dir}")
    print(f"Hedef Klasör: {target_set_output_dir}")

    if not os.path.isdir(source_set_dir):
        print(f"HATA: Kaynak '{set_name}' klasörü bulunamadı: {source_set_dir}")
        return 0, 0, 0 # processed_pairs, copied_jpgs, created_jsons

    all_files_in_source_set = os.listdir(source_set_dir)
    original_jpg_files = sorted([f for f in all_files_in_source_set if f.lower().endswith(".jpg")])

    if not original_jpg_files:
        print(f"UYARI: '{source_set_dir}' klasöründe .jpg bulunamadı.")
        return 0, 0, 0

    total_jpgs_in_set = len(original_jpg_files)
    print(f"'{set_name}' setinde toplam {total_jpgs_in_set} orijinal .jpg dosyası bulundu.")
    
    processed_pairs_count = 0
    copied_jpg_count = 0
    json_created_count = 0

    for i, jpg_filename in enumerate(original_jpg_files):
        original_jpg_path = os.path.join(source_set_dir, jpg_filename)
        base_jpg_name_no_ext = os.path.splitext(jpg_filename)[0]
        
        found_mask_path = None
        potential_mask_name_simple = base_jpg_name_no_ext + ".png"
        if potential_mask_name_simple in all_files_in_source_set:
            found_mask_path = os.path.join(source_set_dir, potential_mask_name_simple)
        else:
            for candidate_file in all_files_in_source_set:
                if candidate_file.lower().endswith(".png"):
                    candidate_base = os.path.splitext(candidate_file)[0]
                    if ".rf." in candidate_base: # .rf. içeren maske adları için
                        candidate_prefix = candidate_base.split(".rf.")[0].replace("_png", "")
                        if base_jpg_name_no_ext == candidate_prefix:
                            found_mask_path = os.path.join(source_set_dir, candidate_file)
                            break
        
        if not found_mask_path:
            print(f"  ({i+1}/{total_jpgs_in_set}) UYARI: '{jpg_filename}' için eşleşen PNG maskesi bulunamadı. Atlanıyor.")
            continue
        
        processed_pairs_count += 1
        # print(f"  ({i+1}/{total_jpgs_in_set}) İşleniyor: Orijinal='{jpg_filename}', Maske='{os.path.basename(found_mask_path)}'")

        target_jpg_path = os.path.join(target_set_output_dir, jpg_filename)
        try:
            shutil.copy2(original_jpg_path, target_jpg_path)
            copied_jpg_count += 1
        except Exception as e:
            print(f"    -> HATA: '{jpg_filename}' kopyalanamadı: {e}")
            continue

        target_json_filename = base_jpg_name_no_ext + ".json"
        target_json_path = os.path.join(target_set_output_dir, target_json_filename)

        json_success = convert_mask_to_yolov8_json(
            mask_path=found_mask_path,
            original_jpg_filename_for_json=jpg_filename,
            output_json_path=target_json_path,
            min_contour_area=min_contour_area_param,
            apply_morph_operations=apply_morph_op_param,
            morph_open_kernel_size=morph_open_kernel_param,
            approx_poly_epsilon_factor=approx_poly_eps_factor_param
        )
        if json_success:
            json_created_count += 1

    print(f"'{set_name}' seti için işlem tamamlandı.")
    print(f"  İşlenen JPG-PNG çifti: {processed_pairs_count}")
    print(f"  Kopyalanan JPG sayısı: {copied_jpg_count}")
    print(f"  Oluşturulan JSON etiketi sayısı: {json_created_count}")
    return processed_pairs_count, copied_jpg_count, json_created_count


# --- Ana Program ---
if __name__ == "__main__":
    # --- KULLANICI AYARLARI ---
    # 1. Orijinal JPG ve PNG maskelerinin bulunduğu ana kaynak klasör
    SOURCE_BASE_DATA_DIR = r"C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_data_ock_08_25"
    
    # 2. Çıktıların kaydedileceği ana klasör adı (script'in çalıştığı dizinde oluşturulacak)
    OUTPUT_DATASET_MAIN_FOLDER_NAME = "yolov8_prepared_dataset_final"

    # 3. İşlenecek setlerin adları (kaynak klasördeki alt klasör adları)
    SETS_TO_PROCESS = ["train", "test"] # "val" varsa onu da ekleyebilirsiniz

    # 4. Kontur işleme parametreleri (TÜM SETLER İÇİN AYNI KULLANILACAK)
    MIN_CONTOUR_AREA = 50
    APPLY_MORPH_OP = True
    MORPH_OPEN_KERNEL = (5,5)
    APPROX_POLY_EPS_FACTOR = 0.005
    # --- KULLANICI AYARLARI SONU ---

    current_working_dir = os.getcwd()
    target_prepared_dataset_base_dir = os.path.join(current_working_dir, OUTPUT_DATASET_MAIN_FOLDER_NAME)
    os.makedirs(target_prepared_dataset_base_dir, exist_ok=True)

    total_processed_all_sets = 0
    total_copied_jpgs_all_sets = 0
    total_created_jsons_all_sets = 0

    for set_name_to_process in SETS_TO_PROCESS:
        p_pairs, c_jpgs, c_jsons = process_dataset_set(
            set_name=set_name_to_process,
            source_base_dir=SOURCE_BASE_DATA_DIR,
            target_dataset_base_dir=target_prepared_dataset_base_dir,
            min_contour_area_param=MIN_CONTOUR_AREA,
            apply_morph_op_param=APPLY_MORPH_OP,
            morph_open_kernel_param=MORPH_OPEN_KERNEL,
            approx_poly_eps_factor_param=APPROX_POLY_EPS_FACTOR
        )
        total_processed_all_sets += p_pairs
        total_copied_jpgs_all_sets += c_jpgs
        total_created_jsons_all_sets += c_jsons

    print(f"\n--- TÜM VERİ SETİ HAZIRLAMA İŞLEMLERİ TAMAMLANDI ---")
    print(f"Tüm setlerde toplam işlenen JPG-PNG çifti: {total_processed_all_sets}")
    print(f"Tüm setlerde toplam kopyalanan JPG sayısı: {total_copied_jpgs_all_sets}")
    print(f"Tüm setlerde toplam oluşturulan JSON etiketi sayısı: {total_created_jsons_all_sets}")
    print(f"Hazırlanmış veri seti '{target_prepared_dataset_base_dir}' klasörüne kaydedildi.")