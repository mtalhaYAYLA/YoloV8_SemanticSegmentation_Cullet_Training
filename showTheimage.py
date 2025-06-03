import cv2
import numpy as np
import os
import json # Sadece etiket oluşturma kısmı için, görselleştirmede şart değil
import traceback
import matplotlib.pyplot as plt

# --- Bir önceki script'teki label oluşturma fonksiyonunu kullanacağız ---
# generate_segmentation_labels fonksiyonunu buraya kopyalayın veya import edin.
# Kolaylık olması için, sadece kontur çıkarma ve basitleştirme mantığını
# içeren bir yardımcı fonksiyon yazalım ve ana label oluşturma fonksiyonunu
# ayrıca çağıralım.

def get_processed_contours_from_mask(
    mask_path,
    min_contour_area=150,
    apply_morphological_operations=True,
    morph_open_kernel_size=(3,3),
    morph_close_kernel_size=None,
    approx_poly_epsilon_factor=0.0015
    ):
    """
    Verilen bir maskeden işlenmiş ve basitleştirilmiş konturları döndürür.
    Görselleştirme için ara maskeleri de döndürür.
    """
    visualization_output = {
        'original_mask_loaded': None,
        'processed_binary_mask': None,
        'final_contours': [],
        'error': None
    }
    try:
        mask_image_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image_original is None:
            visualization_output['error'] = f"Maske yüklenemedi: {mask_path}"
            return visualization_output
        
        visualization_output['original_mask_loaded'] = mask_image_original.copy()

        _, binary_mask_initial = cv2.threshold(mask_image_original, 0, 255, cv2.THRESH_BINARY)
        processed_binary_mask = binary_mask_initial.copy()

        if apply_morphological_operations:
            if morph_open_kernel_size:
                open_kernel = np.ones(morph_open_kernel_size, np.uint8)
                processed_binary_mask = cv2.morphologyEx(processed_binary_mask, cv2.MORPH_OPEN, open_kernel)
            if morph_close_kernel_size:
                close_kernel = np.ones(morph_close_kernel_size, np.uint8)
                processed_binary_mask = cv2.morphologyEx(processed_binary_mask, cv2.MORPH_CLOSE, close_kernel)
        
        visualization_output['processed_binary_mask'] = processed_binary_mask.copy()

        contours_found, _ = cv2.findContours(processed_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_found:
            if cv2.contourArea(contour) < min_contour_area:
                continue

            epsilon = approx_poly_epsilon_factor * cv2.arcLength(contour, True)
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approximated_contour) >= 3:
                visualization_output['final_contours'].append(approximated_contour)
        
        return visualization_output

    except Exception as e:
        visualization_output['error'] = f"Kontur işlenirken hata: {e} \n{traceback.format_exc()}"
        return visualization_output


# generate_segmentation_labels fonksiyonunu buraya kopyalayın
# (bir önceki cevaptaki gibi) eğer etiketleri de oluşturmak istiyorsanız.
# Şimdilik sadece görselleştirmeye odaklanacağız.
# Eğer etiket oluşturma da gerekirse, o fonksiyonu da çağırırız.

# --- Ana İşlem ve Görselleştirme Alanı ---
if __name__ == "__main__":
    # --- KULLANICI AYARLARI ---
    # 1. Maske dosyalarınızın bulunduğu klasör
    input_masks_dir = r"C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_data_ock_08_25\train"
    
    # 2. Orijinal renkli görüntülerinizin bulunduğu klasör
    # Bu klasördeki dosya adları, maske adlarından ".rf.HASH" kısmı çıkarılarak elde edilmeli
    original_images_dir = r"C:\Users\beu\Desktop\SAKARYA\esref\cullet_count\datasets\fercam_data_ock_08_25\original_images_train" # ÖRNEK YOL, KENDİNİZE GÖRE DEĞİŞTİRİN
    original_image_extensions = [".jpg", ".jpeg", ".png"] # Orijinal görüntülerin olası uzantıları

    # 3. Kontur işleme parametreleri
    MIN_CONTOUR_AREA = 150
    APPLY_MORPH_OP = True
    MORPH_OPEN_KERNEL = (3,3)
    MORPH_CLOSE_KERNEL = None # veya (3,3)
    APPROX_POLY_EPS_FACTOR = 0.0015

    # 4. Gösterilecek maksimum görüntü sayısı
    MAX_IMAGES_TO_SHOW = 5
    # --- KULLANICI AYARLARI SONU ---

    # (Etiket oluşturma istenirse) Çıktı klasörleri
    # current_working_directory = os.getcwd()
    # target_output_yolo_txt_dir = os.path.join(current_working_directory, "generated_yolo_labels_test")
    # target_output_json_dir = os.path.join(current_working_directory, "generated_json_labels_test")
    # os.makedirs(target_output_yolo_txt_dir, exist_ok=True)
    # os.makedirs(target_output_json_dir, exist_ok=True)


    print(f"Maskeler şuradan okunacak: {input_masks_dir}")
    print(f"Orijinal renkli görüntüler şuradan aranacak: {original_images_dir}")

    shown_image_count = 0

    if not os.path.isdir(input_masks_dir):
        print(f"HATA: Girdi maske klasörü bulunamadı: {input_masks_dir}")
        exit()
    if not os.path.isdir(original_images_dir):
        print(f"HATA: Orijinal görüntüler klasörü bulunamadı: {original_images_dir}")
        print("Lütfen 'original_images_dir' yolunu doğru ayarlayın veya bu özelliği kullanmak istemiyorsanız kodu düzenleyin.")
        # exit() # Orijinal görüntüler olmadan da devam edebiliriz, sadece maskeyi gösteririz.

    mask_files_to_process = [
        f for f in os.listdir(input_masks_dir) 
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    if not mask_files_to_process:
        print(f"UYARI: '{input_masks_dir}' klasöründe işlenecek maske dosyası bulunamadı.")
    else:
        print(f"Toplam {len(mask_files_to_process)} maske dosyası var. İlk {MAX_IMAGES_TO_SHOW} tanesi (eğer varsa) için test yapılacak.")

        for i, mask_filename in enumerate(mask_files_to_process):
            if shown_image_count >= MAX_IMAGES_TO_SHOW:
                print(f"\nİlk {MAX_IMAGES_TO_SHOW} görüntü testi tamamlandı.")
                break 

            print(f"\nTest ediliyor ({i+1}/{len(mask_files_to_process)}): {mask_filename}")
            mask_file_path = os.path.join(input_masks_dir, mask_filename)

            # Orijinal renkli görüntüyü bulmaya çalış
            base_mask_name_part = os.path.splitext(mask_filename)[0]
            if ".rf." in base_mask_name_part:
                original_image_base = base_mask_name_part.split(".rf.")[0]
            else:
                original_image_base = base_mask_name_part
            
            original_image_path = None
            if os.path.isdir(original_images_dir):
                for ext in original_image_extensions:
                    potential_orig_path = os.path.join(original_images_dir, original_image_base + ext)
                    if os.path.exists(potential_orig_path):
                        original_image_path = potential_orig_path
                        break
            
            original_rgb_image = None
            if original_image_path:
                original_rgb_image = cv2.imread(original_image_path)
                if original_rgb_image is not None:
                    print(f"  -> Eşleşen orijinal görüntü bulundu: {os.path.basename(original_image_path)}")
                else:
                    print(f"  -> UYARI: Orijinal görüntü '{os.path.basename(original_image_path)}' yüklenemedi.")
            else:
                print(f"  -> UYARI: Eşleşen orijinal renkli görüntü bulunamadı ({original_image_base} için).")


            # Maskeden konturları al
            viz_results = get_processed_contours_from_mask(
                mask_path=mask_file_path,
                min_contour_area=MIN_CONTOUR_AREA,
                apply_morphological_operations=APPLY_MORPH_OP,
                morph_open_kernel_size=MORPH_OPEN_KERNEL,
                morph_close_kernel_size=MORPH_CLOSE_KERNEL,
                approx_poly_epsilon_factor=APPROX_POLY_EPS_FACTOR
            )

            if viz_results.get('error'):
                print(f"  -> HATA (get_processed_contours_from_mask): {viz_results['error']}")
                continue # Bir sonraki dosyaya geç

            shown_image_count += 1
            
            # Görselleştirme
            num_subplots = 2
            if original_rgb_image is not None:
                num_subplots = 3
            
            fig, axs = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))
            fig.suptitle(f"Test: {mask_filename}\n(min_area={MIN_CONTOUR_AREA}, eps_factor={APPROX_POLY_EPS_FACTOR})", fontsize=10)

            subplot_idx = 0

            # 1. Orijinal Renkli Görüntü (varsa)
            if original_rgb_image is not None:
                axs[subplot_idx].imshow(cv2.cvtColor(original_rgb_image, cv2.COLOR_BGR2RGB))
                axs[subplot_idx].set_title("1. Orijinal Görüntü")
                axs[subplot_idx].axis('off')
                subplot_idx += 1

            # 2. Yüklenen Maske Görüntüsü
            loaded_mask_display = viz_results.get('original_mask_loaded', np.array([[0]])).copy()
            if np.max(loaded_mask_display) == 1: # 0-1 ise
                loaded_mask_display = loaded_mask_display * 255
            axs[subplot_idx].imshow(loaded_mask_display, cmap='gray')
            axs[subplot_idx].set_title(f"{subplot_idx+1}. Yüklenen Maske")
            axs[subplot_idx].axis('off')
            subplot_idx += 1
            
            # 3. Konturların Orijinal Görüntü Üzerine Bindirilmesi (varsa)
            #    veya İşlenmiş Maske Üzerine Bindirilmesi
            display_image_for_contours = None
            title_for_contours_plot = ""

            if original_rgb_image is not None:
                display_image_for_contours = original_rgb_image.copy()
                title_for_contours_plot = f"{subplot_idx+1}. Konturlar Orijinal Üzerinde"
            elif viz_results.get('processed_binary_mask') is not None: # Orijinal yoksa işlenmiş maskeyi kullan
                processed_m_color = cv2.cvtColor(viz_results['processed_binary_mask'], cv2.COLOR_GRAY2BGR)
                display_image_for_contours = processed_m_color
                title_for_contours_plot = f"{subplot_idx+1}. Konturlar İşlenmiş Maske Üzerinde"
            
            final_contours = viz_results.get('final_contours', [])
            if display_image_for_contours is not None and final_contours:
                cv2.drawContours(display_image_for_contours, final_contours, -1, (0, 255, 0), 2) # Yeşil konturlar
                if original_rgb_image is not None: # Eğer renkli ise RGB'ye çevir
                     axs[subplot_idx].imshow(cv2.cvtColor(display_image_for_contours, cv2.COLOR_BGR2RGB))
                else: # Zaten BGR (renkli maske üzerine çizim) veya Gri ise imshow halleder
                     axs[subplot_idx].imshow(display_image_for_contours)

                axs[subplot_idx].set_title(f"{title_for_contours_plot} ({len(final_contours)} adet)")
            elif display_image_for_contours is not None: # Kontur yoksa bile resmi göster
                 if original_rgb_image is not None:
                     axs[subplot_idx].imshow(cv2.cvtColor(display_image_for_contours, cv2.COLOR_BGR2RGB))
                 else:
                     axs[subplot_idx].imshow(display_image_for_contours)
                 axs[subplot_idx].set_title(f"{title_for_contours_plot} (Kontur Yok)")
            else:
                axs[subplot_idx].text(0.5, 0.5, "Kontur Gösterimi İçin Görüntü Yok", ha='center', va='center')

            axs[subplot_idx].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            # İsteğe bağlı: Etiketleri de burada oluşturabilirsiniz
            # generate_segmentation_labels(...) çağrısını yaparak.

    print("\n--- Test ve Görselleştirme İşlemi Tamamlandı ---")