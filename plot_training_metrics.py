import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def find_column_name(df_columns, possible_names, default_if_not_found=None):
    for name in possible_names:
        if name in df_columns:
            return name
    print(f"UYARI: '{possible_names}' sütunlarından hiçbiri CSV'de bulunamadı. '{default_if_not_found}' varsayılıyor.")
    return default_if_not_found

def plot_metrics_on_figure(df, epoch_col, 
                           metrics_to_plot, # Çizdirilecek sütun adları listesi
                           labels_for_plot, # Karşılık gelen etiketler listesi
                           colors_for_plot, # Karşılık gelen renkler listesi
                           linestyles_for_plot, # Karşılık gelen çizgi stilleri
                           title, ylabel, output_dir, filename_prefix, plot_idx_ref, 
                           y_limit=(0, None)):
    """
    Belirtilen metrikleri tek bir figürdeki tek bir alt grafikte çizer.
    Her çağrıldığında yeni bir figür oluşturur.
    """
    valid_metrics_to_plot = [m for m in metrics_to_plot if m and m in df.columns]
    if not valid_metrics_to_plot:
        # print(f"UYARI: '{title}' için çizdirilecek geçerli sütun bulunamadı.")
        return

    plot_idx_ref[0] += 1
    
    plt.figure(figsize=(10, 6)) # Tek bir subplot için figür boyutu
    
    for i, col_name in enumerate(valid_metrics_to_plot):
        metric_label = labels_for_plot[metrics_to_plot.index(col_name)] # Orijinal listedeki indekse göre etiket al
        metric_color = colors_for_plot[metrics_to_plot.index(col_name)]
        metric_linestyle = linestyles_for_plot[metrics_to_plot.index(col_name)]
        plt.plot(df[epoch_col], df[col_name], label=metric_label, color=metric_color, linestyle=metric_linestyle)

    plt.title(f"{title} / Epoch", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    if y_limit[1] is not None: # Eğer bir üst limit varsa
        plt.ylim(y_limit[0], y_limit[1])
    else:
        plt.ylim(bottom=y_limit[0])
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        plot_filename = f"{plot_idx_ref[0]:02d}_{filename_prefix.replace('/', '_').replace('(', '').replace(')', '')}.png"
        try:
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
            print(f"  Grafik kaydedildi: {plot_filename}")
        except Exception as e:
            print(f"  HATA: Grafik '{plot_filename}' kaydedilemedi: {e}")
    plt.show()


def plot_all_yolo_training_metrics(results_csv_path, output_dir):
    if not os.path.exists(results_csv_path):
        print(f"HATA: results.csv dosyası bulunamadı: {results_csv_path}"); return
    try:
        df = pd.read_csv(results_csv_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        print("Okunan CSV'den sütun adları (temizlenmiş):"); print(df.columns.tolist())
    except Exception as e: print(f"HATA: CSV okunurken: {e}"); return
    if df.empty: print("UYARI: CSV boş."); return

    epoch_col = 'epoch'
    if epoch_col not in df.columns:
        if df.index.name == epoch_col or len(df) > 0: df[epoch_col] = df.index + 1
        else: print(f"HATA: '{epoch_col}' sütunu yok."); return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Grafikler şuraya kaydedilecek: {output_dir}")
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    plot_counter_list = [0]

    # --- Sütun Adlarını Tanımla (CSV'nize göre güncelleyin!) ---
    # Kayıplar
    train_box_loss = find_column_name(df.columns, ['train/box_loss'])
    val_box_loss = find_column_name(df.columns, ['val/box_loss'])
    train_seg_loss = find_column_name(df.columns, ['train/seg_loss'])
    val_seg_loss = find_column_name(df.columns, ['val/seg_loss'])
    train_cls_loss = find_column_name(df.columns, ['train/cls_loss'])
    val_cls_loss = find_column_name(df.columns, ['val/cls_loss'])
    train_dfl_loss = find_column_name(df.columns, ['train/dfl_loss'])
    val_dfl_loss = find_column_name(df.columns, ['val/dfl_loss'])

    # Metrikler (Doğrulama seti üzerinden)
    precision_b = find_column_name(df.columns, ['metrics/precision(B)'])
    recall_b = find_column_name(df.columns, ['metrics/recall(B)'])
    map50_b = find_column_name(df.columns, ['metrics/mAP50(B)'])
    map50_95_b = find_column_name(df.columns, ['metrics/mAP50-95(B)'])
    f1_b_col_name = 'F1-Score (Box)'

    precision_m = find_column_name(df.columns, ['metrics/precision(M)'])
    recall_m = find_column_name(df.columns, ['metrics/recall(M)'])
    map50_m = find_column_name(df.columns, ['metrics/mAP50(M)'])
    map50_95_m = find_column_name(df.columns, ['metrics/mAP50-95(M)'])
    f1_m_col_name = 'F1-Score (Mask)'

    # F1-Skorlarını hesapla
    if precision_b and recall_b and precision_b in df.columns and recall_b in df.columns:
        p_b, r_b = df[precision_b].astype(float), df[recall_b].astype(float)
        df[f1_b_col_name] = np.where((p_b + r_b) == 0, 0, 2 * (p_b * r_b) / (p_b + r_b))
    if precision_m and recall_m and precision_m in df.columns and recall_m in df.columns:
        p_m, r_m = df[precision_m].astype(float), df[recall_m].astype(float)
        df[f1_m_col_name] = np.where((p_m + r_m) == 0, 0, 2 * (p_m * r_m) / (p_m + r_m))

    # --- Grafik Çizdirme ---

    # 1. Kayıp Grafikleri (Her kayıp türü için Train ve Val aynı grafikte)
    plot_metrics_on_figure(df, epoch_col, [train_box_loss, val_box_loss], ['Train Box Loss', 'Val Box Loss'], ['royalblue', 'darkorange'], ['-', '--'], 'Box Kaybı', 'Kayıp', output_dir, "loss_box", plot_counter_list)
    plot_metrics_on_figure(df, epoch_col, [train_seg_loss, val_seg_loss], ['Train Seg Loss', 'Val Seg Loss'], ['royalblue', 'darkorange'], ['-', '--'], 'Segmentasyon Kaybı', 'Kayıp', output_dir, "loss_seg", plot_counter_list)
    plot_metrics_on_figure(df, epoch_col, [train_cls_loss, val_cls_loss], ['Train Cls Loss', 'Val Cls Loss'], ['royalblue', 'darkorange'], ['-', '--'], 'Sınıflandırma Kaybı', 'Kayıp', output_dir, "loss_cls", plot_counter_list)
    plot_metrics_on_figure(df, epoch_col, [train_dfl_loss, val_dfl_loss], ['Train DFL Loss', 'Val DFL Loss'], ['royalblue', 'darkorange'], ['-', '--'], 'DFL Kaybı', 'Kayıp', output_dir, "loss_dfl", plot_counter_list)

    # 2. Box Metrikleri (Her metrik için ayrı figür)
    if precision_b in df.columns: plot_metrics_on_figure(df, epoch_col, [precision_b], ['Box Precision (Val)'], ['purple'], ['-'], 'Box Precision (Val)', 'Skor (0-1)', output_dir, "metric_box_p", plot_counter_list, y_limit=(0,1.05))
    if recall_b in df.columns: plot_metrics_on_figure(df, epoch_col, [recall_b], ['Box Recall (Val)'], ['red'], ['-'], 'Box Recall (Val)', 'Skor (0-1)', output_dir, "metric_box_r", plot_counter_list, y_limit=(0,1.05))
    if f1_b_col_name in df.columns: plot_metrics_on_figure(df, epoch_col, [f1_b_col_name], ['Box F1-Score (Val)'], ['brown'], ['-'], 'Box F1-Skoru (Val)', 'Skor (0-1)', output_dir, "metric_box_f1", plot_counter_list, y_limit=(0,1.05))
    if map50_b in df.columns: plot_metrics_on_figure(df, epoch_col, [map50_b], ['Box mAP@0.50 (Val)'], ['green'], ['-'], 'Box mAP@0.50 (Val)', 'mAP', output_dir, "metric_box_map50", plot_counter_list, y_limit=(0,1.05))
    if map50_95_b in df.columns: plot_metrics_on_figure(df, epoch_col, [map50_95_b], ['Box mAP@0.50-0.95 (Val)'], ['blue'], ['-'], 'Box mAP@0.50-0.95 (Val)', 'mAP', output_dir, "metric_box_map50_95", plot_counter_list, y_limit=(0,1.05))

    # 3. Mask Metrikleri (Her metrik için ayrı figür)
    if precision_m in df.columns: plot_metrics_on_figure(df, epoch_col, [precision_m], ['Mask Precision (Val)'], ['purple'], ['-'], 'Mask Precision (Val)', 'Skor (0-1)', output_dir, "metric_mask_p", plot_counter_list, y_limit=(0,1.05))
    if recall_m in df.columns: plot_metrics_on_figure(df, epoch_col, [recall_m], ['Mask Recall (Val)'], ['red'], ['-'], 'Mask Recall (Val)', 'Skor (0-1)', output_dir, "metric_mask_r", plot_counter_list, y_limit=(0,1.05))
    if f1_m_col_name in df.columns: plot_metrics_on_figure(df, epoch_col, [f1_m_col_name], ['Mask F1-Score (Val)'], ['brown'], ['-'], 'Mask F1-Skoru (Val)', 'Skor (0-1)', output_dir, "metric_mask_f1", plot_counter_list, y_limit=(0,1.05))
    if map50_m in df.columns: plot_metrics_on_figure(df, epoch_col, [map50_m], ['Mask mAP@0.50 (Val)'], ['green'], ['-'], 'Mask mAP@0.50 (Val)', 'mAP', output_dir, "metric_mask_map50", plot_counter_list, y_limit=(0,1.05))
    if map50_95_m in df.columns: plot_metrics_on_figure(df, epoch_col, [map50_95_m], ['Mask mAP@0.50-0.95 (Val)'], ['blue'], ['-'], 'Mask mAP@0.50-0.95 (Val)', 'mAP', output_dir, "metric_mask_map50_95", plot_counter_list, y_limit=(0,1.05))

    print("Grafik çizdirme işlemi tamamlandı.")

if __name__ == '__main__':
    training_run_dir = r"runs/segment/cullet_seg_yolov8x_local_pt_run" 
    results_file = os.path.join(training_run_dir, "results.csv")
    output_graphics_subdir = os.path.join(training_run_dir, "individual_metrics_figs") # Farklı klasör adı

    if os.path.exists(results_file):
        plot_all_yolo_training_metrics(results_csv_path=results_file, output_dir=output_graphics_subdir)
    else:
        print(f"HATA: results.csv dosyası bulunamadı: {results_file}")