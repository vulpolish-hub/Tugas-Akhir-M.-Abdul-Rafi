# Tesis: Pembersihan Awan Citra Satelit Multi-Temporal Menggunakan Model ResUNet dan GAN (Studi Kasus: asiaWest_n)

Repositori ini berisi seluruh kode, eksperimen, hasil evaluasi, dan pemodelan untuk Tugas Akhir/Tesis mengenai restorasi citra satelit Sentinel-2 yang tertutup awan menggunakan bantuan radar Sentinel-1 pada dataset **SEN12MS-CR-TS (wilayah Asia Barat / `asiaWest_n`)**.

Studi ini membandingkan dua pendekatan model:
1. **Model Baseline v14**: Multi-Temporal ResUNet dengan optimasi supervised murni (L1 Loss + Gradient Loss).
2. **Model GAN v9**: Generator Multi-Temporal ResUNet + Diskriminator ResNet-18 dengan Spectral Normalization (Adversarial Hinge Loss + L1 Loss + Gradient Loss).

---

## Daftar Isi Proyek

Secara garis besar, proyek ini terbagi menjadi empat tahap/notebook utama:

1. **Preprocessing Citra Spasial & Pembuatan Input (Lokal)**: [`dataset_preprocessing_info.ipynb`](dataset_preprocessing_info.ipynb)
2. **Pelatihan Model Baseline ResUNet (Kaggle)**: [`sen12ms-cr-ts-cloud-removal-eda-baseline-unet.ipynb`](kaggle_sen12_eda_unet/sen12ms-cr-ts-cloud-removal-eda-baseline-unet.ipynb)
3. **Pelatihan Model GAN ResNet (Kaggle)**: [`sen12ms-cr-ts-cloud-removal-gan.ipynb`](kaggle_sen12_gan_resnet/sen12ms-cr-ts-cloud-removal-gan.ipynb)
4. **Evaluasi Statistik & Inferensi (Lokal)**: [`model_evaluation_inference.ipynb`](model_evaluation_inference.ipynb)

---

## 1. Preprocessing Citra Satelit (`dataset_preprocessing_info.ipynb`)

Tahap ini mencakup pembacaan file TIFF 16-bit satelit, normalisasi piksel, pembuatan peta awan, dan perakitan tensor 64 kanal input:

*   **Sentinel-2 (Optik RGB-NIR)**:
    *   Nilai piksel asli direpresentasikan dalam Digital Number (DN) berskala $0 - 10000$.
    *   **Normalisasi**: Dibagi dengan `10000.0` agar bernilai $[0, 1]$ spektral reflectance.
    *   Citra temporal terdiri dari 4 tanggal perekaman. Masing-masing tanggal memiliki 13 band. Total saluran optik = $13 \times 4 = 52$ kanal.
*   **Sentinel-1 (Radar SAR VV/VH)**:
    *   Nilai piksel asli berupa koefisien hamburan balik dalam skala desibel (dB) berkisar dari $-35\text{ dB}$ s.d $+5\text{ dB}$.
    *   **Normalisasi**: Dikonversi menjadi skala linier $[0, 1]$ menggunakan rumus:
        $$\text{S1}_{\text{norm}} = \frac{\text{S1} + 35,0}{40,0}$$
    *   Terdiri dari 2 band (VV dan VH) $\times$ 4 tanggal perekaman. Total saluran radar = $2 \times 4 = 8$ kanal.
*   **Peta Probabilitas Awan (Cloud Probability)**:
    *   Dihitung secara spektral (menggunakan NDSI, NDVI, Whiteness, dan Cirrus/SWIR bands) serta temporal (kontras kecerahan terhadap median deret waktu).
    *   Terdiri dari 1 peta probabilitas per tanggal. Total saluran probabilitas awan = $1 \times 4 = 4$ kanal.
*   **Struktur Input (64 Kanal)**:
    *   Gabungan seluruh saluran di atas menghasilkan input berukuran `[64, 256, 256]` untuk dimasukkan ke model deep learning.

---

## 2. Pelatihan Model Baseline ResUNet (`sen12ms-cr-ts-cloud-removal-eda-baseline-unet.ipynb`)

Notebook ini dijalankan di server GPU Kaggle untuk melatih model Baseline (Versi 14):
*   **Arsitektur**: Multi-Temporal ResUNet (Encoder-Decoder dengan skip-connection + residual block).
*   **Fungsi Loss**:
    *   **L1 Loss**: Mengoptimalkan kemiripan piksel matematis secara langsung.
    *   **Gradient Loss**: Memastikan pemulihan tepi/tekstur fisik yang tajam.
*   **Hasil Terbaik (Validation Set)**: Cloud MAE = `0.0108` (Akurasi Piksel Spektral mencapai **98,92%**), PSNR = `37,74 dB`.

---

## 3. Pelatihan Model GAN (`sen12ms-cr-ts-cloud-removal-gan.ipynb`)

Notebook ini melatih model berbasis Generative Adversarial Network (GAN) di server GPU Kaggle (Versi 9):
*   **Arsitektur Generator**: Multi-Temporal ResUNet.
*   **Arsitektur Diskriminator**: ResNet-18 dengan **Spectral Normalization** untuk menstabilkan latihan adversarial.
*   **Fungsi Loss Gabungan**:
    *   $L_{\text{adv}}$ (Adversarial Hinge Loss): Mendorong generator memproduksi visual spasial yang tajam dan realistis.
    *   $L_1$ Loss (MAE): Menjaga keselarasan piksel matematis terhadap target.
    *   $L_{\text{grad}}$ (Gradient Loss): Menghindari blur spasial pada wilayah transisi tanah.
*   **Hasil Terbaik (Validation Set)**: Cloud MAE = `0.01279` (Akurasi Piksel Spektral mencapai **98,72%**), PSNR = `37,25 dB`.

---

## 4. Evaluasi Inferensi & Uji Statistika (`model_evaluation_inference.ipynb`)

Dijalankan secara lokal untuk membandingkan model Baseline v14 dan GAN v9 secara ilmiah:

*   **Metrik Evaluasi Piksel**:
    *   **MAE / RMSE**: Mengukur tingkat penyimpangan piksel.
    *   **PSNR**: Mengukur rasio derau rekonstruksi dalam desibel (dB) ($>30\text{ dB}$ menandakan noise sangat rendah).
    *   **SSIM**: Mengukur tingkat kemiripan tekstur visual berdasarkan kontras dan pencahayaan spasial lokal ($0 - 1$).
    *   **Korelasi Pearson**: Menunjukkan keeratan hubungan linier piksel prediksi terhadap target.
*   **Uji t Berpasangan (Paired t-test)**:
    *   Menguji apakah perbedaan performa MAE antara Baseline dan GAN signifikan secara statistik.
    *   Hasil uji menunjukkan **P-Value = 0,0119** (di bawah $\alpha = 0,05$), yang berarti perbedaan performa tersebut **signifikan secara statistika**.
*   **Analisis Koefisien Determinasi ($R^2$)**:
    *   Melalui scatter plot piksel, model Baseline menghasilkan $R^2 \approx 0,98$ (menjelaskan **98%** variabilitas piksel target) dan GAN menghasilkan $R^2 \approx 0,96$ (menjelaskan **96%** variabilitas piksel target).
*   **Analisis Kategori Awan (Stratified)**:
    *   Notebook mengelompokkan sampel menjadi Awan Ringan ($<15\%$), Sedang ($15-35\%$), dan Tebal ($\ge 35\%$) untuk melihat performa detail di berbagai kondisi spasial.

---

## Analisis Trade-Off Model (Untuk Sidang/Laporan)

Secara statistika dan numerik, **Baseline v14** mengungguli GAN v9 sebesar ~0.2% akurasi piksel. Namun, secara visual dan persepsi mata manusia, **GAN v9** jauh lebih unggul karena mampu merestorasi tekstur permukaan bumi (sawah, jalan, pemukiman) secara tajam di bawah awan tebal, sedangkan Baseline v14 cenderung menghasilkan citra yang agak kabur (*blurry*). Trade-off ini dikenal sebagai **Perception-Distortion Trade-off** dalam pengolahan citra digital.
