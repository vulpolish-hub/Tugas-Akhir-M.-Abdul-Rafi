# Spesifikasi Desain: Visualisasi & Dokumentasi Analisis Cloud Removal Satelit

Spesifikasi desain ini merinci pengembangan visual, tabel analisis statistika, serta pembuatan berkas `README.md` utama pada proyek Tugas Akhir pembersihan awan Sentinel-1 & Sentinel-2 menggunakan model ResUNet dan GAN.

---

## 1. Komponen Perubahan Sistem

### A. Preprocessing (`dataset_preprocessing_info.ipynb`)
- **Tujuan**: Menambahkan pembuktian visual mengenai masker awan spektral dan masukan radar.
- **Komponen**: Sel kode baru yang memuat satu baris perbandingan spasial citra satelit berukuran `[256, 256]`:
  1. **S2 RGB**: Citra Sentinel-2 asli (B4-B3-B2) berskala $[0, 1]$ yang tertutup awan.
  2. **S1 False Color**: Citra Sentinel-1 (VV, VH, |VV-VH|) untuk mendemonstrasikan penembusan radar.
  3. **Cloud Probability Map**: Representasi piksel $[0, 1]$ hasil rumus probabilitas awan spektral.
  4. **Soft Cloud Mask**: Masker awan halus yang terbentuk setelah pemotongan threshold ($0,08 - 0,20$).
  5. **Pseudo Ground Truth**: Hasil blending citra bersih komposit temporal.

### B. Evaluasi & Inferensi (`model_evaluation_inference.ipynb`)
- **Tujuan**: Memperdalam analisis statistika dengan stratified analysis dan menyertakan visual grid pembanding model.
- **Komponen 1: Analisis Kategori Awan**:
  - Sel kode baru yang mengelompokkan 15 sampel uji lokal berdasarkan rata-rata piksel masker awannya:
    - *Awan Ringan*: rata-rata masker $< 0,15$.
    - *Awan Sedang*: $0,15 \\le$ rata-rata masker $< 0,35$.
    - *Awan Tebal*: rata-rata masker $\\ge 0,35$.
  - Menampilkan rata-rata MAE model Baseline v14 vs GAN v9 pada setiap kategori dalam format DataFrame Pandas.
- **Komponen 2: Visual Comparison Grid**:
  - Menampilkan 2 sampel uji dalam bentuk grid: `[Cloudy Input | Pseudo Ground Truth | Baseline Pred | GAN Pred | Cloud Mask]`.

### C. Dokumentasi Utama (`README.md` di Root Folder)
- **Tujuan**: Menjadi panduan komprehensif repositori GitHub Tugas Akhir.
- **Komponen**: Penjelasan terstruktur dalam Bahasa Indonesia mencakup:
  1. *Preprocessing*: DN Scaling S2, dB Normalization S1, rumus Cloud Probability, dan pembentukan 64 kanal input.
  2. *Baseline*: Arsitektur ResUNet dan L1 Loss.
  3. *GAN*: Generator ResUNet + ResNet Discriminator, Hinge GAN Loss, dan Gradient Loss.
  4. *Evaluation*: Interpretasi MAE, RMSE, PSNR, SSIM, Pearson r, Paired t-test, dan Scatter Plot $R^2$.

---

## 2. Alur Data & Struktur Modul
```
[asiaWest_n/ (Raw TIFFs)]
      │
      ▼ (dataset_preprocessing_info.ipynb)
[Preprocessing: S2 DN/10000, S1 dB norm, Cloud Probability Map]
      │
      ▼
[Input Tensor: 64-Channel [64, 256, 256]]
      │
      ▼ (model_evaluation_inference.ipynb)
[Inferensi Model: Baseline & GAN CPU/GPU]
      │
      ├─► [Evaluasi Kuantitatif: MAE, RMSE, PSNR, SSIM, Pearson r]
      ├─► [Stratified Cloud Bucket Analysis: Light, Medium, Heavy Clouds]
      ├─► [Visual Grid Comparison: Cloudy, GT, Baseline, GAN, Mask]
      └─► [Uji Hipotesis: Paired t-test & Scatter Plot R²]
```

---

## 3. Rencana Pengujian
1. **Sintaks Validasi**: Menjalankan skrip validasi JSON untuk memastikan tidak ada sel notebook yang rusak pasca pengeditan.
2. **Visual Check**: Membuka berkas `dataset_preprocessing_info.ipynb` dan `model_evaluation_inference.ipynb` di VS Code untuk memastikan semua gambar matplotlib ter-render secara visual.
3. **Dokumentasi Check**: Memastikan berkas `README.md` ter-render sebagai markdown Github yang valid dan mudah dipahami.
