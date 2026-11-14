# Tangis: Klasifikasi Arti Tangisan Bayi (CNN log-mel + baseline DTW)

Model mengenali 9 kelas suara bayi (lapar, sakit perut, ingin sendawa, kedinginan/kepanasan, tidak nyaman, lelah, tertawa, bising, hening) dari dataset Kaggle [Baby Crying Sounds](https://www.kaggle.com/datasets/mennaahmed23/baby-crying-sounds-dataset). Website di `docs/` menjalankan model langsung di browser (onnxruntime-web), jadi bisa di-host gratis di GitHub Pages.

## Isi folder

| Path | Isi |
|---|---|
| `baby_cry.ipynb` | Audit data, EDA, baseline DTW 1-NN, pelatihan CNN, evaluasi, ekspor ONNX, uji muat ulang |
| `cry_utils.py` | Kode bersama: inventaris & pembersihan data, split per grup, `load_wave`, MFCC, DTW, model `CryCNN` |
| `docs/` | Website statis (`index.html`, `style.css`, `app.js`, `model.onnx`) |
| `models/` | `cry_cnn.pt` (bobot PyTorch), `history.csv`, `perbandingan.csv` |
| `figures/` | Grafik notebook dan tangkapan layar website |
| `test_site.py` | Uji website di Chrome headless (unggah, rekam mic palsu, file rusak) |
| `laporan_baby_cry.md` / `.docx` | Laporan lengkap |
| `plans/` | Rencana implementasi dan catatan keputusan |

## Menjalankan ulang

```bash
pip install -r requirements.txt
python cry_utils.py                      # self-check -> ok
python -m nbconvert --execute --to notebook --inplace baby_cry.ipynb   # unduh data, latih, ekspor docs/model.onnx
python test_site.py                      # uji website -> ok
```

Coba website di komputer sendiri: `python -m http.server -d docs 8000` lalu buka <http://localhost:8000>. Membuka `index.html` langsung (file://) tidak bisa karena browser memblokir pemuatan model.

## Deploy ke GitHub Pages

1. Buat repository baru di GitHub. `.gitignore` hanya meloloskan file website (`docs/index.html`, `style.css`, `app.js`, `model.onnx`) dan README; data, notebook, laporan, dan file lain tetap lokal.
   ```bash
   cd baby_cry
   git init
   git add .gitignore README.md docs/index.html docs/style.css docs/app.js docs/model.onnx
   git commit -m "Tangis: klasifikasi tangisan bayi"
   git branch -M main
   git remote add origin https://github.com/<username>/<nama-repo>.git
   git push -u origin main
   ```
2. Di GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder `/docs` → Save**.
3. Tunggu 1-2 menit, website aktif di `https://<username>.github.io/<nama-repo>/`. GitHub Pages memakai HTTPS, jadi tombol rekam bisa meminta izin mikrofon.

## Catatan

Alat bantu edukasi, bukan alat diagnosis. Akurasi uji 77,8% (macro-F1 0,52), jauh di atas baseline DTW (39,5%). Kelas kedinginan/kepanasan, tidak nyaman, dan lelah hanya punya 15-24 rekaman asli dan belum terkenali, jadi hasilnya perlu dibaca bersama tanda-tanda lain pada bayi. Detail di `laporan_baby_cry.docx`.
