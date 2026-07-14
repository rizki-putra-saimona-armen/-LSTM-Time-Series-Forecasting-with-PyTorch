#  LSTM Time Series Forecasting with PyTorch

Proyek peramalan (forecasting) data time series menggunakan **Long Short-Term Memory (LSTM)** — varian dari RNN yang lebih baik dalam menangkap dependensi jangka panjang — dibangun dengan **PyTorch** dan dibantu library [`jcopdl`](https://pypi.org/project/jcopdl/) untuk custom dataset, konfigurasi, dan callback training.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Proyek ini merupakan lanjutan dari eksperimen **RNN forecasting**, kali ini menggunakan **LSTM** — arsitektur yang dirancang untuk mengatasi kelemahan RNN standar (*vanishing gradient*) dalam mempelajari pola jangka panjang. Studi kasusnya tetap sama: **memprediksi suhu harian minimum** (*daily minimum temperature*), namun dengan eksplorasi lebih dalam terkait pengaruh panjang sequence (`seq_len`) dan jarak horizon forecasting terhadap akurasi prediksi.

Proyek ini cocok sebagai referensi belajar:
- Perbedaan implementasi **LSTM** vs **RNN** biasa di PyTorch (`nn.LSTM` vs `nn.RNN`)
- Pengaruh panjang **sequence length** terhadap kualitas forecasting
- Konsep **context window** (`n_prior`) dan **horizon forecasting** (`n_forecast`) dalam prediksi rekursif jangka panjang
- Kenapa **ketidakpastian meningkat** seiring bertambahnya jarak prediksi ke masa depan

---

##  Struktur Proyek

```
.
├── LSTM_With_Pytorch.ipynb   # Notebook utama
├── data/
│   └── daily_min_temp.csv                  # Dataset suhu harian
├── utils.py                                 # Helper: data4pred & pred4pred
├── model/
│   └── LSTM/                                # Output checkpoint model (otomatis dibuat)
└── README.md
```

> Dataset berupa data deret waktu (time series) univariat dengan satu kolom target: **`Temp`**.

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn tqdm jupyter
pip install jcopdl
```

>  Pastikan path dataset (`daily_min_temp.csv`) disesuaikan dengan lokasi file di komputer kamu — pada notebook asli path masih menggunakan direktori lokal Windows.

---

##  Alur Data (Data Pipeline)

1. **Load & Parse Date** — data suhu harian dibaca dengan `Date` sebagai index, di-*parse* sebagai datetime.
2. **Train-Test Split** — data dibagi menjadi `ts_train` dan `ts_test` dengan `shuffle=False` (data time series **tidak boleh diacak**).
3. **Windowing / Sequencing** — menggunakan `TimeSeriesDataset` dari `jcopdl` dengan `seq_len = 16` (setara ±2 minggu konteks), lebih panjang dibanding eksperimen RNN sebelumnya (`seq_len = 6`) untuk memberi model lebih banyak informasi historis.

---

##  Arsitektur Model

```python
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden
```

**Perbedaan utama dari RNN:**
- Cukup mengganti `nn.RNN` menjadi `nn.LSTM` — PyTorch sudah menyediakan LSTM block secara built-in, tanpa perlu implementasi manual gate-gate LSTM (forget, input, output gate).

**Konfigurasi model:**

| Parameter      | Nilai |
|-----------------|-------|
| `input_size`    | Jumlah fitur (1, hanya kolom `Temp`) |
| `hidden_size`   | 64 |
| `num_layers`    | 2 |
| `dropout`       | 0 |
| `output_size`   | 1 |
| `seq_len`       | 16 |
| `batch_size`    | 8 |

---

##  Training

| Komponen | Detail |
|---|---|
| **Loss function** | `MSELoss` (Mean Squared Error) — cocok untuk masalah regresi |
| **Optimizer** | `AdamW`, learning rate `0.001` |
| **Early stopping** | Otomatis berdasarkan `test_cost` via `jcopdl.callback` |

Jalankan seluruh sel notebook secara berurutan, atau langsung via terminal:

```bash
jupyter notebook Part_4_-_LSTM_With_Pytorch_copy.ipynb
```

Selama training, akan tampil progress bar per epoch serta plot cost secara real-time, dengan checkpoint model otomatis tersimpan di `model/LSTM/`.

---

##  Forecasting & Eksperimen

Notebook ini melakukan beberapa eksperimen forecasting untuk memahami batas kemampuan model:

### 1️ Evaluasi Dasar (`data4pred` vs `pred4pred`)
- **`data4pred`** — one-step prediction dengan data historis asli sebagai input di setiap langkah → hasil sangat baik.
- **`pred4pred`** — multi-step forecasting rekursif menggunakan hasil prediksi sebelumnya sebagai input berikutnya → lebih menantang dan realistis.

### 2️ Long-Horizon Forecasting
Model diuji memprediksi jauh ke depan dengan variasi context window:

| Eksperimen | `n_prior` (konteks awal) | `n_forecast` (langkah prediksi) | Hasil |
|---|---|---|---|
| Percobaan 1 | 10 | 140 | Prediksi awal cukup baik, namun makin meleset seiring bertambahnya horizon |
| Percobaan 2 | 30 | 120 | Konteks lebih panjang membantu, tapi ketidakpastian tetap meningkat di horizon jauh |

> 💡 **Insight kunci:** Dengan `seq_len = 16`, model mendapat cukup informasi historis sehingga performa forecasting lebih baik dibanding RNN dengan `seq_len` pendek. Namun, prinsip dasar tetap berlaku — **semakin jauh horizon prediksi ke masa depan, semakin tinggi ketidakpastiannya**. Ini bukan kegagalan model, melainkan sifat alami dari time series forecasting: machine learning hanya mencari pola dari data yang tersedia, bukan memprediksi kepastian mutlak.

---

##  Tech Stack

- **PyTorch** — Deep learning framework (`nn.LSTM`)
- **jcopdl** — Custom dataset, config, dan callback management
- **pandas** & **NumPy** — Manipulasi data time series
- **Matplotlib** — Visualisasi tren & hasil forecasting
- **scikit-learn** — Train-test split
- **tqdm** — Progress bar training

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* jika ingin membantu mengembangkan proyek ini — terutama eksperimen dengan fitur musiman (*seasonal features*) atau arsitektur Bidirectional LSTM untuk meningkatkan akurasi forecasting jangka panjang.

---

<p align="center">Made with ❤️ using PyTorch</p>
