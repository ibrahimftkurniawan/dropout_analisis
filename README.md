# Proyek Akhir: Prediksi Dropout Mahasiswa Jaya Jaya Institute

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Meskipun memiliki reputasi yang baik dalam mencetak lulusan berkualitas, institusi ini menghadapi masalah tingginya tingkat dropout mahasiswa. Hal ini menjadi perhatian serius karena dapat mempengaruhi reputasi, kinerja akademik, dan keberlanjutan finansial institusi.

Proyek ini bertujuan untuk membantu Jaya Jaya Institut mendeteksi secara dini mahasiswa yang berisiko dropout dengan mengembangkan model machine learning prediktif. Melalui sistem prediksi ini, institusi dapat memberikan bimbingan khusus kepada mahasiswa yang teridentifikasi berisiko tinggi, sehingga tingkat dropout dapat dikurangi.

### Permasalahan Bisnis

Bagaimana mendeteksi dini mahasiswa yang berpotensi dropout (putus kuliah) sejak semester pertama?

### Cakupan Proyek

1. **Pengumpulan Data:** Mengumpulkan data yang berisi informasi terkait mahasiswa, termasuk jalur akademik, demografi, sosial ekonomi, performa akademik, kegiatan ekstrakurikuler dan kebiasaan belajar.
2. **Data Understanding:** Melakukan eksplorasi data untuk memahami karakteristik dataset, melakukan analisis distribusi variabel target (status dropout), melakukan identifikasi korelasi antar fitur serta melakukan analisis fitur-fitur yang potensial mempengaruhi keputusan dropout.
3. **Data Preparation** : Melakukan pembersihan data, penanganan missing values, transformasi fitur, melakukan feature encoding untuk variabel kategorikal untuk memastikan data siap digunakan dalam pengembangan model machine learning, serta melakukan pemisahan data training dan testing.
4. **Pengembangan Model:** Membangun model baseline (Logistic Regression), mengembangkan model lanjutan (Random Forest, SVM, Gradient Boosting, XGBoost), melakukan tuning hyperparameter untuk meningkatkan performa model serta melakukan perbandingan performa model.
5. **Evaluasi:** Mengukur kinerja model yang dikembangkan menggunakan metrik evaluasi yang relevan (seperti akurasi, presisi, recall dan f1 score) serta melakukan analisis lebih lanjut untuk memastikan model memenuhi kebutuhan bisnis dan akurasi yang diharapkan.
6. **Deployment** : Membuat dashboard monitoring menggunakan Metabase dan membuat aplikasi prediksi berbasis web menggunakan Streamlit

### Persiapan

Dataset yang digunakan dalam proyek ini berisi informasi historis mahasiswa Jaya Jaya Institute, termasuk data demografis, akademik, dan status dropout. Dataset dapat diakses melalui [tautan ini](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv).

Setup environment:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, accuracy_score, precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
```

## Business Dashboard

Business dashboard dibuat menggunakan Metabase yang dapat diakses dengan kredensial berikut:

* URL: https://metabase.xxxxxxx.com
* Email: baim@xxxxxxx.com
* Password: xxxxxxxxx

Langkah-langkah mengakses dashboard adalah sebagai berikut.

1. Buka browser Google Chrome atau lainnya.
2. Masukkan URL seperti diatas
3. Login menggunakan email dan password seperti di atas.
4. Pada Bagian Collection (sebelah kiri), pilih Dicoding
5. Pilih Dashboard yang ini di lihat. Adapun list dashboard yang dapat diakses dapat dilihat di bawah ini.

Terdapat beberapa dashboard yang dibuat antara lain sebagai berikut:

1. **Student Performance Overview**
   Dashboard ini bertujuan untuk melacak tingkat keberhasilan mahasiswa secara keseluruhan, demografi, dan indikator performa utama. Adapun informasi yang terdapat pada dashboard ini adalah sebagai berikut.
   a. **Graduation Rate KPI** : menampilkan persentase tingkat kelulusan. Informasi ini berguna untuk melihat seberapa efektif program pendidikan secara keseluruhan
   b. **Student Status by Gender** : menampilkan distribusi status mahasiswa (enrolled, graduate, dropout) berdasarkan jenis kelamin, Informasi ini membantu mengidentifikasi apakah ada kesenjangan gender dalam tingkat kelulusan
   c. **Average Grades by Course** : menampilkan perbandingan nilai rata-rata antar program studi. Informasi ini membantu mengidentifikasi program studi dengan performa akademik tertinggi dan terendah
   d. **Age Distribution** : menampilkan dan mengelompokkan mahasiswa berdasarkan kelompok usia saat mendaftar dan status mahasiswa saat ini. Informasi ini membantu memahami bagaimana usia saat pendaftaran memengaruhi hasil pendidikan
   e. **Performance by Scholarship Status** : menampilkan perbandingan performa mahasiswa yang menerima beasiswa dengan yang tidak menerima beasiswa. Informasi ini membantu mengukur efektivitas program beasiswa dalam meningkatkan keberhasilan akademik
   f. **Course Popularity** : menampilkan program studi berdasarkan jumlah pendaftaran. Informasi ini membantu untuk mengukur tingkat keberhasilan untuk setiap program studi
2. **Dropout Risk Analysis**
   Dashboard ini bertujuan untuk mengidentifikasi faktor-faktor yang terkait dengan dropout untuk membantu intervensi dini. Adapun informasi yang terdapat pada dashboard ini adalah sebagai berikut.
   a. **Dropout Rate KPI** : menampilkan persentase tingkat putus kuliah (dropout). Informasi ini berguna untuk mengukur tingkat retensi mahasiswa.
   b. **Dropout by Academic Performance** : menampilkan tingkat dropout berdasarkan keberhasilan mahasiswa pada semester pertama. Informasi ini berguna untuk menunjukkan hubungan antara kinerja akademik awal semerster dengan risiko dropout.
   c. **Dropout by Economic Factors** : menampilkan informasi pengaruh utang dan status pembayaran biaya kuliah terhadap tingkat dropout. Informasi ini berguna untuk mengidentifikasi hambatan ekonomi yang memengaruhi keberlanjutan pendidikan
   d. **Dropout by Parent Education Level** : menampilkan informasi pengaruh tingkat pendidikan ibu dan ayah terhadap risiko dropout. Informasi ini berguna untuk mengidentifikasi apakah latar belakang pendidikan keluarga memengaruhi keberhasilan pendidikan
   e. **Dropout Rate vs Unemployment Rate** : menampilkan perbandingan tingkat putus kuliah dengan tingkat pengangguran. Informasi ini berguna untuk menganalisis bagaimana faktor ekonomi makro dapat mempengaruhi keputusan mahasiswa untuk melanjutkan pendidikan

## Menjalankan Sistem Machine Learning

Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

1. Download project lalu ekstrak
2. Masuk ke dalam directory project
3. Pastikan python telah terinstall dan package manager pip telah terinstall
4. Install library yang dibutuhkan
   ```
   pip install -r requirements.txt
   ```
5. Jalankan aplikasi dengan perintah
   ```
   streamlit run app.py
   ```

Untuk melihat aplikasi secara online dapat melalui link berikut.

## Conclusion

Berdasarkan analisis data dan insight yang telah diperoleh, beberapa faktor kunci yang dapat memprediksi potensi mahasiswa untuk putus kuliah (dropout) adalah sebagai berikut:

1. **Gambaran Umum**
   a. Tingkat kelulusan saat ini adalah 49,93% dengan tingkat dropout yang cukup tinggi yaitu 32,12%.
   b. Dari analisis gender, perempuan memiliki tingkat kelulusan lebih tinggi dibandingkan laki-laki, namun juga memiliki tingkat dropout yang lebih tinggi.
2. **Faktor Akademik**
   a. Performa akademik sangat mempengaruhi dropout: mahasiswa yang tidak lulus satupun mata kuliah memiliki tingkat dropout tertinggi
   b. Mahasiswa dengan tingkat kelulusan 50-89% juga memiliki risiko dropout tinggi, menunjukkan bahwa kesulitan parsial juga bermasalah
3. **Faktor Ekonomi**
   a. Mayoritas mahasiswa yang dropout (78%) tidak memiliki hutang pendidikan
   b. Mahasiswa dengan status "Fees Up to Date" justru memiliki jumlah dropout lebih tinggi dibanding yang belum melunasi biaya kuliah
   c. Ini menunjukkan bahwa faktor ekonomi bukanlah faktor utama penyebab dropout
4. **Faktor Demografis**
   a. Kelompok usia "Under 20" memiliki tingkat kelulusan tertinggi
   b. Kelompok usia "20-25" memiliki tingkat dropout yang signifikan
   c. Latar belakang pendidikan orang tua sangat berpengaruh.
5. **Faktor Program Studi**
   a. Program studi Keperawatan (Nursing) memiliki tingkat pendaftaran tinggi dan tingkat keberhasilan yang baik
   b. Beberapa program dengan tingkat keberhasilan tinggi tetapi pendaftaran rendah, menunjukkan potensi untuk pengembangan program
   c. Status beasiswa berpengaruh signifikan: mahasiswa tanpa beasiswa memiliki tingkat kelulusan lebih tinggi
6. **Korelasi dengan Faktor Eksternal**
   a. Ada korelasi positif antara tingkat pengangguran dan tingkat dropout, menunjukkan bahwa kondisi ekonomi makro berpengaruh terhadap keberlangsungan studi

### Rekomendasi Action Items

Berikut adalah rekomendasi action items untuk Jaya Jaya Institute berdasarkan hasil analisis:

1. **Implementasi Sistem Early Warning**:
   a) Integrasikan model prediksi ke dalam sistem akademik
   b) Set up notifikasi otomatis untuk mahasiswa berisiko tinggi
   c) Review status mahasiswa secara berkala (setiap bulan)
2. **Program Intervensi Terstruktur**:
   a) Bentuk tim respons cepat untuk menangani mahasiswa berisiko tinggi
   b) Kembangkan program mentoring khusus dengan rasio mentor-mentee kecil
   c) Sediakan dukungan psikologis dan konseling karir
3. **Penyesuaian Kebijakan Akademik**:
   a) Revisi kebijakan kehadiran dan evaluasi
   b) Kembangkan sistem remedial yang lebih fleksibel
   c) Tinjau ulang beban studi untuk program tertentu
4. **Pelatihan Staf**:
   a) Latih dosen dan staf akademik untuk mengenali tanda-tanda awal dropout
   b) Kembangkan protokol eskalasi untuk kasus berisiko tinggi
   c) Workshop regular tentang teknik mentoring dan bimbingan
5. **Review dan Perbaikan Berkelanjutan**:
   a) Evaluasi efektivitas intervensi setiap semester
   b) Update model prediksi dengan data baru
   c) Lakukan survei exit untuk mahasiswa dropout untuk insight tambahan
