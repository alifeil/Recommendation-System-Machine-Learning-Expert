
# Laporan Proyek Machine Learning - Alifia Feiling A.S

## Daftar Isi

-   [Project Overview](#project-overview)
-   [Business Understanding](#business-understanding)
-   [Data Understanding](#data-understanding)
-   [Data Preparation](#data-preparation)
-   [Modeling](#modeling)
-   [Evaluation](#evaluation)
-   [Referensi](#referensi)

## Project Overview

<p align="center">
  <img width="500" height="250" src="https://static.republika.co.id/uploads/images/inpicture_slide/film_210811145904-108.jpg" alt="Sumber : https://www.republika.co.id/berita/qzi5w4463/sundance-film-festival-2021-digelar-pekan-depan">
</p>

Pada proyek ini, akan dibuat sebuah sistem rekomendasi film untuk pengguna. Film, juga dikenal sebagai movie, gambar hidup, film teater atau foto bergerak, merupakan serangkaian gambar diam, yang ketika ditampilkan pada layar akan menciptakan ilusi gambar bergerak karena efek fenomena phi. Ilusi optik ini memaksa penonton untuk melihat gerakan berkelanjutan antar objek yang berbeda secara cepat dan berturut-turut. Proses pembuatan film merupakan gabungan dari seni dan industri. [[1](https://id.wikipedia.org/wiki/Film)]. Menonton film adalah hal yang dilakukan oleh pengguna untuk mencari hiburan disaat hari kerja atau hari biasa tetapi pengguna membutuhkan rekomendasi dari opini pengguna lain dalam memilih sebuah film. Untuk itu proyek ini dibuat untuk memudahkan pengguna dalam mendapatkan rekomendasi film.

[← Kembali ke Daftar Isi](#daftar-isi)

## Business Understanding

### Problem Statements

Setelah mengetahui beberapa masalah diatas, berikut ini merupakan rincian masalah yang perlu diselesaikan di proyek ini:

-   Sistem rekomendasi apa yang diterapkan pada kasus ini?
-   Bagaimana cara membuat sistem rekomendasi film?

### Goals

Berikut adalah tujuan dari dibuatnya proyek ini:

-   Membuat sistem rekomendasi film untuk pengguna.
-   Memberikan rekomendasi hasil dari model _machine learning_ untuk pengguna.

### Solution approach

Gambar dibawah ini adalah diagram alir langkah-langkah yang dilakukan untuk melaksanakan proyek ini :

![Diagram Alir](https://user-images.githubusercontent.com/58651943/134550445-a2595ae1-89f7-439e-ac2f-b6a8d656ebc2.png)

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

-   Untuk bagian pra-pemrosesan data dilakukan beberapa teknik diantaranya :

    -   Menggabungkan dua data set antara rating dan judul untuk mengambil satu fitur kolom `title` di dataset judul.
    -   Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji

    Penjelasan lengkap mengenai persiapan data dapat dilihat lebih lengkap pada bagian _Data Preparation_.

-   Kemudian untuk sistem rekomendasi yang dibuat, dipilih sistem rekomendasi _collaborative filtering_ karena sesuai dengan datasetnya. Sehingga sistem rekomendasi dibuat untuk memberikan rekomendasi opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna. Beberapa algoritma yang digunakan untuk membuat sistem rekomendasi di proyek ini diantaranya :

**1. Algoritma Singular Value Decomposition (SVD)**
Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk Collaborative Filtering di sistem rekomendasi. Singular Value Decomposition (SVD) adalah metode faktorisasi matriks yang menggeneralisasikan dekomposisi eigen dari matriks persegi (nxn) ke matriks apa pun (nxm).(diterjemahkan dari [[2](https://towardsdatascience.com/simple-svd-algorithms-13291ad2eef2)]) Algoritma SVD dapat dilihat pada gambar dibawah ini:
   
   <p align="center">
  <img width="500" height="250" src="https://miro.medium.com/max/700/1*mo8loFarEKeNeVX49205-g.png" alt="Sumber : https://towardsdatascience.com/simple-svd-algorithms-13291ad2eef2">
</p>
                    
SVD adalah serupa dengan sebuah r untuk Principal Component Analysis (PCA), tetapi lebih umum. PCA mengasumsikan bahwa matriks persegi input, SVD tidak memiliki asumsi ini. Rumus umum SVD adalah:

M = UΣV ᵗ

Rumus diatas menjelaskan :
- M adalah matriks asli yang ingin kita dekomposisi
- U adalah matriks singular kiri (kolom adalah vektor singular kiri). U kolom berisi vektor eigen dari matriks MM ᵗ
- Σ adalah diagonal matriks yang mengandung singular (eigen) nilai.
- V adalah matriks singular kanan (kolom adalah vektor singular kanan). V kolom berisi vektor eigen dari matriks M ᵗ M2. 

**2. Algoritma K-Nearest Neighbors**
Algoritma KNN mengasumsikan bahwa hal serupa ada dalam jarak dekat. Dengan kata lain, hal-hal serupa dekat satu sama lain.(diterjemahkan dari [[2](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]) Algoritma KNN dapat dilihat pada gambar dibawah ini :
 
 <p align="center">
  <img width="500" height="250" src="https://miro.medium.com/max/611/1*wW8O-0xVQUFhBGexx2B6hg.png" alt="Sumber : https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761">
</p>

Perhatikan pada gambar di atas bahwa sebagian besar waktu, titik data yang serupa berdekatan satu sama lain. Algoritma KNN bergantung pada asumsi ini yang cukup benar untuk algoritma yang akan berguna. KNN menangkap ide kesamaan (terkadang disebut jarak, kedekatan, atau kedekatan).


[← Kembali ke Daftar Isi](#daftar-isi)

## Data Understanding

![dataset](https://user-images.githubusercontent.com/83399671/140976763-5bedf7f2-8a78-437a-9684-0f4618a9667d.png)


Tabel dibawah ini merupakan informasi dari dataset yang digunakan :

| Jenis                   | Keterangan                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)    |
| Lisensi                 | CC0: Public Domain                                                                              |
| Kategori                | Film                                                                                            |
| Rating Penggunaan       | 8.2 (Gold)                                                                                      |
| Jenis dan Ukuran Berkas | zip (227.8 MB)                                                                                  |

Gambar dibawah ini merupakan pratinjau dari dataset pada berkas :

**1. Dataset Rating  <code>ratings_small.csv</code>**

![dataset rating](https://user-images.githubusercontent.com/83399671/140977427-2ac96d0c-7cc4-4fa0-92ac-18bf896460d0.png)

**2. Dataset Judul <code>movies_metadata.csv</code>**

![dataset judul](https://user-images.githubusercontent.com/83399671/140977918-7856ef6d-2888-4468-98be-88287e9d796c.png)

**3. Menggabungkan 2 dataset yaitu Rating dan Judul**

![penggabungan dataset](https://user-images.githubusercontent.com/83399671/140978178-c446ff97-5388-4367-a71b-1fad0cc23399.png)


 Berikut ini adalah uraian variabel dari setiap kolom pada dataset :

1. Kolom <code>userId</code> merupakan variabel yang mengarah pada bagian spesifik dari pengguna (kode unik).

2. Kolom <code>movieId</code> merupakan variabel yang mengarah pada bagian spesifik dari film (kode unik).

3. Kolom <code>rating</code> merupakan kolom peringkat yang diberikan pengguna.

4. Kolom <code>timestamp</code> merupakan kolom waktu melakukan penilaian pengguna.

5. Kolom <code>title</code> merupakan kolom judul film.

Terakhir, kumpulan gambar dibawah ini merupakan visualisasi data dari dataset yang digunakan  :

-   Visualisasi data kosong

![data kosong](https://user-images.githubusercontent.com/83399671/140979276-6260c313-7a10-4a4a-945f-e89429e6eb8a.png)

-   Visualisasi korelasi antar kolom

![korelasi antar kolom](https://user-images.githubusercontent.com/83399671/140979381-1db1db6b-d3d0-412c-9444-86a16492941b.png)

-   Visualisasi data berdasarkan rating

![rating](https://user-images.githubusercontent.com/83399671/140979470-ba5ca1f8-ea04-4a46-b1d4-4e657c0bfd06.png)

[← Kembali ke Daftar Isi](#daftar-isi)

## Data Preparation

Seperti yang sudah dijelaskan pada bagian _Solution approach_, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :

-   Menggabungkan dua data set antara rating dan judul untuk mengambil satu fitur kolom `title` di dataset judul diperuntukan mengetahui judul film dari dataset lainnya dan kolom tersebut digabungkan dengan dataset rating.

-   Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
Pengujian performa model pada data sebenarnya, maka perlu dilakukan pembagian dataset kedalam dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji dengan rasio 80:20. Data latih dilakukan sepenuhnya untuk melatih model, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih.Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn

[← Kembali ke Daftar Isi](#daftar-isi)

## Modeling

Setelah dilakukan pra-pemrosesan data, selanjutnya adalah membuat sistem rekomendasi _collaborative filtering_. Model yang dibuat menggunakan library surprise, library surprise berasal dari scikit Python yang digunakan untuk membangun dan menganalisis sistem rekomendasi yang menangani data peringkat eksplisit. [[Library Surprise](http://surpriselib.com/)]

**1. Singular Value Decomposition (SVD)**

  Model ini dibangun menggunakan _library surprise_ yang menggunakan metode matrix factorization.
   
   ![model svd](https://user-images.githubusercontent.com/83399671/140980669-0bcd527d-ffe4-4cba-94bb-46c23e3a617c.png)

Terlihat dari gambar diatas model diatas terdapat  6 variabel prediksi yang dihasilkan yang terdiri dari : uid, iid, r_ui, est, details, dan reason. Reason akan ada jika hasil details menunjkan ‘True’.

**2. K-Nearest Neighbors (KNN)**

Model ini dibangun mnggunakan _library surprise_ didapatkan dari `from surprise import KNNBasic`, pada metode KNNBasic, penggunaan fungsi similarity dapat diaplikasikan.

![model knn](https://user-images.githubusercontent.com/83399671/140981446-a71a8fa2-c129-4473-8e52-5bdcb504f160.png)

Terlihat dari gambar diatas model diatas terdapat  6 variabel prediksi yang dihasilkan yang terdiri dari : uid, iid, r_ui, est, details, dan reason. Reason akan ada jika hasil details menunjkan ‘True’.


[← Kembali ke Daftar Isi](#daftar-isi)

## Testing

Pengujian dilakukan untuk menguji model dalam melakukan prediksi merekomendasikan data dari data film. Dibawah ini akan dijelaskan pengujian untuk model Singular Value Decomposition (SVD) dan K — Nearest Neighbors (KNN).

**1. Singular Value Decomposition (SVD)**

![pengujian svd](https://user-images.githubusercontent.com/83399671/140982751-3b5bd0c6-1500-4b4b-87d6-467fc29bb1bf.png)

Dari hasil keluaran contoh gambar diatas menunjukan bahwa memiliki kemungkinan untuk direkomendasikan kepada uid (userId) ‘1’ pada iid (filmId) ‘5’.

**2. K-Nearest Neighbors (KNN)**

![pengujian knn](https://user-images.githubusercontent.com/83399671/140983494-abd0a5a3-b1b2-44b7-b8af-94ee144984ee.png)

Dari hasil keluaran contoh gambar diatas menunjukkan bahwa username uid (userId) ‘30’ pada iid (filmId) ‘5655’ yang sama tidak bisa direkomendasikan.


[← Kembali ke Daftar Isi](#daftar-isi)

## Evaluation

Evaluasi dilakukan untuk mengetahui peforma akurasi dan error yang terjadi. Metode evaluasi yang digunakan meliputi RMSE, MAE, dan FCP untuk kedua algoritima yang digunakan, metode tersebut didapatkan dari library surprise yang dipanggil dengan menuliskan `from surprise import accuracy`.

**1. Singular Value Decomposition (SVD)**

![evaluasi svd](https://user-images.githubusercontent.com/83399671/140984267-806adc7a-f061-4882-8f68-11312037a0a9.png)
   
**2. K-Nearest Neighbors (KNN)**

![evaluasi knn](https://user-images.githubusercontent.com/83399671/140984354-d17d65bb-9bc6-4c0e-b254-7e3c9cbd7020.png)


[← Kembali ke Daftar Isi](#daftar-isi)

# Referensi

 A. Herve, "Singular Value Decomposition (SVD) and Generalized Singular Value Decomposition (GSVD)," Encyclopedia of Measurement and Statistic, 2007. 
 
Dzikrulloh, N, N. (2017). Penerapan Metode K-Nearest Neighbor (KNN) Dan Metode Weighted Product (WP) Dalam Penerimaan Calon Guru DanKaryawan Tata Usaha Baru Berwawasan Teknologi (Studi Kasus : Sekolah Menengah Kejuruan Muhammadiyah 2 Kediri). Malang : Program Studi Teknik Informatika Fakultas Ilmu Komputer Universitas Brawijaya. E-ISSN : 2548-964X.


[← Kembali ke Daftar Isi](#daftar-isi)
