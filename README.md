# Tugas Besar 1 IF3270
Tugas Besar 1 IF3270 Pembelajaran Mesin Feedforward Neural Network

## Anggota Kelompok (Kelompok 9 - ciamso)
| No | NIM | Nama | Tugas |
| :---: | :---: | :---: | :---: |
| 1 | Anella Utari Gunadi | 13523078 | softmax, BCE, Network Layer, RMSNorm, Adam Optimizer, pengujian, laporan, debug |
| 2 | Nayla Zahira | 13523079 | Linear, Sigmoid, CCE , Dense Layer, pengujian, laporan, debug |
| 3 | Diyah Susan Nugrahani | 13523080 |  ReLU, Hyperbolic, Initializer, MSE, Activation Layer, init repo, pengujian, laporan, debug |

## Deskripsi Program 
Repositori ini berisi implementasi Feedforward Neural Network from scratch dengan menggunakan bahasa python dan library numpy. Program ini dirancang untuk memecahkan masalah klasifikasi biner untuk memprediksi apakah seorang mahasiswa akan mendapatkan pemempatan kerja atau tidak berdasarkan data profil akademik dan teknis yang ada pada dataset.

## Detail Implementasi
1. Fungsi Aktivasi
   - Linar
   - ReLU
   - Sigmoid
   - Hyperbolic Tangent
   - Softmax
2. Fungsi Loss
   - MSE
   - Binary Cross-Entropy
   - Categorical Cross-Entropy
  3. Initialization
     - Zero initialization
     - Random dengan distribusi uniform
     - Random dengan distribusi normal
  4. Bonus
     - RMSNormLayer
     - Xavier initialization
     - He initialization
     - Optimizer Adam
       
## Cara Menjalankan Program
1. Clone repository
``` bash   
git clone https://github.com/DiyahSusan/Tubes1-IF3270-ciamso
```
2. Import FFNN dari folder src.model
3. Gunakan model seperti menggunakan library biasa
```
# contoh penggunaan

from model.network import FFNN

model = FFNN(layers=layer_sizes, activations=activations, init_func=init_funcs)
```
4. Untuk pengujian, jalankan test.ipynb pada folder src
   

