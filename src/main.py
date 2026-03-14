# panggil model nya di sini nanti, ini main nya

import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model.network import NeuralNetwork
from model.layers import DenseLayer
from model.activation import ActivationLayer
from model.functions import (
    categorical_cross_entropy, cce_prime, relu, relu_prime, sigmoid, sigmoid_prime, 
    hyperbolic_tangent, hyperbolic_tangent_prime, linear, linear_prime,
    softmax, softmax_prime, mse, mse_prime, 
    binary_cross_entropy, bce_prime
)
from model.initializers import uniform_init, normal_init, zero_init

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_user_config():
    # fungsi untuk mengambil input user, nanti disimpan di dict config
    config = {}
    
    print("[ Neural Network by Ciamso ]")
    
    # Layer
    layers_str = input("Masukkan jumlah neuron tiap layer (pisahkan dengan spasi, contoh: 4 10 1): ")
    config['layers'] = [int(x) for x in layers_str.split()]
    
    # Fungsi Aktivasi
    print("\n [Fungsi Aktivasi]")
    print("\nPilihan Aktivasi: linear, relu, sigmoid, tanh, softmax")
    config['activation'] = input("Pilih aktivasi hidden layer: ").lower()
    config['output_activation'] = input("Pilih aktivasi output layer: ").lower()
    
    # Hyperparameters
    print("\n [Hyperparameters]")
    config['lr'] = float(input("Learning Rate (contoh 0.01): "))
    config['epochs'] = int(input("Jumlah Epoch: "))
    config['batch_size'] = int(input("Batch Size: "))
    config['loss'] = input("Fungsi Loss (mse / bce / cce): ").lower()
    
    # Inisialisasi
    print("\n [Inisialisasi]")
    config['init'] = input("Metode Inisialisasi (uniform / normal / zero): ").lower()
    config['seed'] = int(input("Masukkan Seed (angka): "))

    if config['init'] == 'normal':
        config['mean'] = float(input("Masukkan Mean (rata-rata): "))
        config['variance'] = float(input("Masukkan Variance: "))
    elif config['init'] == 'uniform':
        config['low'] = float(input("Masukkan Lower Bound: "))
        config['high'] = float(input("Masukkan Upper Bound: "))
    
    return config

def setup_model(config):
    # fungsi untuk setup model sesuai input user, nanti dipanggil di main
    model = NeuralNetwork()
    
    # Mapping fungsi
    act_map = {
        'relu': (relu, relu_prime), 'sigmoid': (sigmoid, sigmoid_prime),
        'tanh': (hyperbolic_tangent, hyperbolic_tangent_prime), 'linear': (linear, linear_prime),
        'softmax': (softmax, softmax_prime)
    }
    init_map = {'uniform': uniform_init, 'normal': normal_init, 'zero': zero_init}
    
    init_func = init_map[config['init']]
    act, act_p = act_map[config['activation']]
    out_act, out_act_p = act_map[config['output_activation']]
    
    # kwargs untuk initializer sesuai input user, nanti unpack di DenseLayer dengan **init_kwargs
    init_kwargs = {'seed': config['seed']}
    if config['init'] == 'normal':
        init_kwargs['mean'] = config['mean']
        init_kwargs['variance'] = config['variance']
    elif config['init'] == 'uniform':
        init_kwargs['low'] = config['low']
        init_kwargs['high'] = config['high']

    layers_dim = config['layers']
    for i in range(len(layers_dim) - 1):
        in_dim, out_dim = layers_dim[i], layers_dim[i+1]
        
        # Tambahkan layer Dense terlebih dahulu, baru Activation
        model.add(DenseLayer(in_dim, out_dim, init_func, **init_kwargs))
        
        if i == len(layers_dim) - 2:
            model.add(ActivationLayer(out_act, out_act_p))
        else:
            model.add(ActivationLayer(act, act_p))
            
    return model

def load_data_placement():
    # Sesuaikan path dengan struktur folder kamu
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    path = os.path.join(current_dir, 'data', 'datasetml_2026.csv')
    df = pd.read_csv(path)
    
    # Jika data categorical, encode dulu
    X = df.drop(columns=['placement_status'])
    y = df['placement_status']

    # kolom kategori dan numerik
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    # Preprocessing fitur kategori (One-Hot)
    encoder = OneHotEncoder(sparse_output=False)
    X_cat = encoder.fit_transform(X[cat_cols])
    
    # Preprocessing fitur numerik (Scaling)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])

    # Gabungkan kembali jadi satu
    X_final = np.hstack([X_num, X_cat])
    
    # Encode label target (0 dan 1)
    y_final = (y == 'Placed').astype(int).values.reshape(-1, 1)

    return train_test_split(X_final, y_final, test_size=0.2, random_state=42)

def main():
    clear_screen()
    
    # ambil input user untuk konfigurasi model
    user_params = get_user_config()
    
    # summary konfigurasi sebelum training
    print("\n" + "="*30)
    print("Konfigurasi Berhasil Disimpan!")
    print("="*30)
    
    # load data dan split, nanti bisa ditambah validasi juga
    print("\nSedang memproses dataset...")
    X_train, X_val, y_train, y_val = load_data_placement()

    # validasi input layer dengan jumlah fitur di data, kalau beda kasih warning dan sesuaikan otomatis
    if user_params['layers'][0] != X_train.shape[1]:
        print(f"\n[WARNING] Input layer di config ({user_params['layers'][0]}) berbeda dengan kolom data ({X_train.shape[1]}).")
        print("Menyesuaikan input layer secara otomatis...")
        user_params['layers'][0] = X_train.shape[1]

    model = setup_model(user_params)
    
    confirm = input("\n Data siap. Mulai training? (y/n): ")
    if confirm.lower() != 'y': return

    loss_map = {
        'mse': (mse, mse_prime), 
        'bce': (binary_cross_entropy, bce_prime),
        'cce': (categorical_cross_entropy, cce_prime)
    }
    loss_f, loss_p = loss_map[user_params['loss']]
    model.use(loss_f, loss_p)
    
    print(f"\n Training Dimulai (Total {user_params['epochs']} Epoch) ")
    
    # INI BELUM DITES, TADI MASIH ERROR 
    history = model.train(
        X_train, y_train,
        epochs=user_params['epochs'],
        lr=user_params['lr']
    )
    
    model.save("model_terakhir.pkl")
    
    print("\n" + "="*30)
    print("Selesai! Model disimpan sebagai 'model_terakhir.pkl'")
    print("="*30)
    
    # akurasi
    # y_pred = model.predict(X_val)

if __name__ == "__main__":
    main()