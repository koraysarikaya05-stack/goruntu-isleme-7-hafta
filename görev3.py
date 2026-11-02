import numpy as np
import matplotlib.pyplot as plt

# --- 1. ADIM: SİNYAL VE GÜRÜLTÜ OLUŞTURMA ---
N = 256
t = np.linspace(0, 1, N, endpoint=False)
temiz_vektor = 5.0 * np.sin(2 * np.pi * 3.0 * t + np.pi / 2)

amplitude_range = np.max(temiz_vektor) - np.min(temiz_vektor)
noise = np.random.uniform(-amplitude_range * 0.25 / 2, amplitude_range * 0.25 / 2, size=N)
gurultulu_vektor = temiz_vektor + noise

# --- 2. ADIM: NUMPY İLE FİLTRELEME ---

# A) Gauss Filtresi (Yumuşatma çekirdeği ile konvolüsyon)
window_size = 5
sigma = 1.5
x = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
gauss_kernel = np.exp(-x**2 / (2 * sigma**2))
gauss_kernel = gauss_kernel / np.sum(gauss_kernel) # Normalize etme

# Padding ile kenarları koruyarak konvolüsyon yapıyoruz
gauss_temizlenmis = np.convolve(gurultulu_vektor, gauss_kernel, mode='same')

# B) Medyan Filtresi (Kayan pencere ile ortanca değer alma)
k = 5  # Pencere boyutu (tek sayı)
pad_width = k // 2
padded_vector = np.pad(gurultulu_vektor, pad_width, mode='edge')
medyan_temizlenmis = np.zeros(N)

for i in range(N):
    pencere = padded_vector[i : i + k]
    medyan_temizlenmis[i] = np.median(pencere)

# --- 3. ADIM: GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 6))

plt.plot(t, gurultulu_vektor, label='Gürültülü Sinyal', color='gray', alpha=0.5, linestyle=':')
plt.plot(t, temiz_vektor, label='Orijinal Temiz Sinyal', color='blue', linewidth=1.5, alpha=0.7)
plt.plot(t, gauss_temizlenmis, label='Gauss Filtresi', color='green', linewidth=2)
plt.plot(t, medyan_temizlenmis, label='Medyan Filtresi', color='red', linewidth=2)

plt.title('Gürültülü Sinyalin Gauss ve Medyan Filtreleri ile Temizlenmesi (NumPy)')
plt.xlabel('Zaman (t)')
plt.ylabel('Genlik')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()