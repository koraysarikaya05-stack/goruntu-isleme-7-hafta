import numpy as np
import matplotlib.pyplot as plt

# 1. ADIM: ÖNCEKİ GÖREVDEN DFT FONKSİYONU
def calculate_dft(vector):
    N = len(vector)
    n = np.arange(N)
    k = n.reshape((N, 1))
    dft_matrix = np.exp(-2j * np.pi * k * n / N)
    dft_coefficients = np.dot(dft_matrix, vector)
    harmonics = np.abs(dft_coefficients[:N // 2]) * (2.0 / N)
    return harmonics

# Parametreler
N = 256
t = np.linspace(0, 1, N, endpoint=False)

# 2. ADIM: SİNYALLERİN VE GÜRÜLTÜLERİN OLUŞTURULMASI 

# 1. Sinyal (Frekans: 50 Hz, Genlik: 12) ve %20 gürültü
s1_temiz = 12.0 * np.sin(2 * np.pi * 50.0 * t)
s1_noise = np.random.uniform(-12 * 0.2, 12 * 0.2, size=N)
s1_gurultulu = s1_temiz + s1_noise

# 2. Sinyal (Frekans: 30 Hz, Genlik: 5) ve %20 gürültü
s2_temiz = 5.0 * np.sin(2 * np.pi * 30.0 * t)
s2_noise = np.random.uniform(-5 * 0.2, 5 * 0.2, size=N)
s2_gurultulu = s2_temiz + s2_noise

# 3. Toplam Sinyal
toplam_sinyal = s1_gurultulu + s2_gurultulu

# 3. ADIM: DFT HESAPLAMALARI 
dft_s1 = calculate_dft(s1_gurultulu)
dft_s2 = calculate_dft(s2_gurultulu)
dft_toplam = calculate_dft(toplam_sinyal)

# X ekseni için frekans vektörü (Nyquist sınırına kadar)
frekans_ekseni = np.linspace(0.0, N / 2, N // 2)

# Sadece ilk 60 Hz'i çizdirelim (detayları net görmek için)
limit = 60 

# 4. ADIM: GÖRSELLEŞTİRME 
plt.figure(figsize=(14, 10))

# 1. Sinyal Çizimleri
plt.subplot(3, 2, 1)
plt.plot(t, s1_gurultulu, color='blue')
plt.title('1. Sinyal (50 Hz, Genlik 12) - Zaman Uzayı')
plt.grid(True, alpha=0.5)

plt.subplot(3, 2, 2)
plt.stem(frekans_ekseni[:limit], dft_s1[:limit], linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('1. Sinyal - DFT Harmonik Vektörü')
plt.grid(True, alpha=0.5)

# 2. Sinyal Çizimleri
plt.subplot(3, 2, 3)
plt.plot(t, s2_gurultulu, color='green')
plt.title('2. Sinyal (30 Hz, Genlik 5) - Zaman Uzayı')
plt.grid(True, alpha=0.5)

plt.subplot(3, 2, 4)
plt.stem(frekans_ekseni[:limit], dft_s2[:limit], linefmt='g-', markerfmt='go', basefmt=' ')
plt.title('2. Sinyal - DFT Harmonik Vektörü')
plt.grid(True, alpha=0.5)

# Toplam Sinyal Çizimleri
plt.subplot(3, 2, 5)
plt.plot(t, toplam_sinyal, color='purple')
plt.title('Toplam Sinyal (S1 + S2) - Zaman Uzayı')
plt.grid(True, alpha=0.5)

plt.subplot(3, 2, 6)
plt.stem(frekans_ekseni[:limit], dft_toplam[:limit], linefmt='purple', markerfmt='mo', basefmt=' ')
plt.title('Toplam Sinyal - DFT Harmonik Vektörü')
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()