import numpy as np
import matplotlib.pyplot as plt

#1. ADIM: DFT FONKSİYONUNUN YAZILMASI
def calculate_dft(vector):
    """
    2'nin kuvveti boyuttaki bir vektör üzerinde DFT hesaplar
    ve harmonik (genlik spektrumu) vektörünü döndürür.
    """
    N = len(vector)
    
    # 2'nin kuvveti kontrolü
    if (N & (N - 1)) != 0 or N == 0:
        print("Uyarı: Vektör boyutu 2'nin kuvveti değil!")
        
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # DFT Matrisi formülü: e^(-j * 2 * pi * k * n / N)
    dft_matrix = np.exp(-2j * np.pi * k * n / N)
    
    # Matris çarpımı ile DFT katsayılarının hesabı
    dft_coefficients = np.dot(dft_matrix, vector)
    
    # Harmonik vektörü (Genlik Spektrumu)
    # Ölçeklendirme (1/N) ve Nyquist sınırına kadar ilk yarısını alma
    harmonics = np.abs(dft_coefficients[:N // 2]) * (2.0 / N)
    
    return harmonics

# 2. ADIM: SİNYAL VE GÜRÜLTÜ OLUŞTURMA (ÖNCELLİKİ ÖDEVLERDEN)
N = 256 # 2^8
t = np.linspace(0, 1, N, endpoint=False)
frekans = 5.0 # Hz cinsinden ana frekansımız

# Gürültüsüz Sinyal (5 Hz Sinüs)
temiz_vektor = 5.0 * np.sin(2 * np.pi * frekans * t + np.pi / 2)

# Gürültülü Sinyal (%30 gürültü eklenmiş)
amp_range = np.max(temiz_vektor) - np.min(temiz_vektor)
noise = np.random.uniform(-amp_range * 0.3 / 2, amp_range * 0.3 / 2, size=N)
gurultulu_vektor = temiz_vektor + noise

#3. ADIM: DFT HESAPLAMALARI
temiz_harmonikler = calculate_dft(temiz_vektor)
gurultulu_harmonikler = calculate_dft(gurultulu_vektor)

# Frekans Ekseni (0 Hz'den Nyquist frekansına kadar)
frekans_ekseni = np.linspace(0.0, N / 2, N // 2)

# 4. ADIM: GÖRSELLEŞTİRME 
plt.figure(figsize=(12, 8))

# Zaman Uzayı Sinyalleri
plt.subplot(2, 1, 1)
plt.plot(t, temiz_vektor, label='Gürültüsüz Sinyal', color='blue', alpha=0.7)
plt.plot(t, gurultulu_vektor, label='Gürültülü Sinyal', color='red', linestyle='--', alpha=0.6)
plt.title('Zaman Uzayı (Time Domain)')
plt.xlabel('Zaman (t)')
plt.ylabel('Genlik')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Frekans Uzayı (Harmonikler / Genlik Spektrumu)
plt.subplot(2, 1, 2)
plt.stem(frekans_ekseni[:30], temiz_harmonikler[:30], linefmt='b-', markerfmt='bo', basefmt=' ', label='Gürültüsüz Harmonikler')
plt.stem(frekans_ekseni[:30], gurultulu_harmonikler[:30], linefmt='r--', markerfmt='rx', basefmt=' ', label='Gürültülü Harmonikler')
plt.title('Frekans Uzayı / DFT Harmonik Spektrumu (Frequency Domain)')
plt.xlabel('Frekans (Hz)')
plt.ylabel('Harmonik Genliği')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()