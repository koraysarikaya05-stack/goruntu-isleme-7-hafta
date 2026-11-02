import numpy as np
import matplotlib.pyplot as plt

# 1. ADIM: DFT VE IDFT FONKSİYONLARININ YAZILMASI

def get_dft_coeffs(vector):
    """Sinyalin karmaşık DFT katsayılarını hesaplar."""
    N = len(vector)
    n = np.arange(N)
    k = n.reshape((N, 1))
    dft_matrix = np.exp(-2j * np.pi * k * n / N)
    return np.dot(dft_matrix, vector)

def calculate_idft(dft_coefficients):
    """Karmaşık DFT katsayılarından zaman uzayındaki sinyali (IDFT) geri üretir."""
    N = len(dft_coefficients)
    n = np.arange(N)
    k = n.reshape((N, 1))
    idft_matrix = np.exp(2j * np.pi * k * n / N) / N
    # Sadece reel kısmı alıyoruz (sanal kısımlar sıfıra çok yakın olacaktır)
    return np.dot(idft_matrix, dft_coefficients).real

def frequency_filter(vector, threshold):
    """
    Sinyali frekans uzayına dönüştürür, eşik değerin altındaki 
    harmonikleri sıfırlar ve temizlenmiş sinyali geri döndürür.
    """
    N = len(vector)
    
    # 1. DFT Katsayılarını al
    coeffs = get_dft_coeffs(vector)
    
    # 2. Genlikleri (Harmonikleri) hesapla
    harmonics = np.abs(coeffs) * (2.0 / N)
    
    # 3. Filtreleme: Genliği threshold'dan küçük olan katsayıları 0 yap
    # np.where(koşul, doğruysa yapılacak, yanlışsa yapılacak)
    filtered_coeffs = np.where(harmonics >= threshold, coeffs, 0)
    
    # 4. Ters Dönüşüm (IDFT) ile sinyali yeniden üret
    filtered_signal = calculate_idft(filtered_coeffs)
    
    return filtered_signal

# 2. ADIM: 5. GÖREVDEKİ SİNYALLERİN OLUŞTURULMASI
N = 256
t = np.linspace(0, 1, N, endpoint=False)

# 1. Sinyal (50 Hz, Genlik 12) + Gürültü
s1_temiz = 12.0 * np.sin(2 * np.pi * 50.0 * t)
s1_noise = np.random.uniform(-12 * 0.2, 12 * 0.2, size=N)
s1_gurultulu = s1_temiz + s1_noise

# 2. Sinyal (30 Hz, Genlik 5) + Gürültü
s2_temiz = 5.0 * np.sin(2 * np.pi * 30.0 * t)
s2_noise = np.random.uniform(-5 * 0.2, 5 * 0.2, size=N)
s2_gurultulu = s2_temiz + s2_noise

# Toplam Sinyal
toplam_sinyal = s1_gurultulu + s2_gurultulu

# 3. ADIM: FREKANS FİLTRESİNİN UYGULANMASI 
# Eşik değeri olarak 3.0 belirliyoruz (Çünkü ana genliklerimiz 5 ve 12, gürültü genlikleri 3'ten küçüktür)
esik_degeri = 3.0

s1_filtrelenmis = frequency_filter(s1_gurultulu, esik_degeri)
s2_filtrelenmis = frequency_filter(s2_gurultulu, esik_degeri)
toplam_filtrelenmis = frequency_filter(toplam_sinyal, esik_degeri)

# 4. ADIM: GÖRSELLEŞTİRME
plt.figure(figsize=(14, 10))

# 1. Sinyal Karşılaştırması
plt.subplot(3, 1, 1)
plt.plot(t, s1_gurultulu, label='Gürültülü 1. Sinyal (50 Hz)', color='gray', alpha=0.6, linestyle=':')
plt.plot(t, s1_filtrelenmis, label='Frekans Filtresi ile Temizlenmiş', color='blue')
plt.title('1. Sinyal: Frekans Filtresi Öncesi ve Sonrası')
plt.legend()
plt.grid(True, alpha=0.5)

# 2. Sinyal Karşılaştırması
plt.subplot(3, 1, 2)
plt.plot(t, s2_gurultulu, label='Gürültülü 2. Sinyal (30 Hz)', color='gray', alpha=0.6, linestyle=':')
plt.plot(t, s2_filtrelenmis, label='Frekans Filtresi ile Temizlenmiş', color='green')
plt.title('2. Sinyal: Frekans Filtresi Öncesi ve Sonrası')
plt.legend()
plt.grid(True, alpha=0.5)

# Toplam Sinyal Karşılaştırması
plt.subplot(3, 1, 3)
plt.plot(t, toplam_sinyal, label='Gürültülü Toplam Sinyal (S1 + S2)', color='gray', alpha=0.6, linestyle=':')
plt.plot(t, toplam_filtrelenmis, label='Frekans Filtresi ile Temizlenmiş Toplam', color='purple')
plt.title('Toplam Sinyal: Frekans Filtresi Öncesi ve Sonrası')
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()