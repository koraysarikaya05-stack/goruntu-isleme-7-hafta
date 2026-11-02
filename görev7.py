import numpy as np
import matplotlib.pyplot as plt

def calculate_2d_dft(image):
    """
    2-boyutlu bir resim matrisi üzerinde DFT hesaplar
    ve logaritmik olarak ölçeklenmiş harmonik matrisini döndürür.
    """
    # Resmin boyutlarını alıyoruz (M: Satır, N: Sütun)
    M, N = image.shape
    
    # 1. Satırlar için DFT Matrisinin (A_M) oluşturulması
    m = np.arange(M)
    k_m = m.reshape((M, 1))
    dft_matrix_M = np.exp(-2j * np.pi * k_m * m / M)
    
    # 2. Sütunlar için DFT Matrisinin (A_N) oluşturulması
    n = np.arange(N)
    k_n = n.reshape((N, 1))
    dft_matrix_N = np.exp(-2j * np.pi * k_n * n / N)
    
    # 3. Ayrılabilirlik (Separability) özelliği ile 2D DFT hesabı (F = A_M * f * A_N)
    # Önce satırlara, sonra sütunlara dönüşüm uygulanır
    dft_2d = np.dot(dft_matrix_M, image)
    dft_2d = np.dot(dft_2d, dft_matrix_N)
    
    # 4. Düşük frekansları (DC bileşeni) matrisin köşelerinden tam merkeze kaydırma
    dft_shifted = np.fft.fftshift(dft_2d)
    
    # 5. Harmonik matrisi (Genlik Spektrumu)
    harmonics = np.abs(dft_shifted)
    
    # İnsan gözünün görebilmesi için logaritmik dönüşüm uyguluyoruz
    # (Çünkü merkezdeki sıfır frekansı diğer tüm detayları ezecek kadar büyüktür)
    harmonics_log = 20 * np.log(harmonics + 1)
    
    return harmonics_log

# ÖRNEK KULLANIM VE GÖRSELLEŞTİRME

# Boyutu 2'nin kuvveti olan (128x128) örnek bir sentetik resim oluşturalım
boyut = 128
test_resmi = np.zeros((boyut, boyut))

# Resmin ortasına beyaz bir kare çizelim (Bu, frekans uzayında bir "Sinc" fonksiyonu oluşturur)
test_resmi[48:80, 48:80] = 255

# Fonksiyonu çağırarak harmonik matrisini elde edelim
harmonik_spektrum = calculate_2d_dft(test_resmi)

# Sonuçları Çizdirme
plt.figure(figsize=(12, 5))

# 1. Orijinal Resim
plt.subplot(1, 2, 1)
plt.imshow(test_resmi, cmap='gray')
plt.title('Orijinal Resim (Zaman/Uzay Alanı)')
plt.axis('off')

# 2. DFT Genlik Spektrumu
plt.subplot(1, 2, 2)
plt.imshow(harmonik_spektrum, cmap='gray')
plt.title('2D DFT Harmonik Matrisi (Frekans Uzayı)')
plt.axis('off')

plt.tight_layout()
plt.show()