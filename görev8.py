import numpy as np
import matplotlib.pyplot as plt

# 1. GÜRÜLTÜ FONKSİYONLARI

def add_gaussian_noise(image, mean=0, std=50):
    """
    Resme belirtilen ortalama (mean) ve standart sapma (std/kuvvet) 
    değerlerine göre Gauss gürültüsü ekler.
    """
    # Resimle aynı boyutta Gauss dağılımlı rastgele gürültü üret
    gauss_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gauss_noise
    
    # Piksel değerlerini 0-255 aralığına sıkıştır (taşmaları önlemek için)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

def add_salt_and_pepper_noise(image, noise_ratio=0.1):
    """
    Resme belirtilen oranda [0, 1] Tuz (beyaz) ve Karabiber (siyah) gürültüsü ekler.
    """
    noisy_image = np.copy(image)
    
    # 0 ile 1 arasında resimle aynı boyutta rastgele bir matris oluştur
    rand_mat = np.random.rand(*noisy_image.shape)
    
    # Oranın yarısı kadar pikseli Tuz (255 - Beyaz) yap
    noisy_image[rand_mat < (noise_ratio / 2)] = 255
    
    # Oranın diğer yarısı kadar pikseli Karabiber (0 - Siyah) yap
    noisy_image[rand_mat > (1 - noise_ratio / 2)] = 0
    
    return noisy_image

# 2. DFT FONKSİYONU

def calculate_2d_dft(image):
    M, N = image.shape
    m = np.arange(M)
    k_m = m.reshape((M, 1))
    dft_M = np.exp(-2j * np.pi * k_m * m / M)
    
    n = np.arange(N)
    k_n = n.reshape((N, 1))
    dft_N = np.exp(-2j * np.pi * k_n * n / N)
    
    dft_2d = np.dot(dft_M, image)
    dft_2d = np.dot(dft_2d, dft_N)
    
    dft_shifted = np.fft.fftshift(dft_2d)
    harmonics_log = 20 * np.log(np.abs(dft_shifted) + 1)
    
    return harmonics_log

# 3. UYGULAMA VE GÖRSELLEŞTİRME

# 128x128 boyutunda ortası beyaz kare olan sentetik test resmi
boyut = 128
orijinal_resim = np.zeros((boyut, boyut))
orijinal_resim[48:80, 48:80] = 255

# Gürültülü resimleri oluşturma
gauss_resim = add_gaussian_noise(orijinal_resim, mean=0, std=60)
sp_resim = add_salt_and_pepper_noise(orijinal_resim, noise_ratio=0.15)

# DFT Harmonik matrislerini hesaplama
dft_orijinal = calculate_2d_dft(orijinal_resim)
dft_gauss = calculate_2d_dft(gauss_resim)
dft_sp = calculate_2d_dft(sp_resim)

# Ekrana Çizdirme
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Sütun: Orijinal
axes[0, 0].imshow(orijinal_resim, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Orijinal Resim')
axes[0, 0].axis('off')

axes[1, 0].imshow(dft_orijinal, cmap='gray')
axes[1, 0].set_title('Orijinal Harmonik Matris')
axes[1, 0].axis('off')

# 2. Sütun: Gauss Gürültüsü
axes[0, 1].imshow(gauss_resim, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Gauss Gürültüsü (std=60)')
axes[0, 1].axis('off')

axes[1, 1].imshow(dft_gauss, cmap='gray')
axes[1, 1].set_title('Gauss Harmonik Matris')
axes[1, 1].axis('off')

# 3. Sütun: Salt & Pepper Gürültüsü
axes[0, 2].imshow(sp_resim, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Salt & Pepper Gürültüsü (%15)')
axes[0, 2].axis('off')

axes[1, 2].imshow(dft_sp, cmap='gray')
axes[1, 2].set_title('S&P Harmonik Matris')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()