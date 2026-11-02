import numpy as np
import matplotlib.pyplot as plt

# --- 1. YARDIMCI FONKSİYONLAR ---
def add_gaussian_noise(image, mean=0, std=50):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 255)

def add_salt_and_pepper_noise(image, ratio=0.1):
    noisy = np.copy(image)
    rand = np.random.rand(*noisy.shape)
    noisy[rand < (ratio / 2)] = 255
    noisy[rand > (1 - ratio / 2)] = 0
    return noisy

# --- 2. DFT VE IDFT FONKSİYONLARI (NumPy FFT ile hızlandırılmış) ---
def apply_low_pass_filter(image, radius=20):
    """Görüntüyü DFT ile frekans uzayına alır, alçak geçiren filtre uygular ve IDFT ile geri döndürür."""
    # 1. 2D DFT İşlemi ve Merkeze Kaydırma
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    
    # 2. Alçak Geçiren Filtre (Low-Pass Filter) Maskesi Oluşturma
    M, N = image.shape
    center_M, center_N = M // 2, N // 2
    y, x = np.ogrid[:M, :N]
    mask = (x - center_N)**2 + (y - center_M)**2 <= radius**2
    
    # 3. Maskeyi Uygulama (Yüksek frekanslı gürültüleri sıfırla)
    dft_filtered = np.zeros_like(dft_shifted)
    dft_filtered[mask] = dft_shifted[mask]
    
    # 4. Ters Kaydırma ve IDFT (Ters Fourier) İşlemi
    idft_shifted = np.fft.ifftshift(dft_filtered)
    reconstructed_image = np.fft.ifft2(idft_shifted)
    
    return np.abs(reconstructed_image) # Reel kısmı (genliği) al

# --- 3. ZNCC HESAPLAMA FONKSİYONU ---
def calculate_zncc(img1, img2):
    """Zero Mean Normalized Cross Correlation metriğini hesaplar [-1, 1]"""
    img1_zm = img1 - np.mean(img1)
    img2_zm = img2 - np.mean(img2)
    
    pay = np.sum(img1_zm * img2_zm)
    payda = np.sqrt(np.sum(img1_zm**2) * np.sum(img2_zm**2))
    
    return pay / (payda + 1e-10)

# --- 4. UYGULAMA VE TEST ---
# 128x128 Sentetik Orijinal Görüntü
boyut = 128
orijinal_resim = np.zeros((boyut, boyut))
orijinal_resim[48:80, 48:80] = 255

# Gürültü Ekleme
gauss_resim = add_gaussian_noise(orijinal_resim, std=60)
sp_resim = add_salt_and_pepper_noise(orijinal_resim, ratio=0.15)

# Filtreleme ve Ters DFT (IDFT) ile Geri Üretim
radius_val = 25 # Filtre yarıçapı
geri_gauss = apply_low_pass_filter(gauss_resim, radius=radius_val)
geri_sp = apply_low_pass_filter(sp_resim, radius=radius_val)

# ZNCC Skorlarının Paketlenmesi
zncc_gauss_ilk = calculate_zncc(orijinal_resim, gauss_resim)
zncc_gauss_son = calculate_zncc(orijinal_resim, geri_gauss)
zncc_sp_ilk = calculate_zncc(orijinal_resim, sp_resim)
zncc_sp_son = calculate_zncc(orijinal_resim, geri_sp)

print(f"Gauss Gürültüsü - Filtre Öncesi ZNCC: {zncc_gauss_ilk:.4f}")
print(f"Gauss Gürültüsü - Ters DFT Sonrası ZNCC: {zncc_gauss_son:.4f}")
print("-" * 50)
print(f"S&P Gürültüsü - Filtre Öncesi ZNCC: {zncc_sp_ilk:.4f}")
print(f"S&P Gürültüsü - Ters DFT Sonrası ZNCC: {zncc_sp_son:.4f}")

# --- 5. GÖRSELLEŞTİRME ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(orijinal_resim, cmap='gray'); axes[0, 0].set_title('Orijinal Resim')
axes[0, 1].imshow(gauss_resim, cmap='gray'); axes[0, 1].set_title(f'Gauss Gürültülü (ZNCC: {zncc_gauss_ilk:.2f})')
axes[0, 2].imshow(geri_gauss, cmap='gray'); axes[0, 2].set_title(f'Ters DFT ile Onarılan (ZNCC: {zncc_gauss_son:.2f})')

axes[1, 0].imshow(orijinal_resim, cmap='gray'); axes[1, 0].set_title('Orijinal Resim')
axes[1, 1].imshow(sp_resim, cmap='gray'); axes[1, 1].set_title(f'S&P Gürültülü (ZNCC: {zncc_sp_ilk:.2f})')
axes[1, 2].imshow(geri_sp, cmap='gray'); axes[1, 2].set_title(f'Ters DFT ile Onarılan (ZNCC: {zncc_sp_son:.2f})')

for ax in axes.flatten():
    ax.axis('off')
    
plt.tight_layout()
plt.show()