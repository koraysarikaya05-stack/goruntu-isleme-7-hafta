import numpy as np
import matplotlib.pyplot as plt

# --- 1. Görüntü ve Gürültü Oluşturma ---
boyut = 128
orijinal = np.zeros((boyut, boyut))
orijinal[40:88, 40:88] = 255  # Merkezde sentetik beyaz kare

def add_gaussian_noise(img, std=40):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255)

gurultulu = add_gaussian_noise(orijinal)

# --- 2. DFT ve Filtre Uygulama Fonksiyonu ---
def apply_filter(img, mask):
    # Frekans uzayına geçip merkeze kaydırıyoruz
    dft_shifted = np.fft.fftshift(np.fft.fft2(img))
    # Maskeyi çarpıp filtreleme yapıyoruz
    filtered = dft_shifted * mask
    # Ters kaydırma ve Ters DFT ile resme dönüyoruz
    idft = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(idft)

# --- 3. Maskeleri (Filtreleri) Oluşturma ---
M, N = orijinal.shape
y, x = np.ogrid[:M, :N]
center_M, center_N = M // 2, N // 2
D = np.sqrt((x - center_N)**2 + (y - center_M)**2) # Merkeze olan uzaklıklar

r_low = 20
r_high = 20
r_b1, r_b2 = 10, 30

lpf_mask = (D <= r_low).astype(float)                # Alçak geçiren
hpf_mask = (D > r_high).astype(float)                # Yüksek geçiren
bpf_mask = ((D >= r_b1) & (D <= r_b2)).astype(float) # Bant geçiren
brf_mask = ((D < r_b1) | (D > r_b2)).astype(float)   # Bant söndüren

# --- 4. Filtreleri Uygulama ---
# Temiz resim üzerinde
ori_lpf = apply_filter(orijinal, lpf_mask)
ori_hpf = apply_filter(orijinal, hpf_mask)
ori_bpf = apply_filter(orijinal, bpf_mask)
ori_brf = apply_filter(orijinal, brf_mask)

# Gürültülü resim üzerinde
gur_lpf = apply_filter(gurultulu, lpf_mask)
gur_hpf = apply_filter(gurultulu, hpf_mask)
gur_bpf = apply_filter(gurultulu, bpf_mask)
gur_brf = apply_filter(gurultulu, brf_mask)

# --- 5. Görselleştirme ---
fig, axes = plt.subplots(2, 5, figsize=(16, 6))

axes[0, 0].imshow(orijinal, cmap='gray'); axes[0, 0].set_title('Orijinal')
axes[0, 1].imshow(ori_lpf, cmap='gray');  axes[0, 1].set_title('Low-Pass')
axes[0, 2].imshow(ori_hpf, cmap='gray');  axes[0, 2].set_title('High-Pass')
axes[0, 3].imshow(ori_bpf, cmap='gray');  axes[0, 3].set_title('Band-Pass')
axes[0, 4].imshow(ori_brf, cmap='gray');  axes[0, 4].set_title('Band-Reject')

axes[1, 0].imshow(gurultulu, cmap='gray'); axes[1, 0].set_title('Gürültülü')
axes[1, 1].imshow(gur_lpf, cmap='gray');   axes[1, 1].set_title('Low-Pass')
axes[1, 2].imshow(gur_hpf, cmap='gray');   axes[1, 2].set_title('High-Pass')
axes[1, 3].imshow(gur_bpf, cmap='gray');   axes[1, 3].set_title('Band-Pass')
axes[1, 4].imshow(gur_brf, cmap='gray');   axes[1, 4].set_title('Band-Reject')

for ax in axes.flatten():
    ax.axis('off')
    
plt.tight_layout()
plt.show()