import numpy as np
import matplotlib.pyplot as plt

def add_noise_to_vector(vector, noise_rate):
    """
    Vektörün genlik aralığına ve belirtilen gürültü oranına (0-1) göre
    rastgele gürültü oluşturur ve vektöre ekler.
    """
    # Vektörün maksimum ve minimum değerleri arasındaki genlik farkını hesaplıyoruz
    amplitude_range = np.max(vector) - np.min(vector)
    
    # Gürültü miktarını genlik farkı ve gürültü oranı ile ölçeklendiriyoruz
    noise_level = amplitude_range * noise_rate
    
    # Vektör ile aynı boyutta, sıfır merkezli rastgele gürültü üretiyoruz
    noise = np.random.uniform(-noise_level / 2, noise_level / 2, size=len(vector))
    
    # Gürültülü yeni vektörü oluşturuyoruz
    noisy_vector = vector + noise
    
    return noisy_vector

# --- ÖRNEK KULLANIM VE GÖRSELLEŞTİRME ---

# 1. Bir önceki ödevde oluşturduğunuz sinüs vektörünü hazırlayalım (Örnek: 256 boyutlu)
N = 256
t = np.linspace(0, 1, N, endpoint=False)
temiz_vektor = 5.0 * np.sin(2 * np.pi * 3.0 * t + np.pi / 2)

# 2. Gürültü oranını belirleyelim (0 ile 1 arasında, örn: 0.2 yani %20)
gurultu_orani = 0.25

# 3. Fonksiyonu çağırarak gürültülü vektörü elde edelim
gurultulu_vektor = add_noise_to_vector(temiz_vektor, gurultu_orani)

# 4. Görselleştirme (Orijinal ve Gürültülü Sinyali Üst Üste Çizdirme)
plt.figure(figsize=(10, 5))
plt.plot(t, temiz_vektor, label='Orijinal Temiz Sinyal', color='blue', linewidth=2, alpha=0.7)
plt.plot(t, gurultulu_vektor, label=f'Gürültülü Sinyal (Oran: {gurultu_orani})', color='red', linestyle='--', alpha=0.8)

plt.title(f'Gürültü Eklenmiş Sinüsoidal Dalga (Gürültü Oranı: {gurultu_orani})')
plt.xlabel('Zaman (t)')
plt.ylabel('Genlik')
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()