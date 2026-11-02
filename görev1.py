import numpy as np
import matplotlib.pyplot as plt

def fill_sine_vector(vector, amplitude, frequency, phase):
    """
    Parametre olarak alınan vektörü belirtilen genlik, frekans ve faz
    bilgilerine göre sinüsoidal değerlerle doldurur.
    """
    N = len(vector)
    
    if (N & (N - 1)) != 0 or N == 0:
        print("Uyarı: Vektör boyutu 2'nin kuvveti değil!")

    # Zaman veya x eksenini oluşturuyoruz (0 ile 1 saniye aralığında N adet örnek)
    # Sinüs dalgasının periyotlarını düzgün görmek için zaman vektörü yaratıyoruz.
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Vektörün içini sinüs değerleri ile dolduruyoruz (in-place / yerinde atama)
    # Formül: A * sin(2 * pi * f * t + faz)
    vector[:] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    return t, vector

# 1. Aşama: 2'nin kuvveti olan bir vektör oluştur (Örnek: 2^8 = 256)
vector_size = 256 
my_vector = np.zeros(vector_size)

# 2. Aşama: Sinüs parametrelerini belirle
genlik = 5.0                 # Dalganın yüksekliği (Amplitude)
frekans = 3.0                # 1 birim zamanda oluşacak tam dalga sayısı (Frequency)
faz = np.pi / 2              # Başlangıç açısı (Phase - burada 90 derece yani kosinüs gibi başlar)

# 3. Aşama: Fonksiyonu çağırarak vektörü doldur
zaman, dolu_vektor = fill_sine_vector(my_vector, genlik, frekans, faz)

# 4. Aşama: Görselleştirme
plt.figure(figsize=(10, 4))
plt.plot(zaman, dolu_vektor, color='blue', linewidth=2, label='Sinüs Sinyali')

# Veri noktalarını (vektör elemanlarını) daha net görmek için noktalar ekleyelim
plt.scatter(zaman, dolu_vektor, color='red', s=10, zorder=3, label='Vektör Verileri (Örnekler)')

plt.title(f'Sinüsoidal Dalga\n(Boyut: {vector_size}, Genlik: {genlik}, Frekans: {frekans}Hz, Faz: {faz:.2f} rad)')
plt.xlabel('Zaman (t)')
plt.ylabel('Genlik Değeri')
plt.axhline(0, color='black', linewidth=1) # Sıfır çizgisi
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()