import pandas as pd
import numpy as np

# Jumlah data dummy yang ingin digenerate
jumlah = 200

# Buat DataFrame kosong
dummy_data = pd.DataFrame()

# Tambah kolom nama file gambar sepeda + Generate tiap fitur based on rentang/range
dummy_data['gambar_sepeda'] = [f"{i}.jpg" for i in range(1, jumlah + 1)]
dummy_data['panjang_sepeda'] = np.random.uniform(160, 185, jumlah)
dummy_data['poros_pedal_ke_poros_ban_depan'] = np.random.uniform(54, 65, jumlah)
dummy_data['poros_pedal_ke_dasar_ban'] = np.random.uniform(24, 30, jumlah)
dummy_data['poros_pedal_ke_ujung_sadel'] = np.random.uniform(2, 12, jumlah)

# Bulatkan ke 2 desimal
dummy_data = dummy_data.round(2)

# Tampilkan hasil & simpan data dummy
print("Data Dummy Sepeda (cm):")
print(dummy_data.head(10)) 
# dummy_data.to_csv('bike_dummy_data.csv', index=False)