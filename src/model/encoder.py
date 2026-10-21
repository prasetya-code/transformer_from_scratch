import torch
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    """
    =====================
     initialize sections
    =====================
    ✅ nn.Module adalah class dasar semua layer PyTorch

    ✅ Ini self-attention
        * embed_dim     = dimensi vektor input
        * num_heads     = jumlah kepala (multi-head attention)
        * batch_first   = True → input model dalam bentuk (batch, seq_len, dim) != (seq_len, batch, dim)

    ✅ Pada FFN terdapat dua layer linear (dense) dengan ReLU di tengah.
        Gunanya untuk meningkatkan kapasitas representasi token.

    ✅ Setiap step (attention dan ffn) akan diberi residual connection → hasilnya distabilkan oleh LayerNorm. 
        Dropout mencegah overfitting.

    =====================
       forward sections
    =====================
    🔄 Flow: 
        1. Input x masuk ke self-attention → hasil: `attn_output`
        2. Hasilnya ditambahkan ke x (residual) lalu dinormalkan → `x = norm(x + attn)`
        3. Dimasukkan ke FFN → ditambahkan lagi ke inputnya → dinormalkan lagi
    """
    

    def __init__(
            self, 
            embedding_dim, 
            num_heads, 
            ffn_dim, 
            dropout
    ):
        super().__init__()

        # self atention
        self.self_attn = nn.MultiheadAttention(
            embed_dim = embedding_dim, 
            num_heads = num_heads, 
            dropout = dropout, 
            batch_first = True
        )

        # Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(
                embedding_dim, 
                ffn_dim
            ),
            nn.ReLU(),
            nn.Linear(
                ffn_dim, 
                embedding_dim
            )
        )

        # LayerNorm 
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Input tensor 'x' harus memiliki bentuk/dimensi: [batch_size, seq_len, embedding_dim]
        try:
            # 1. Menghitung korelasi antar token menggunakan Multi-head Self-attention.
            # Menggunakan x sebagai Query, Key, dan Value karena ini adalah mekanisme *self*-attention.
            attn_output, _ = self.self_attn(x, x, x, key_padding_mask = mask)
            
            # 2. Menerapkan Dropout pada output attention untuk mencegah ketergantungan berlebih antar neuron (overfitting).
            # Ditambahkan dengan input 'x' asal (Residual Connection) untuk mencegah hilangnya gradien (vanishing gradient).
            # Dinormalisasi dengan LayerNorm agar skala nilai aktivasi tetap stabil di setiap batch.
            x = self.norm1(x + self.dropout1(attn_output))  # Residual + Norm

            # 3. Mentransformasikan representasi fitur token secara independen lewat Feed-Forward Network (FFN).
            ffn_output = self.ffn(x)
            
            # 4. Menerapkan Dropout kedua pada output FFN.
            # Ditambahkan kembali ke input FFN (skema residual kedua) lalu dinormalisasi ulang menggunakan LayerNorm kedua.
            x = self.norm2(x + self.dropout2(ffn_output))  # Residual + Norm

            return x
            
        except Exception as error:
            # Menangkap kegagalan runtime (misal: dimensi 'x' tidak sesuai dengan embedding_dim saat inisialisasi)
            raise RuntimeError(
                f"[Gagal di EncoderLayer] Terjadi error pada kalkulasi internal layer.\n"
                f"Detail Error: {error}\n"
                f"Pastikan ukuran fitur terakhir dari tensor 'x' cocok dengan dimensi LayerNorm: {self.norm1.normalized_shape}"
            )



class Encoder(nn.Module):
    """
    =====================
     initialize sections
    =====================
    🧠 Komponen 1 — Token Embedding
        🔤 Fungsi ini mengubah token ID (angka) menjadi vektor berdimensi embedding_dim.

    🧠 Komponen 2 — Positional Encoding
        🧭 Posisi kata dalam kalimat itu penting (misal: "kamu makan" ≠ "makan kamu").
        Kita tambahkan sinyal posisi ke setiap token.

    🧠 Komponen 3 — Layer Stack
        🔁 Ini membuat sekumpulan encoder layer. Misal: 6 layer → ulangi proses attention + FFN sebanyak 6x.

    =====================
       forward sections
    =====================
    🔄 Proses:
        1. Ubah token ID jadi vector → token_embedding
        2. Tambahkan posisi → + positional_encoding
        3. Dropout
        4. Masukkan ke setiap EncoderLayer
        5. Kembalikan output

    🔍 Positional Encoding (dalam _generate_positional_encoding)
        📐 Sinus dan Cosinus memberikan nilai unik berdasarkan posisi token, supaya model bisa tahu: "token ini di posisi ke-5" misalnya.
    """


    def __init__(self, vocab_size, max_len, embedding_dim, num_layers, num_heads, ffn_dim, dropout):
        super().__init__()
        
        # Mengonversi representasi index token diskrit menjadi representasi vektor kontinu dense.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Mendaftarkan positional encoding sebagai 'buffer', bukan parameter teroptimasi.
        # Catatan: register_buffer memastikan tensor ini otomatis ikut pindah device (CPU/GPU) 
        # saat memanggil `model.to(device)` tanpa ikut diperbarui lewat backpropagation.
        self.register_buffer(
            'positional_encoding', 
            self._generate_positional_encoding(max_len, embedding_dim)
        )

        # Menyimpan sekumpulan objek EncoderLayer ke dalam list terstruktur agar terdaftar di arsitektur PyTorch.
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Input tensor 'x' bertipe integer/long dengan bentuk/dimensi: [batch_size, seq_len]
        try:
            # 1. Mendapatkan nilai panjang urutan (sequence length) secara dinamis dari dimensi kedua tensor input.
            seq_len = x.size(1)
            
            # 2. Mengubah token ID menjadi vektor embedding, kemudian ditambahkan dengan matriks posisi (positional encoding).
            # Menggunakan teknik slicing `[:, :seq_len, :]` agar ukuran posisi pas dengan panjang kalimat input saat ini.
            # Jika ada token ID di dalam 'x' yang nilainya >= vocab_size, baris ini akan melempar IndexError.
            x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :]
            
            # 3. Menerapkan dropout awal pada representasi gabungan embedding + posisi untuk regularisasi jaringan.
            x = self.dropout(x)

            # 4. Melakukan perulangan bertahap (iterasi) untuk melewatkan data ke setiap blok EncoderLayer yang ada di dalam tumpukan.
            for layer in self.layers:
                x = layer(x, mask)

            return x  # Output Tensor memiliki bentuk: [batch_size, seq_len, embedding_dim]
            
        except IndexError as index_error:
            # Pencegahan khusus untuk token korup di luar kamus atau teks yang melebihi kapasitas max_len model.
            raise IndexError(
                f"[Gagal di Encoder] Masalah batas indeks tensor.\n"
                f"Detail Error: {index_error}\n"
                f"Solusi: Periksa apakah ada Token ID yang melampaui ukuran vocab atau pastikan panjang teks input (seq_len = {x.size(1) if x.dim() > 1 else 'Tidak diketahui'}) tidak melebihi max_len."
            )
        except Exception as general_error:
            # Menangkap jenis galat tipe data (misal inputnya bertipe FloatTensor padahal diwajibkan LongTensor untuk embedding)
            raise RuntimeError(
                f"[Gagal di Encoder] Proses forward terhenti karena kesalahan data input.\n"
                f"Detail Error: {general_error}\n"
                f"Pastikan tensor input bertipe data integer (torch.long) dan berdimensi 2D [batch_size, seq_len]."
            )

    def _generate_positional_encoding(self, max_len, d_model):
        # 1. Membuat tensor dasar bernilai nol dengan ukuran maksimum alokasi [max_len, d_model].
        pe = torch.zeros(max_len, d_model)
        
        # 2. Membuat deret angka index posisi dari 0 sampai max_len-1 dan menaikkan dimensinya menjadi 2D [max_len, 1].
        position = torch.arange(0, max_len).unsqueeze(1)

        # 3. Menghitung skala pembagi eksponensial berbasis fungsi logaritma natural sesuai formula aslinya.
        # Rumus ini dihitung melompati setiap 2 indeks (step=2) untuk mewakili dimensi genap dan ganjil.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 4. Mengisi kolom indeks genap (0, 2, 4, dst.) menggunakan fungsi gelombang Sinus.
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 5. Mengisi kolom indeks ganjil (1, 3, 5, dst.) menggunakan fungsi gelombang Kosinus.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 6. Menambahkan dimensi batch bohongan di depan sehingga bentuknya menjadi [1, max_len, d_model]
        # Hal ini mempermudah proses penjumlahan otomatis (broadcasting) dengan tensor data di fungsi forward.
        pe = pe.unsqueeze(0)  

        return pe
