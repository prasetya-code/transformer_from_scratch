# Tanda Baca

Berikut adalah daftar tanda baca yang umum digunakan dalam bahasa Indonesia beserta fungsinya secara singkat:

## 1. Titik (.)
- Mengakhiri kalimat pernyataan.  
  Contoh: *Saya pergi ke pasar.*

## 2. Koma (,)
- Memisahkan unsur dalam daftar.
- Memisahkan anak kalimat dari induk kalimat.  
  Contoh: *Saya membeli apel, jeruk, dan pisang.*

## 3. Tanda Tanya (?)
- Mengakhiri kalimat tanya.  
  Contoh: *Kamu sudah makan?*

## 4. Tanda Seru (!)
- Mengakhiri kalimat perintah atau seruan.  
  Contoh: *Cepat pergi!*

## 5. Titik Dua (:)
- Mengawali daftar, penjelasan, atau kutipan langsung.  
  Contoh: *Dia membawa: buku, pensil, dan penggaris.*

## 6. Titik Koma (;)
- Memisahkan kalimat setara yang panjang atau sudah mengandung koma.  
  Contoh: *Dia pergi ke Bandung; saya, ke Yogyakarta.*

## 7. Tanda Hubung (-)
- Menggabungkan dua kata menjadi satu istilah.
- Memenggal kata saat pergantian baris.  
  Contoh: *bertepuk-tangan, kilo-meter.*

## 8. Tanda Pisah (– atau —)
- Memberi penjelasan atau keterangan tambahan di antara kalimat.  
  Contoh: *Rina—teman lamaku—datang berkunjung.*

## 9. Tanda Kurung (())
- Menyisipkan keterangan tambahan.  
  Contoh: *Ia lahir di Bandung (tahun 1990).*

## 10. Tanda Kurung Siku ([])
- Menambahkan informasi dalam kutipan atau teks asli.  
  Contoh: *Ia berkata, “Aku akan datang [besok pagi].”*

## 11. Tanda Petik (" ")
- Mengapit kutipan langsung atau judul.  
  Contoh: *Ia berkata, "Saya siap."*

## 12. Tanda Petik Tunggal (' ')
- Menandai makna khusus atau istilah asing.  
  Contoh: *Kata 'data' berasal dari bahasa Latin.*

## 13. Tanda Elipsis (...)
- Menandakan bagian yang dihilangkan atau kalimat yang menggantung.  
  Contoh: *Kalau saja aku tahu...*

## 14. Garis Miring (/)
- Menunjukkan pilihan atau alternatif.  
  Contoh: *dan/atau, pria/wanita.*




# Sample 1
```bash
# def test_encoder_user_text_dynamic():
#     user_text = input("Masukkan kalimat: ")

#     vocab, tokens, sentences = build_vocab_from_text(user_text)
#     vocab_size = len(vocab)

#     # Tokenisasi per kalimat
#     all_token_ids = []
#     max_len = 0

#     for sentence in sentences:
#         toks, _ = tokenize_and_split_sentences(sentence)
#         ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
#         all_token_ids.append(ids)
#         max_len = max(max_len, len(ids))

#     # Padding manual agar ukuran seragam
#     padded_tensor = torch.full((len(all_token_ids), max_len), vocab["<pad>"], dtype=torch.long)
#     for i, ids in enumerate(all_token_ids):
#         padded_tensor[i, :len(ids)] = torch.tensor(ids)

#     token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
#     input_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, seq_len]

#     # Inisialisasi encoder di CPU
#     model = Encoder(
#         vocab_size=vocab_size,
#         max_len=50,
#         embedding_dim=32,
#         num_layers=2,
#         num_heads=4,
#         ffn_dim=64,
#         dropout=0.1
#     )

#     output = model(input_tensor)
#     print("\n✅ Vocabulary:", vocab)
#     print("✅ Tokens:", tokens)
#     print("✅ Token IDs:", token_ids)

#     print("\n✅ Sentences:", sentences)
#     print("✅ Output shape:", output.shape)
#     print("✅ Output tensor:\n", output)
```








# Sample 2

```bash
def smart_lowercase(token):
    if token in known_acronyms or (token.isupper() and len(token) <= 5) or token == "I":
        return token
    return token.lower()
```


```bash
token_pattern = r"""
        (?:https?://[^\s]+) |                                          # URL lengkap
        (?:\b(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:/[^\s]*)? |     # Domain dan subdomain
        (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) |           # Email
        (?:@\w+) |                                                     # Mention
        (?:\#\w+) |                                                    # Hashtag
        (?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}) |                            # Tanggal umum
        (?:\d{4}[-/]\d{1,2}[-/]\d{1,2}) |                              # Tanggal format ISO
        (?:\d{1,2}\s+\w+\s+\d{2,4}) |                                  # Tanggal panjang
        (?:\w+\s+\d{1,2},\s+\d{4}) |                                   # Tanggal gaya Inggris
        (?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?) |                        # Angka besar
        (?:\d+(?:<DECIMAL>\d+)?%) |                                    # Persentase dengan titik/koma
        (?:[$€£¥₩₹]\s?\d+(?:[.,]\d+)*) |                               # Mata uang
        (?:rp\s?\d+(?:[.,]\d+)*) |                                     # Rupiah
        (?:\b\w+\b) |                                                  # Kata biasa
        (?:[^\w\s])                                                    # Simbol lain
    """
```