import utility as root_project

root_project

from src.config import *
from src.model import *

import torch
import re
import string


# ====================== Normalisasi Typo ======================
def normalize_typo(text):
    text = re.sub(f"([{re.escape(string.punctuation)}])", r" \1 ", text)
    words = text.split()
    normalized_words = [typo_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)


# ====================== Smart Lowercase untuk Token ======================
def smart_lowercase(token):
    if token in known_acronyms or (token.isupper() and len(token) <= 5) or token == "I":
        return token
    return token.lower()


# ====================== Tokenisasi dan Pemisahan Kalimat ======================
def tokenize_and_split_sentences(text):
    # Normalisasi titik-titik beruntun
    text = re.sub(r"\.\.\.+", " ", text)

    # Normalisasi typo/gaul
    text = normalize_typo(text)

    # Proteksi entitas penting
    protected_patterns = {
        r"\b(www)\s*\.\s*([a-zA-Z0-9-]+)\s*\.\s*([a-zA-Z]{2,})\b": r"\1<DOT>\2<DOT>\3<DOT>\4",
        r"\b([a-zA-Z0-9._%+-]+)\s*@\s*([a-zA-Z0-9.-]+)\s*\.\s*([a-zA-Z]{2,})\b": r"\1<AT>\2<DOT>\3",
        r"\b([a-zA-Z0-9-]+)\s*\.\s*([a-zA-Z]{2,})\b": r"\1<DOT>\2",
        r"(?<=\d)\s*[.,]\s*(?=\d)": r"<DECIMAL>",  # Mengganti desimal koma/perioda menjadi <DECIMAL>
        r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})": r"\1/\2/\3",  # Menghindari spasi pada tanggal
        r"(\d+)\s*%\s*": r"\1<PC>",  # Menghindari spasi setelah persentase
    }

    for pattern, repl in protected_patterns.items():
        text = re.sub(pattern, repl, text)

    # Menangani tanda seru berulang '!!!' menjadi satu tanda seru
    text = re.sub(r"(!)\1+", r"\1", text)

    # Tambahkan spasi setelah titik jika langsung diikuti huruf atau angka tanpa spasi
    text = re.sub(r"(?<=[.?!])(?=\w)", r" ", text)

    # Tokenisasi
    token_pattern = r"""
        (?:https?://[^\s]+) |
        (?:\b(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:/[^\s]*)? |
        (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) |
        (?:\b\w+<AT>\w+<DOT>\w+\b) |
        (?:@\w+) |
        (?:\#\w+) |
        (?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}) |
        (?:\d{4}[-/]\d{1,2}[-/]\d{1,2}) |
        (?:\d{1,2}\s+\w+\s+\d{2,4}) |
        (?:\w+\s+\d{1,2},\s+\d{4}) |
        (?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?) |
        (?:\d+(?:<DECIMAL>\d+)?%) |
        (?:[$€£¥₩₹]\s?\d+(?:[.,]\d+)*) |
        (?:rp\s?\d+(?:[.,]\d+)*) |
        (?:\b\w+\b) |
        (?:[^\w\s])
    """
    raw_tokens = re.findall(token_pattern, text, flags=re.VERBOSE)
    tokens = [smart_lowercase(tok) for tok in raw_tokens if tok.strip()]

    # Pola pemisahan kalimat yang lebih hati-hati
    split_pattern = re.compile(r'''
        (?<=\w[.!?…])           # Sebelum titik, tanda seru, atau tanya
        (?=\s*[A-Z0-9])           # Diikuti oleh huruf kapital atau angka
        | (?<=\?)                # setelah tanda tanya
        | (?<=\!)                # setelah tanda seru
        | (?<=\.)                # Setelah titik jika diikuti huruf kecil
    ''', re.VERBOSE)

    rough_sentences = split_pattern.split(text.strip())

    # Gabungkan kalimat jika setelah "!" diikuti kata lanjutan informal
    sentences = []
    for i, s in enumerate(rough_sentences):
        s = s.strip()
        if i > 0 and sentences:
            first_word = s.split()[0].lower() if s.split() else ""
            informal_followups = ['lho', 'dong', 'nih', 'deh', 'ya', 'kan', 'kok', 'sih', 'banget']
            if first_word in informal_followups:
                sentences[-1] += " " + s  # gabungkan ke sebelumnya
                continue
        if s:  # Pastikan kalimat bukan kosong
            sentences.append(s)

    # Unprotect simbol
    def unprotect(s):
        return s.replace("<DOT>", ".").replace("<AT>", "@").replace("<DECIMAL>", ".").replace("<PC>", "%")

    tokens = [unprotect(tok) for tok in tokens]
    sentences = [unprotect(s) for s in sentences]

    return tokens, sentences


# ====================== Bangun Vocab dari Teks ======================
def build_vocab_from_text(text):
    tokens, sentences = tokenize_and_split_sentences(text)
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(sorted(set(tokens)), start=2):
        vocab[word] = i
    return vocab, tokens, sentences


# ====================== Unit Test Dinamis untuk Encoder ======================
def test_encoder_user_text_dynamic():
    user_text = input("Masukkan kalimat: ")

    vocab, tokens, sentences = build_vocab_from_text(user_text)
    vocab_size = len(vocab)

    # Tokenisasi per kalimat
    all_token_ids = []
    max_len = 0
    for sentence in sentences:
        toks, _ = tokenize_and_split_sentences(sentence)
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
        all_token_ids.append(ids)
        max_len = max(max_len, len(ids))

    # Padding tensor [num_sentences, max_len]
    padded_tensor = torch.full((len(all_token_ids), max_len), vocab["<pad>"], dtype=torch.long)
    for i, ids in enumerate(all_token_ids):
        padded_tensor[i, :len(ids)] = torch.tensor(ids)

    # Tokenisasi keseluruhan teks
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    # Inisialisasi model Encoder
    model = Encoder(
        vocab_size=vocab_size,
        max_len=max(max_len, len(token_ids)),
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.1
    )

    output_batch = model(padded_tensor)       # Output per kalimat (batched)
    output = model(input_tensor)              # Output keseluruhan teks

    # Print hasil
    print("\n✅ Vocabulary:", vocab)
    print("✅ Tokens:", tokens)
    print("✅ Token IDs:", token_ids)

    print("\n✅ Sentences:", sentences)
    print("✅ Input tensor shape (full text):", input_tensor.shape)
    print("✅ Output shape (full text):", output.shape)
    print("✅ Output tensor (full text):\n", output)

    print("\n📌 Input tensor shape (batch kalimat):", padded_tensor.shape)
    print("📌 Output shape (batch kalimat):", output_batch.shape)
    print("📌 Output tensor (batch kalimat):\n", output_batch)


if __name__ == "__main__":
    test_encoder_user_text_dynamic()
