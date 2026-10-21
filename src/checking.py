import os
import spacy
import torch

from spacy.cli import download

def torch_check():
    print("Versi PyTorch:", torch.__version__)

    cuda_availibility = torch.cuda.is_available()
    print("Apakah CUDA tersedia?", cuda_availibility)

    if torch.cuda.is_available():
        print("Jumlah GPU terdeteksi:", torch.cuda.device_count())
        print("Nama GPU:", torch.cuda.get_device_name(0))

    else:
        print("GPU CUDA tidak tersedia. Menggunakan CPU.", '\n')


def lib_spacy():
    """
    Fungsi ini akan mengunduh dan menyimpan model spaCy 'xx_sent_ud_sm' ke dalam folder lokal.
    Jika folder model sudah ada, model tidak akan diunduh ulang.
    """

    # Path tujuan penyimpanan model
    model_name = 'xx_sent_ud_sm'
    model_path = f'src/spacy/{model_name}'

    # Cek apakah model sudah ada di folder
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"Model spaCy sudah ada di '{model_path}', tidak perlu mengunduh ulang.")

    else:
        # Jika folder belum ada atau kosong, buat folder dan unduh model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print(f"Folder '{model_path}' berhasil dibuat.")
        
        # Mengunduh model menggunakan spacy.cli.download()
        print(f"Mengunduh model spaCy '{model_name}'...")
        download(model_name)  # Mengunduh model secara otomatis

        # Load model yang sudah diunduh dan simpan ke folder tujuan
        nlp = spacy.load(model_name)
        nlp.to_disk(model_path)

        print(f"Model berhasil disimpan ke '{model_path}'.")

    # Menggunakan model yang sudah disimpan
    nlp = spacy.load(model_path)
    return nlp




if __name__ == "__main__":
    torch_check()
    lib_spacy()