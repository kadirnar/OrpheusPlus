import os
import json
import pandas as pd
import math
from datasets import Dataset, Audio, Features, Value, DatasetDict, concatenate_datasets
from glob import glob
from tqdm import tqdm
import shutil
import multiprocessing
from functools import partial

# Ana klasör yolu
ana_klasor = "Emilia-DE-extracted"

# Hugging Face dataset adı
huggingface_dataset_id = "kadirnar/Emilia-DE-Complete"

# Tüm alt klasörleri bulma
alt_klasorler = [f for f in os.listdir(ana_klasor) if os.path.isdir(os.path.join(ana_klasor, f))]
print(f"Toplam {len(alt_klasorler)} alt klasör bulundu.")

# Cache klasörü oluşturma
cache_dir = "./.cache"
os.makedirs(cache_dir, exist_ok=True)

# Her bir JSON dosyasını işleme fonksiyonu
def json_dosyasini_isle(json_dosyasi, max_phone_count=600):
    try:
        with open(json_dosyasi, 'r', encoding='utf-8') as f:
            veri = json.load(f)
            
            # Sadece phone_count değeri max_phone_count veya daha az olanları al
            if veri["phone_count"] <= max_phone_count:
                # MP3 dosyasının tam yolunu ekle (daha sonra Audio() özelliğine dönüştürülecek)
                mp3_dosyasi = os.path.splitext(json_dosyasi)[0] + ".mp3"
                if os.path.exists(mp3_dosyasi):  # MP3 dosyasının varlığını kontrol et
                    veri["audio"] = mp3_dosyasi
                    # Klasör bilgisini de ekle (hangi alt klasörden geldiğini bilmek için)
                    veri["source_folder"] = os.path.basename(os.path.dirname(os.path.dirname(json_dosyasi)))
                    return veri
    except Exception as e:
        print(f"Hata: {json_dosyasi} dosyası işlenirken bir sorun oluştu: {e}")
    return None

# Parquet dosyalarını boyuta göre bölen fonksiyon
def split_dataset_by_size(dataset, max_size_gb=5.0, output_dir="."):
    # Örnek boyutunu tahmin etmek için ilk 100 örneği kullan
    sample_size = min(100, len(dataset))
    sample_dataset = dataset.select(range(sample_size))
    sample_path = os.path.join(output_dir, "sample.parquet")
    sample_dataset.save_to_disk(sample_path)
    
    # Örnek boyutunu hesapla (GB cinsinden)
    sample_size_bytes = sum(os.path.getsize(os.path.join(sample_path, f)) for f in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, f)))
    sample_size_gb = sample_size_bytes / (1024 ** 3)
    avg_sample_size_gb = sample_size_gb / sample_size
    
    # Parquet başına örnek sayısını hesapla
    samples_per_parquet = math.floor(max_size_gb / avg_sample_size_gb)
    
    # Temizlik
    shutil.rmtree(sample_path, ignore_errors=True)
    
    # Veri kümesini böl
    num_splits = math.ceil(len(dataset) / samples_per_parquet)
    print(f"Dataset {num_splits} parçaya bölünecek, her parça yaklaşık {samples_per_parquet} örnek içerecek")
    
    splits = []
    for i in range(num_splits):
        start_idx = i * samples_per_parquet
        end_idx = min((i + 1) * samples_per_parquet, len(dataset))
        splits.append(dataset.select(range(start_idx, end_idx)))
    
    return splits

# Tüm klasörlerden toplanan veri setlerini saklayacak liste
tum_datasets = []

# Her bir klasörü işleme
for klasor_adi in tqdm(alt_klasorler, desc="Klasörler işleniyor"):
    klasor_yolu = os.path.join(ana_klasor, klasor_adi)
    print(f"\n{klasor_yolu} klasörü işleniyor...")
    
    # JSON dosyalarını bulma
    json_dosyalari = glob(os.path.join(klasor_yolu, "**", "*.json"), recursive=True)
    print(f"Toplam {len(json_dosyalari)} JSON dosyası bulundu.")
    
    if len(json_dosyalari) == 0:
        print(f"Uyarı: {klasor_yolu} klasöründe JSON dosyası bulunamadı. Bu klasör atlanıyor.")
        continue
    
    # Cache dosya adı oluştur
    cache_file = os.path.join(cache_dir, f"{klasor_adi}_processed_data.pkl")
    
    # Eğer cache varsa, onu kullan
    if os.path.exists(cache_file):
        print(f"Cache bulundu: {cache_file}, yükleniyor...")
        df = pd.read_pickle(cache_file)
        veri_listesi = df.to_dict('records')
        print(f"Cache'den {len(veri_listesi)} kayıt yüklendi.")
    else:
        # Multiprocessing ile JSON dosyalarını işleme
        num_proc = 8  # İşlemci sayısı
        with multiprocessing.Pool(processes=num_proc) as pool:
            veri_listesi = list(tqdm(
                pool.imap(partial(json_dosyasini_isle, max_phone_count=600), json_dosyalari),
                total=len(json_dosyalari),
                desc="JSON dosyaları işleniyor"
            ))
        
        # None değerlerini filtrele
        veri_listesi = [veri for veri in veri_listesi if veri is not None]
        
        # Cache'e kaydet
        df = pd.DataFrame(veri_listesi)
        df.to_pickle(cache_file)
        print(f"Veriler cache'e kaydedildi: {cache_file}")
    
    print(f"Toplam {len(json_dosyalari)} JSON dosyasından {len(veri_listesi)} tanesi kriterlere uyuyor.")
    
    if len(veri_listesi) == 0:
        print(f"Uyarı: {klasor_yolu} klasöründe kriterlere uyan veri bulunamadı. Bu klasör atlanıyor.")
        continue
    
    # Pandas DataFrame'e dönüştürme
    df = pd.DataFrame(veri_listesi)
    
    # DataFrame'den Dataset oluşturma
    dataset = Dataset.from_pandas(df)
    
    # Audio sütununu Audio() özelliğine dönüştürme
    dataset = dataset.cast_column("audio", Audio())
    
    # Veri setini listeye ekle
    tum_datasets.append(dataset)
    
    # Yerel olarak kaydet (opsiyonel)
    parquet_dosya_yolu = os.path.join(klasor_yolu, "dataset_filtered.parquet")
    dataset.save_to_disk(parquet_dosya_yolu)
    print(f"Filtrelenmiş veriler '{parquet_dosya_yolu}' klasörüne kaydedildi.")

# Tüm veri setlerini birleştir
if len(tum_datasets) > 0:
    print("\nTüm veri setleri birleştiriliyor...")
    combined_dataset = concatenate_datasets(tum_datasets)
    print(f"Toplam {len(combined_dataset)} örnek birleştirildi.")
    
    # Birleştirilmiş veri setini parçalara böl (her biri 5GB'dan küçük olacak şekilde)
    print("\nBirleştirilmiş veri seti parçalara bölünüyor...")
    split_datasets = split_dataset_by_size(combined_dataset, max_size_gb=4.9, output_dir=".")
    
    # DatasetDict oluştur
    dataset_dict = DatasetDict()
    for i, split_ds in enumerate(split_datasets):
        split_name = f"split_{i+1}"
        dataset_dict[split_name] = split_ds
    
    # Hugging Face'e tek bir dataset olarak yükle
    print(f"\nVeriler Hugging Face'e yükleniyor: {huggingface_dataset_id}")
    dataset_dict.push_to_hub(huggingface_dataset_id)
    print(f"Tüm veriler başarıyla Hugging Face'e yüklendi: {huggingface_dataset_id}")
else:
    print("\nHiçbir klasörde kriterlere uyan veri bulunamadı.")

print("\nTüm işlemler başarıyla tamamlandı!")
