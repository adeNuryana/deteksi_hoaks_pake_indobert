# download_indobert.py
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_indobert_model():
    """Download IndoBERT model dan tokenizer"""
    
    # Path untuk menyimpan
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'trained_models')
    model_path = os.path.join(model_dir, 'indobert_model')
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    
    # Buat directory jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("DOWNLOAD INDOBERT MODEL")
    print("="*70)
    
    # Model IndoBERT yang tersedia
    models = {
        '1': 'indobenchmark/indobert-base-p1',
        '2': 'indobenchmark/indobert-base-p2',
        '3': 'indobenchmark/indobert-lite-base-p1',
        '4': 'indobenchmark/indobert-lite-base-p2',
    }
    
    print("\nPilih model IndoBERT:")
    for key, model_name in models.items():
        print(f"{key}. {model_name}")
    
    choice = input("\nPilih model (1-4, default 2): ").strip() or '2'
    
    if choice not in models:
        print("Pilihan tidak valid, menggunakan model default...")
        choice = '2'
    
    model_name = models[choice]
    
    print(f"\n[1] Downloading {model_name}...")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Download model
        print("Downloading model...")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Untuk klasifikasi hoax/valid
        )
        
        # Simpan ke local
        print("Saving model and tokenizer...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Tokenizer saved to: {tokenizer_path}")
        
        # Test model
        print("\n[2] Testing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Test dengan contoh teks
        test_text = "Vaksin COVID-19 mengandung microchip untuk melacak masyarakat."
        
        inputs = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
        
        print(f"Test text: {test_text}")
        print(f"Prediction: {'HOAX' if predictions[0] == 1 else 'VALID'}")
        
        # Buat file info
        info = {
            'model_name': model_name,
            'download_date': torch.__version__,
            'transformers_version': transformers.__version__,
            'num_labels': 2,
            'max_length': 256,
            'device': str(device)
        }
        
        import json
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Model info saved to: {info_path}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None, None

if __name__ == "__main__":
    import transformers
    model, tokenizer = download_indobert_model()