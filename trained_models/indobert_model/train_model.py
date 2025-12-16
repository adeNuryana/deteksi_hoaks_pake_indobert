# train_model.py
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
from torch.optim import AdamW

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from django.conf import settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hoax_detector.settings')

import django
django.setup()

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import functional as F

class HoaxDataset(Dataset):
    """Dataset untuk training model hoax detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelTrainer:
    """Class untuk melatih model IndoBERT"""
    
    def __init__(self, model_name='indobenchmark/indobert-base-p2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        
        # Path untuk menyimpan model
        self.model_dir = os.path.join(settings.BASE_DIR, 'trained_models')
        self.model_path = os.path.join(self.model_dir, 'indobert_model')
        self.tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        
        # Buat directory jika belum ada
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.tokenizer_path, exist_ok=True)
    
    def load_sample_data(self):
        """Load atau buat data training contoh"""
        
        # Data hoax contoh
        hoax_data = [
            "Vaksin COVID-19 mengandung microchip untuk melacak pergerakan masyarakat. Informasi ini telah dikonfirmasi oleh ahli teknologi.",
            "Gempa besar 9 SR akan mengguncang Jawa Barat dalam 48 jam ke depan berdasarkan prediksi ahli.",
            "Bank Indonesia akan menarik uang kertas lama tanpa pengganti mulai besok.",
            "Makan bawang putih mentah dapat menyembuhkan COVID-19 dalam 3 hari. Sudah terbukti klinis.",
            "Sinyal 5G dapat menyebabkan virus corona menyebar lebih cepat.",
            "Presiden akan mengundurkan diri minggu depan karena sakit parah. Sumber dari dalam istana.",
            "Air hujan mengandung zat beracun dari pabrik kimia. Jangan kehujanan bulan ini.",
            "Pemerintah akan memblokir semua media sosial selama satu bulan penuh.",
            "Ada penculikan anak secara masif di kota-kota besar. Polisi mengkonfirmasi 50 kasus.",
            "Mata uang rupiah akan dinilai ulang menjadi 1 USD = Rp 1.000.",
            "Vaksin menyebabkan infertilitas pada wanita muda. Sudah dibuktikan penelitian.",
            "Minum air rebusan daun sirih dapat mencegah COVID-19 tanpa vaksin.",
            "Pemerintah menyembunyikan fakta tentang efek samping vaksin yang mematikan.",
            "Tes PCR tidak akurat dan sering memberikan hasil positif palsu.",
            "Lockdown hanya untuk mengontrol masyarakat, bukan untuk kesehatan.",
            "Obat Ivermectin lebih efektif dari vaksin untuk COVID-19.",
            "Pemerintah akan memberlakukan sertifikat vaksin untuk semua aktivitas.",
            "COVID-19 adalah konspirasi perusahaan farmasi untuk menjual vaksin.",
            "Masker justru berbahaya karena membuat sulit bernafas dan menumpuk CO2.",
            "Cuci tangan tidak efektif mencegah penyebaran virus.",
        ]
        
        # Data valid contoh
        valid_data = [
            "Pemerintah Indonesia meluncurkan program vaksinasi COVID-19 tahap ketiga bulan depan.",
            "Bank Indonesia menjaga stabilitas nilai tukar rupiah di tengah gejolak pasar global.",
            "Timnas Indonesia akan bertanding dalam kualifikasi Piala Dunia minggu depan.",
            "Kementerian Pendidikan memberikan beasiswa untuk 10.000 mahasiswa berprestasi.",
            "Harga cabai rawit mengalami kenaikan 20% akibat cuaca ekstrem di sentra produksi.",
            "PT Kereta Api Indonesia menambah jumlah perjalanan kereta selama musim liburan.",
            "Badan Meteorologi memprediksi hujan dengan intensitas sedang akan turun di Jakarta besok.",
            "Ekspor komoditas kelapa sawit Indonesia meningkat 15% pada kuartal terakhir.",
            "Pemerintah kota membangun tiga sekolah baru di daerah pinggiran untuk tahun ajaran depan.",
            "Indeks harga saham gabungan ditutup menguat 0.5% pada perdagangan hari ini.",
            "Pemerintah memberikan bantuan sosial kepada masyarakat terdampak pandemi.",
            "Program Kartu Prakerja dibuka kembali untuk 1 juta peserta gelombang baru.",
            "Inflasi Indonesia terkendali di angka 3.5% sesuai target Bank Indonesia.",
            "Pertumbuhan ekonomi Indonesia triwulan III mencapai 5.1% year on year.",
            "Kementerian Kesehatan menambah kapasitas tempat tidur rumah sakit di daerah.",
            "Program vaksinasi anak usia 6-11 tahun dimulai di seluruh Indonesia.",
            "Pemerintah memperpanjang PPKM level 3 di beberapa daerah hingga dua minggu ke depan.",
            "Edukasi tentang pencegahan COVID-19 terus digencarkan melalui media massa.",
            "Program bantuan UMKM berjalan lancar dengan penyaluran dana triliunan rupiah.",
            "Infrastruktur transportasi terus dibangun untuk mendukung perekonomian.",
        ]
        
        # Gabungkan data
        texts = hoax_data + valid_data
        labels = [1] * len(hoax_data) + [0] * len(valid_data)  # 1=hoax, 0=valid
        
        return texts, labels
    
    def load_dataset_from_csv(self, filepath='hoax_dataset.csv'):
        """Load dataset dari file CSV"""
        try:
            df = pd.read_csv(filepath)
            texts = df['text'].values
            labels = df['label'].values
            logger.info(f"Loaded dataset: {len(texts)} samples")
            return texts, labels
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None, None
    
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1, batch_size=16):
        """Persiapkan data untuk training"""
        
        # Split data: train, validation, test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=(test_size + val_size), random_state=42, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=test_size/(test_size + val_size), 
            random_state=42, stratify=temp_labels
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        logger.info(f"Test samples: {len(test_texts)}")
        
        # Buat datasets
        train_dataset = HoaxDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HoaxDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = HoaxDataset(test_texts, test_labels, self.tokenizer)
        
        # Buat dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size
        )
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def train(self, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5):
        """Training loop"""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Tracking
        training_stats = []
        best_val_accuracy = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_dataloader, desc="Training")
            for batch in progress_bar:
                # Pindahkan batch ke device
                b_input_ids = batch['input_ids'].to(self.device)
                b_attention_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)
                
                # Zero gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Statistics
                total_train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == b_labels).sum().item()
                train_total += b_labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct/train_total
                })
            
            # Hitung rata-rata training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            val_accuracy, avg_val_loss = self.evaluate(val_dataloader)
            
            # Simpan statistik
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            })
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Train Acc:  {train_accuracy:.4f}")
            logger.info(f"Val Loss:   {avg_val_loss:.4f}")
            logger.info(f"Val Acc:    {val_accuracy:.4f}")
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                
                # Save best model
                self.save_model()
                logger.info(f"✓ Model terbaik disimpan (Acc: {val_accuracy:.4f})")
            else:
                patience_counter += 1
                logger.info(f"ⓘ Tidak ada peningkatan ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    logger.info(f"⚠ Early stopping di epoch {epoch + 1}")
                    break
        
        # Plot training history
        self.plot_training_history(training_stats)
        
        return training_stats
    
    def evaluate(self, dataloader):
        """Evaluasi model"""
        self.model.eval()
        total_eval_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            b_input_ids = batch['input_ids'].to(self.device)
            b_attention_mask = batch['attention_mask'].to(self.device)
            b_labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
            
            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == b_labels).sum().item()
            total += b_labels.size(0)
        
        avg_val_loss = total_eval_loss / len(dataloader)
        accuracy = correct / total
        
        return accuracy, avg_val_loss
    
    def test(self, test_dataloader):
        """Testing model"""
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        for batch in test_dataloader:
            b_input_ids = batch['input_ids'].to(self.device)
            b_attention_mask = batch['attention_mask'].to(self.device)
            b_labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_attention_mask
                )
                
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
        
        # Hitung metrik
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                      target_names=['VALID', 'HOAX'], digits=4)
        cm = confusion_matrix(true_labels, predictions)
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST SET RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, ['VALID', 'HOAX'])
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_training_history(self, training_stats):
        """Plot training history"""
        epochs = [stat['epoch'] for stat in training_stats]
        train_losses = [stat['train_loss'] for stat in training_stats]
        val_losses = [stat['val_loss'] for stat in training_stats]
        train_accs = [stat['train_accuracy'] for stat in training_stats]
        val_accs = [stat['val_accuracy'] for stat in training_stats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'), dpi=100)
        plt.show()
    
    def plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), dpi=100)
        plt.show()
    
    def save_model(self, path=None):
        """Simpan model dan tokenizer"""
        if path is None:
            path = self.model_path
        
        # Simpan model
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.tokenizer_path)
        
        logger.info(f"Model disimpan ke: {self.model_path}")
        logger.info(f"Tokenizer disimpan ke: {self.tokenizer_path}")
    
    def load_model(self, model_path=None, tokenizer_path=None):
        """Load model yang sudah disimpan"""
        if model_path is None:
            model_path = self.model_path
        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path
        
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model.to(self.device)
        
        logger.info(f"Model dimuat dari: {model_path}")

def main():
    """Main function untuk training"""
    print("\n" + "="*70)
    print("INDOBERT HOAX DETECTION MODEL TRAINER")
    print("="*70)
    
    # Buat trainer
    trainer = ModelTrainer()
    
    print("\n[1] Loading data...")
    
    # Coba load dari CSV, jika tidak ada gunakan data contoh
    texts, labels = trainer.load_dataset_from_csv()
    
    if texts is None:
        print("Dataset CSV tidak ditemukan, menggunakan data contoh...")
        texts, labels = trainer.load_sample_data()
    
    print(f"Loaded {len(texts)} samples")
    print(f"Hoax samples: {sum(labels)}")
    print(f"Valid samples: {len(labels) - sum(labels)}")
    
    print("\n[2] Preparing data...")
    train_loader, val_loader, test_loader = trainer.prepare_data(
        texts, labels, 
        test_size=0.2, 
        val_size=0.1, 
        batch_size=8
    )
    
    print("\n[3] Training model...")
    training_stats = trainer.train(
        train_loader, 
        val_loader,
        epochs=100,
        learning_rate=2e-5
    )
    
    print("\n[4] Testing model...")
    test_results = trainer.test(test_loader)
    
    print("\n[5] Saving final model...")
    trainer.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: {trainer.model_path}")
    print(f"Tokenizer saved to: {trainer.tokenizer_path}")
    print(f"Final test accuracy: {test_results['accuracy']:.4f}")
    
    # Buat summary file
    summary = {
        'model_name': trainer.model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(texts),
        'hoax_samples': sum(labels),
        'valid_samples': len(labels) - sum(labels),
        'final_test_accuracy': float(test_results['accuracy']),
        'best_val_accuracy': float(max([stat['val_accuracy'] for stat in training_stats])),
        'training_stats': training_stats
    }
    
    import json
    summary_path = os.path.join(trainer.model_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()