# train_multiple_files.py
import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiFileDataset(Dataset):
    """Dataset untuk data dari multiple files"""
    
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

class MultiFileTrainer:
    """Trainer untuk multiple file dataset"""
    
    def __init__(self, model_name='indobenchmark/indobert-base-p2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Menggunakan device: {self.device}")
        
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model.to(self.device)
        
        # Setup directory
        self.model_dir = 'trained_models'
        self.model_path = os.path.join(self.model_dir, 'indobert_model')
        self.tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.tokenizer_path, exist_ok=True)
    
    def find_data_files(self):
        """Cari semua file data di folder"""
        data_files = []
        extensions = ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.txt']
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in extensions):
                # Skip file training/test
                if 'train' in file.lower() or 'test' in file.lower():
                    continue
                data_files.append(file)
        
        return sorted(data_files)
    
    def load_single_file(self, file_path):
        """Load single file dengan format detection"""
        logger.info(f"Loading: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.txt'):
            # Format TXT: satu baris per sample, label dipisahkan tab
            texts, labels = [], []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        text, label = line.split('\t', 1)
                        texts.append(text.strip())
                        labels.append(int(label.strip()))
            df = pd.DataFrame({'text': texts, 'label': labels})
        else:
            raise ValueError(f"Format tidak didukung: {file_path}")
        
        logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return df
    
    def combine_multiple_files(self, file_paths):
        """Gabungkan multiple files menjadi satu dataset"""
        all_dfs = []
        
        for file_path in file_paths:
            try:
                df = self.load_single_file(file_path)
                all_dfs.append(df)
                logger.info(f"✓ Loaded {len(df)} samples from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"✗ Error loading {file_path}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("Tidak ada file yang berhasil di-load")
        
        # Gabungkan semua DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"\nTotal combined samples: {len(combined_df)}")
        
        return combined_df
    
    def prepare_combined_data(self, combined_df, text_column='text', label_column='label'):
        """Persiapkan data dari DataFrame gabungan"""
        
        # Auto detect columns
        available_columns = combined_df.columns.tolist()
        logger.info(f"Available columns in combined data: {available_columns}")
        
        # Auto detect text column
        if text_column not in combined_df.columns:
            possible_text_columns = ['text', 'content', 'article', 'tweet', 'post', 
                                    'message', 'news', 'berita', 'konten']
            for col in possible_text_columns:
                if col in combined_df.columns:
                    text_column = col
                    logger.info(f"Auto detected text column: {text_column}")
                    break
        
        # Auto detect label column
        if label_column not in combined_df.columns:
            possible_label_columns = ['label', 'class', 'is_hoax', 'hoax', 
                                     'category', 'type', 'label_hoax']
            for col in possible_label_columns:
                if col in combined_df.columns:
                    label_column = col
                    logger.info(f"Auto detected label column: {label_column}")
                    break
        
        if text_column not in combined_df.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        if label_column not in combined_df.columns:
            raise ValueError(f"Label column '{label_column}' not found")
        
        # Ambil data
        texts = combined_df[text_column].astype(str).values
        labels = combined_df[label_column].values
        
        # Handle different label formats
        labels = self.normalize_labels(labels)
        
        # Validasi dan filter
        valid_indices = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if isinstance(text, str) and text.strip() and len(text.strip()) > 10:
                try:
                    int(label)  # Pastikan bisa di-convert ke int
                    valid_indices.append(i)
                except:
                    continue
        
        texts = texts[valid_indices]
        labels = labels[valid_indices]
        
        logger.info(f"Valid samples after filtering: {len(texts)}")
        logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")
        
        return texts, labels, text_column, label_column
    
    def normalize_labels(self, labels):
        """Normalisasi berbagai format label ke binary (0/1)"""
        unique_labels = np.unique(labels)
        logger.info(f"Unique label values found: {unique_labels}")
        
        # Convert ke numpy array
        labels = np.array(labels)
        
        # Case 1: Already binary (0/1)
        if set(unique_labels).issubset({0, 1}):
            logger.info("Labels already binary (0/1)")
            return labels.astype(int)
        
        # Case 2: String labels
        if labels.dtype == object:
            label_map = {}
            
            # Coba mapping otomatis
            hoax_keywords = ['hoax', 'fake', 'false', '1', 'true', 'yes', 'ya']
            valid_keywords = ['valid', 'real', 'true', '0', 'false', 'no', 'tidak']
            
            for label in unique_labels:
                label_str = str(label).lower().strip()
                
                if any(keyword in label_str for keyword in hoax_keywords):
                    label_map[label] = 1
                elif any(keyword in label_str for keyword in valid_keywords):
                    label_map[label] = 0
                else:
                    # Default mapping
                    label_map[label] = 0
            
            logger.info(f"Label mapping: {label_map}")
            labels = np.array([label_map[l] for l in labels])
        
        # Case 3: Boolean
        elif labels.dtype == bool:
            labels = labels.astype(int)
        
        # Case 4: Multiple categories (0, 1, 2, ...)
        elif len(unique_labels) > 2:
            logger.warning(f"Multiple categories found: {unique_labels}")
            logger.info("Mapping non-zero to 1 (hoax), 0 to 0 (valid)")
            labels = np.array([1 if l != 0 else 0 for l in labels])
        
        return labels.astype(int)
    
    def analyze_combined_data(self, texts, labels, file_names):
        """Analisis dataset gabungan"""
        logger.info("\n" + "="*60)
        logger.info("COMBINED DATASET ANALYSIS")
        logger.info("="*60)
        
        total = len(texts)
        hoax_count = np.sum(labels)
        valid_count = total - hoax_count
        
        logger.info(f"Total samples from {len(file_names)} files: {total}")
        logger.info(f"Hoax samples (1): {hoax_count} ({hoax_count/total:.1%})")
        logger.info(f"Valid samples (0): {valid_count} ({valid_count/total:.1%})")
        
        # Text length analysis
        text_lengths = [len(str(t).split()) for t in texts]
        
        logger.info(f"\nText length statistics (words):")
        logger.info(f"  Min: {np.min(text_lengths)}")
        logger.info(f"  Max: {np.max(text_lengths)}")
        logger.info(f"  Mean: {np.mean(text_lengths):.1f}")
        logger.info(f"  Std: {np.std(text_lengths):.1f}")
        
        # File contribution
        logger.info(f"\nFiles used:")
        for file in file_names:
            logger.info(f"  • {file}")
        
        # Visualize
        self.plot_combined_distribution(texts, labels, text_lengths, file_names)
        
        return {
            'total_samples': total,
            'hoax_samples': int(hoax_count),
            'valid_samples': int(valid_count),
            'avg_text_length': float(np.mean(text_lengths)),
            'num_files': len(file_names),
            'files': file_names
        }
    
    def plot_combined_distribution(self, texts, labels, text_lengths, file_names):
        """Plot distribusi data gabungan"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Label distribution
        counts = np.bincount(labels)
        axes[0, 0].bar(['VALID', 'HOAX'], counts, color=['green', 'red'])
        axes[0, 0].set_title('Distribusi Label Dataset Gabungan')
        axes[0, 0].set_ylabel('Jumlah Sampel')
        
        # Plot 2: Text length distribution
        axes[0, 1].hist(text_lengths, bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_title('Distribusi Panjang Teks')
        axes[0, 1].set_xlabel('Jumlah Kata')
        axes[0, 1].set_ylabel('Frekuensi')
        axes[0, 1].axvline(np.mean(text_lengths), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(text_lengths):.1f}')
        axes[0, 1].legend()
        
        #Plot 3: Label balance
        axes[1, 0].pie([counts[0], counts[1]], 
                      labels=['Valid', 'Hoax'],
                      colors=['green', 'red'],
                      autopct='%1.1f%%')
        axes[1, 0].set_title('Balance Ratio')
        
        # Plot 4: Files info
        axes[1, 1].axis('off')
        files_text = f"Total Files: {len(file_names)}\n\n"
        for i, file in enumerate(file_names[:10], 1):
            files_text += f"{i}. {file}\n"
        if len(file_names) > 10:
            files_text += f"... and {len(file_names)-10} more"
        
        axes[1, 1].text(0.1, 0.5, files_text, fontsize=10,
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Files Used')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'combined_data_distribution.png'), dpi=100)
        plt.show()
    
    def prepare_dataloaders(self, texts, labels, batch_size=16, 
                           test_size=0.2, val_size=0.1):
        """Persiapkan dataloaders untuk training"""
        
        # Split data: train, validation, test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=(test_size + val_size),
            random_state=42,
            stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=test_size/(test_size + val_size),
            random_state=42,
            stratify=temp_labels
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training:   {len(train_texts)} samples")
        logger.info(f"  Validation: {len(val_texts)} samples")
        logger.info(f"  Test:       {len(test_texts)} samples")
        
        # Buat datasets
        train_dataset = MultiFileDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MultiFileDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = MultiFileDataset(test_texts, test_labels, self.tokenizer)
        
        # Buat dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=3e-5):
        """Training loop"""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Early stopping
        best_val_acc = 0
        patience = 3
        patience_counter = 0
        
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct/total
                })
            
            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation phase
            val_acc, avg_val_loss = self.evaluate(val_loader)
            
            # Save history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Train Acc:  {train_acc:.4f}")
            logger.info(f"Val Loss:   {avg_val_loss:.4f}")
            logger.info(f"Val Acc:    {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model()
                logger.info(f"✓ Best model saved! (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                logger.info(f"ⓘ No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    logger.info(f"⚠ Early stopping at epoch {epoch + 1}")
                    break
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def evaluate(self, dataloader):
        """Evaluasi model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return accuracy, avg_loss
    
    def test(self, test_loader):
        """Testing model"""
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels, predictions,
            target_names=['VALID', 'HOAX'],
            digits=4
        )
        cm = confusion_matrix(true_labels, predictions)
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Save detailed results
        self.save_test_results(predictions, true_labels, probabilities)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = history['epoch']
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Training History - Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history_multifile.png'), dpi=100)
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['VALID', 'HOAX'],
                   yticklabels=['VALID', 'HOAX'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix_multifile.png'), dpi=100)
        plt.show()
    
    def save_test_results(self, predictions, true_labels, probabilities):
        """Save test results"""
        results_df = pd.DataFrame({
            'true_label': true_labels,
            'prediction': predictions,
            'hoax_probability': [p[1] for p in probabilities],
            'valid_probability': [p[0] for p in probabilities],
            'correct': [t == p for t, p in zip(true_labels, predictions)]
        })
        
        results_path = os.path.join(self.model_dir, 'test_results_multifile.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Test results saved to: {results_path}")
    
    def save_model(self):
        """Simpan model"""
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.tokenizer_path)
        logger.info(f"Model saved to: {self.model_path}")
    
    def save_combined_dataset(self, combined_df, filename="combined_dataset.csv"):
        """Simpan dataset gabungan untuk referensi"""
        combined_path = os.path.join(self.model_dir, filename)
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        logger.info(f"Combined dataset saved to: {combined_path}")
        return combined_path
    
    def demo_predictions(self, examples=None):
        """Demo predictions"""
        if examples is None:
            examples = [
                "Vaksin COVID-19 mengandung microchip untuk melacak masyarakat.",
                "Pemerintah meluncurkan program vaksinasi COVID-19 tahap ketiga.",
                "Gempa 9 SR akan mengguncang Jawa besok berdasarkan prediksi.",
                "Bank Indonesia menjaga stabilitas nilai tukar rupiah."
            ]
        
        logger.info(f"\n{'='*60}")
        logger.info("DEMO PREDICTIONS")
        logger.info(f"{'='*60}")
        
        for text in examples:
            result = self.predict(text)
            prediction = "HOAX" if result['prediction'] == 1 else "VALID"
            
            logger.info(f"\nText: {text[:80]}...")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Hoax Probability: {result['hoax_probability']:.2%}")
    
    def predict(self, text, threshold=0.5):
        """Prediksi untuk satu teks"""
        self.model.eval()
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            hoax_prob = probabilities[0][1].item()
            valid_prob = probabilities[0][0].item()
            
            if hoax_prob >= threshold:
                prediction = 1
                confidence = hoax_prob
            else:
                prediction = 0
                confidence = valid_prob
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'hoax_probability': hoax_prob,
            'valid_probability': valid_prob,
            'text': text[:200]
        }

def main():
    """Main function untuk training dari multiple files"""
    print("\n" + "="*70)
    print("HOAX DETECTOR TRAINING FROM MULTIPLE FILES")
    print("="*70)
    
    # Buat trainer
    trainer = MultiFileTrainer()
    
    # Cari file data
    print("\n[1] Searching for data files...")
    data_files = trainer.find_data_files()
    
    if not data_files:
        print("\n✗ Tidak ditemukan file data!")
        print("\nFormat yang didukung: .csv, .json, .xlsx, .xls, .parquet, .txt")
        print("\nLetakkan file data di folder yang sama dengan script ini.")
        return
    
    print(f"\n✓ Ditemukan {len(data_files)} file data:")
    for i, file in enumerate(data_files, 1):
        size = os.path.getsize(file)
        print(f"{i}. {file} ({size:,} bytes)")
    
    # Pilih file
    print(f"\nPilih file untuk training:")
    print("1. Gunakan SEMUA file")
    print("2. Pilih file tertentu")
    
    choice = input("\nPilihan (1-2): ").strip() or '1'
    
    if choice == '1':
        selected_files = data_files
    elif choice == '2':
        print("\nMasukkan nomor file (pisahkan dengan koma):")
        print("Contoh: 1,3,4")
        
        try:
            indices = input("Pilihan: ").strip()
            indices = [int(i.strip()) - 1 for i in indices.split(',')]
            selected_files = [data_files[i] for i in indices if 0 <= i < len(data_files)]
        except:
            print("Pilihan tidak valid, menggunakan semua file")
            selected_files = data_files
    else:
        selected_files = data_files
    
    print(f"\nMenggunakan {len(selected_files)} file:")
    for file in selected_files:
        print(f"  • {file}")
    
    # Load dan gabungkan data
    print("\n[2] Loading and combining data...")
    combined_df = trainer.combine_multiple_files(selected_files)
    
    # Tentukan kolom
    print(f"\n[3] Auto-detecting columns...")
    print(f"Columns available: {list(combined_df.columns)}")
    
    text_col = input(f"Nama kolom untuk teks (default 'text'): ").strip() or 'text'
    label_col = input(f"Nama kolom untuk label (default 'label'): ").strip() or 'label'
    
    # Prepare data
    print("\n[4] Preparing combined data...")
    texts, labels, text_col, label_col = trainer.prepare_combined_data(
        combined_df, text_col, label_col
    )
    
    # Analyze data
    print("\n[5] Analyzing combined dataset...")
    data_stats = trainer.analyze_combined_data(texts, labels, selected_files)
    
    # Save combined dataset
    combined_path = trainer.save_combined_dataset(combined_df)
    
    # Training configuration
    print("\n[6] Training configuration:")
    
    epochs = int(input(f"Epochs (default 10): ") or "10")
    batch_size = int(input(f"Batch size (default 16): ") or "16")
    learning_rate = float(input(f"Learning rate (default 3e-5): ") or "3e-5")
    
    # Prepare dataloaders
    print("\n[7] Preparing dataloaders...")
    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        texts, labels,
        batch_size=batch_size,
        test_size=0.2,
        val_size=0.1
    )
    
    # Training
    print("\n[8] Training model...")
    print("Proses training dimulai. Mohon tunggu...")
    
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Testing
    print("\n[9] Testing model...")
    test_results = trainer.test(test_loader)
    
    # Demo
    print("\n[10] Demo predictions...")
    trainer.demo_predictions()
    
    # Save summary
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': trainer.model_name,
        'files_used': selected_files,
        'text_column': text_col,
        'label_column': label_col,
        'data_stats': data_stats,
        'combined_dataset': combined_path,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        },
        'test_accuracy': float(test_results['accuracy']),
        'best_val_accuracy': float(max(history['val_acc']) if history['val_acc'] else 0)
    }
    
    summary_path = os.path.join(trainer.model_dir, 'training_summary_multifile.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Training summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("TRAINING FROM MULTIPLE FILES COMPLETED!")
    print("="*70)
    print(f"\nModel disimpan ke: trained_models/")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Total samples used: {len(texts)}")
    print(f"Files used: {len(selected_files)}")
    
    print(f"\nUntuk menjalankan aplikasi Django:")
    print("1. python manage.py runserver")
    print("2. Buka http://localhost:8000")
    
    return trainer

if __name__ == "__main__":
    trainer = main()