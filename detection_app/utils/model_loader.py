# detection_app/utils/model_loader.py
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from django.conf import settings
import os

logger = logging.getLogger(__name__)

class HoaxDetectionModel:
    """Singleton class untuk memuat dan menggunakan model IndoBERT"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HoaxDetectionModel, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        """Inisialisasi model dan tokenizer"""
        try:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Menggunakan device: {self._device}")
            
            # Cek apakah model sudah ada di local
            if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.TOKENIZER_PATH):
                logger.info("Memuat model dari local storage...")
                self._tokenizer = BertTokenizer.from_pretrained(settings.TOKENIZER_PATH)
                self._model = BertForSequenceClassification.from_pretrained(settings.MODEL_PATH)
            else:
                logger.info("Model tidak ditemukan di local, menggunakan IndoBERT base...")
                self._tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
                self._model = BertForSequenceClassification.from_pretrained(
                    'indobenchmark/indobert-base-p2',
                    num_labels=2
                )
            
            self._model.to(self._device)
            self._model.eval()
            logger.info("Model berhasil dimuat!")
            
        except Exception as e:
            logger.error(f"Error memuat model: {e}")
            raise
    
    @property
    def model(self):
        """Getter untuk model"""
        if self._model is None:
            self._initialize_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Getter untuk tokenizer"""
        if self._tokenizer is None:
            self._initialize_model()
        return self._tokenizer
    
    @property
    def device(self):
        """Getter untuk device"""
        return self._device
    
    def predict(self, text, threshold=0.5):
        """
        Memprediksi apakah teks adalah hoax
        
        Args:
            text (str): Teks berita
            threshold (float): Threshold untuk klasifikasi
        
        Returns:
            dict: Hasil prediksi
        """
        try:
            # Tokenisasi
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )
            
            # Pindahkan ke device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Prediksi
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Ambil probabilitas
            hoax_prob = probabilities[0][1].item()
            valid_prob = probabilities[0][0].item()
            
            # Tentukan hasil
            if hoax_prob >= threshold:
                prediction = 'HOAX'
                confidence = hoax_prob
            else:
                prediction = 'VALID'
                confidence = valid_prob
            
            return {
                'text': text,
                'prediction': prediction,
                'confidence': confidence,
                'hoax_probability': hoax_prob,
                'valid_probability': valid_prob,
                'threshold_used': threshold,
                'model_name': 'IndoBERT'
            }
            
        except Exception as e:
            logger.error(f"Error saat prediksi: {e}")
            return {
                'text': text,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, texts, threshold=0.5):
        """Memprediksi batch teks sekaligus"""
        results = []
        for text in texts:
            results.append(self.predict(text, threshold))
        return results

# Global instance
model_instance = HoaxDetectionModel()