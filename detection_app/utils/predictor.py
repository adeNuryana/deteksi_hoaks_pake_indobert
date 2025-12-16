# detection_app/utils/predictor.py
import re
import logging
from typing import List, Dict, Any
from .model_loader import model_instance
from ..models import HoaxPattern

logger = logging.getLogger(__name__)

class HoaxDetector:
    """Class utama untuk deteksi hoax dengan berbagai fitur"""
    
    def __init__(self):
        self.model = model_instance
    
    def detect(self, text: str, check_patterns: bool = True) -> Dict[str, Any]:
        """
        Deteksi hoax dengan berbagai analisis
        
        Args:
            text (str): Teks berita
            check_patterns (bool): Cek pola hoax umum
        
        Returns:
            dict: Hasil analisis lengkap
        """
        # Prediksi dengan model IndoBERT
        prediction = self.model.predict(text)
        
        # Analisis tambahan
        analysis = {
            **prediction,
            'text_analysis': self._analyze_text(text),
            'patterns_detected': [],
            'risk_factors': [],
            'suggestions': []
        }
        
        # Cek pola hoax jika diminta
        if check_patterns:
            patterns = self._check_hoax_patterns(text)
            analysis['patterns_detected'] = patterns
            
            if patterns:
                analysis['risk_factors'].append('Mengandung pola hoax umum')
                analysis['suggestions'].append(
                    'Terdeteksi pola hoax umum. Verifikasi dari sumber resmi.'
                )
        
        # Cek karakteristik teks
        text_analysis = analysis['text_analysis']
        
        if text_analysis['exclamation_count'] > 3:
            analysis['risk_factors'].append('Banyak tanda seru (emosional)')
            analysis['suggestions'].append('Berita yang emosional seringkali kurang akurat.')
        
        if text_analysis['capital_ratio'] > 0.3:
            analysis['risk_factors'].append('Banyak huruf kapital')
            analysis['suggestions'].append('Penggunaan huruf kapital berlebihan merupakan ciri hoax.')
        
        if text_analysis['word_count'] < 50:
            analysis['risk_factors'].append('Teks terlalu pendek')
            analysis['suggestions'].append('Berita yang valid biasanya memiliki penjelasan lengkap.')
        
        # Generate explanation
        analysis['explanation'] = self._generate_explanation(analysis)
        
        # Risk level
        analysis['risk_level'] = self._calculate_risk_level(analysis)
        
        return analysis
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analisis karakteristik teks"""
        words = text.split()
        
        return {
            'word_count': len(words),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'contains_url': bool(re.search(r'https?://\S+', text)),
            'contains_number': bool(re.search(r'\d+', text)),
        }
    
    def _check_hoax_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Cek pola hoax yang umum"""
        patterns_found = []
        
        try:
            active_patterns = HoaxPattern.objects.filter(is_active=True)
            
            for pattern in active_patterns:
                if self._matches_pattern(text, pattern.pattern_content, pattern.pattern_type):
                    patterns_found.append({
                        'pattern': pattern.pattern_content,
                        'type': pattern.pattern_type,
                        'description': pattern.description,
                        'severity': pattern.severity
                    })
        
        except Exception as e:
            logger.error(f"Error checking patterns: {e}")
        
        # Tambahkan pola default jika tidak ada di database
        default_patterns = [
            (r'\b(sangat rahasia|konfidensial|hanya untuk anda)\b', 'PHRASE', 'Klaim eksklusivitas', 2),
            (r'\b(sebarkan|viralkan|forward)\b.*\b(sebelum dihapus)\b', 'PHRASE', 'Ajakan menyebar', 3),
            (r'\b(dokter|ahli|profesor)\b.*\b(mengungkapkan|menemukan)\b.*\b(rahasia|fakta mengejutkan)\b', 'PATTERN', 'Klaim otoritas', 2),
            (r'\b(100%|pasti|tidak diragukan lagi)\b.*\b(terbukti|efektif)\b', 'PATTERN', 'Klaim kepastian', 2),
            (r'\b(pemerintah|penguasa)\b.*\b(menyembunyikan|menutup-nutupi)\b', 'PATTERN', 'Teori konspirasi', 3),
        ]
        
        for pattern_regex, pattern_type, description, severity in default_patterns:
            if re.search(pattern_regex, text, re.IGNORECASE):
                patterns_found.append({
                    'pattern': pattern_regex,
                    'type': pattern_type,
                    'description': description,
                    'severity': severity
                })
        
        return patterns_found
    
    def _matches_pattern(self, text: str, pattern: str, pattern_type: str) -> bool:
        """Cek apakah teks cocok dengan pola"""
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        
        if pattern_type == 'KEYWORD':
            return pattern_lower in text_lower
        elif pattern_type == 'PHRASE':
            return pattern_lower in text_lower
        elif pattern_type == 'PATTERN':
            try:
                return bool(re.search(pattern_lower, text_lower))
            except:
                return pattern_lower in text_lower
        else:
            return False
    
    def _generate_explanation(self, analysis: Dict[str, Any]) -> str:
        """Generate penjelasan untuk hasil deteksi"""
        prediction = analysis['prediction']
        confidence = analysis['confidence']
        
        if prediction == 'HOAX':
            if confidence > 0.8:
                return (
                    "🔴 TINGKAT KEPERCAYAAN TINGGI - BERITA HOAX\n"
                    "Model mendeteksi dengan keyakinan tinggi bahwa berita ini adalah HOAX. "
                    "Berita mengandung indikasi kuat misinformation."
                )
            elif confidence > 0.6:
                return (
                    "🟡 TINGKAT KEPERCAYAAN SEDANG - KEMUNGKINAN HOAX\n"
                    "Model mendeteksi indikasi hoax pada berita ini. "
                    "Disarankan untuk melakukan verifikasi lebih lanjut dari sumber resmi."
                )
            else:
                return (
                    "🟠 PERLU VERIFIKASI - INDIKASI HOAX\n"
                    "Terdapat beberapa indikasi hoax pada berita ini. "
                    "Meskipun keyakinan model tidak tinggi, disarankan untuk berhati-hati."
                )
        else:
            if confidence > 0.8:
                return (
                    "🟢 TINGKAT KEPERCAYAAN TINGGI - BERITA VALID\n"
                    "Model mendeteksi dengan keyakinan tinggi bahwa berita ini VALID. "
                    "Namun tetap disarankan untuk selalu memverifikasi informasi."
                )
            elif confidence > 0.6:
                return (
                    "🔵 TINGKAT KEPERCAYAAN SEDANG - KEMUNGKINAN VALID\n"
                    "Berita ini tampak valid berdasarkan analisis model. "
                    "Tidak ditemukan indikasi hoax yang kuat."
                )
            else:
                return (
                    "⚪ PERLU VERIFIKASI - INDIKASI VALID\n"
                    "Model tidak dapat memastikan dengan keyakinan tinggi. "
                    "Disarankan untuk memeriksa sumber berita dan melakukan verifikasi."
                )
    
    def _calculate_risk_level(self, analysis: Dict[str, Any]) -> str:
        """Hitung tingkat risiko"""
        score = 0
        
        # Base score dari prediksi model
        if analysis['prediction'] == 'HOAX':
            score += analysis['hoax_probability'] * 100
        else:
            score += analysis['hoax_probability'] * 50
        
        # Tambah score berdasarkan pola
        for pattern in analysis['patterns_detected']:
            score += pattern['severity'] * 10
        
        # Tambah score berdasarkan analisis teks
        text_analysis = analysis['text_analysis']
        if text_analysis['exclamation_count'] > 3:
            score += 10
        if text_analysis['capital_ratio'] > 0.3:
            score += 10
        
        # Tentukan level risiko
        if score >= 70:
            return 'SANGAT TINGGI'
        elif score >= 50:
            return 'TINGGI'
        elif score >= 30:
            return 'SEDANG'
        else:
            return 'RENDAH'

# Global instance
hoax_detector = HoaxDetector()