# detection_app/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class DetectionHistory(models.Model):
    """Model untuk menyimpan history deteksi"""
    
    HOAX = 'HOAX'
    VALID = 'VALID'
    UNCERTAIN = 'UNCERTAIN'
    
    STATUS_CHOICES = [
        (HOAX, 'Hoax'),
        (VALID, 'Valid'),
        (UNCERTAIN, 'Perlu Verifikasi'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='detection_history'
    )
    
    text = models.TextField(verbose_name="Teks Berita")
    result = models.CharField(
        max_length=20, 
        choices=STATUS_CHOICES,
        verbose_name="Hasil Deteksi"
    )
    
    confidence = models.FloatField(
        verbose_name="Tingkat Keyakinan",
        help_text="Persentase keyakinan model (0-1)"
    )
    
    hoax_probability = models.FloatField(
        verbose_name="Probabilitas Hoax",
        default=0.0
    )
    
    valid_probability = models.FloatField(
        verbose_name="Probabilitas Valid",
        default=0.0
    )
    
    source_url = models.URLField(
        blank=True, 
        null=True,
        verbose_name="URL Sumber"
    )
    
    metadata = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Metadata Tambahan"
    )
    
    created_at = models.DateTimeField(
        default=timezone.now,
        verbose_name="Waktu Deteksi"
    )
    
    ip_address = models.GenericIPAddressField(
        blank=True, 
        null=True,
        verbose_name="Alamat IP"
    )
    
    is_verified = models.BooleanField(
        default=False,
        verbose_name="Terverifikasi Manual"
    )
    
    verified_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='verified_detections'
    )
    
    verification_date = models.DateTimeField(
        null=True, 
        blank=True,
        verbose_name="Waktu Verifikasi"
    )
    
    verification_notes = models.TextField(
        blank=True,
        verbose_name="Catatan Verifikasi"
    )
    
    class Meta:
        verbose_name = "History Deteksi"
        verbose_name_plural = "History Deteksi"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['result']),
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"Deteksi {self.id} - {self.result}"
    
    def get_truncated_text(self, length=100):
        """Mendapatkan teks yang dipotong"""
        if len(self.text) > length:
            return self.text[:length] + "..."
        return self.text

class HoaxPattern(models.Model):
    """Model untuk pola-pola hoax yang umum"""
    
    pattern_type = models.CharField(
        max_length=50,
        verbose_name="Jenis Pola",
        choices=[
            ('KEYWORD', 'Kata Kunci'),
            ('PHRASE', 'Frasa'),
            ('PATTERN', 'Pola Kalimat'),
            ('SOURCE', 'Sumber Terindikasi'),
        ]
    )
    
    pattern_content = models.TextField(verbose_name="Konten Pola")
    description = models.TextField(verbose_name="Deskripsi")
    severity = models.IntegerField(
        verbose_name="Tingkat Keparahan",
        choices=[(1, 'Rendah'), (2, 'Sedang'), (3, 'Tinggi')],
        default=2
    )
    
    is_active = models.BooleanField(default=True, verbose_name="Aktif")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Pola Hoax"
        verbose_name_plural = "Pola-pola Hoax"
    
    def __str__(self):
        return f"{self.pattern_type}: {self.pattern_content[:50]}"

class Feedback(models.Model):
    """Model untuk feedback dari pengguna"""
    
    detection = models.ForeignKey(
        DetectionHistory,
        on_delete=models.CASCADE,
        related_name='feedbacks'
    )
    
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    is_correct = models.BooleanField(verbose_name="Hasil Benar?")
    comments = models.TextField(blank=True, verbose_name="Komentar")
    suggested_label = models.CharField(
        max_length=20,
        choices=DetectionHistory.STATUS_CHOICES,
        blank=True
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Feedback"
        verbose_name_plural = "Feedback"
    
    def __str__(self):
        return f"Feedback untuk Deteksi {self.detection.id}"