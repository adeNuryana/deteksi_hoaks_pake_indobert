# detection_app/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Feedback

class DetectionForm(forms.Form):
    """Form untuk input teks deteksi"""
    
    news_text = forms.CharField(
        label="Masukkan Teks Berita",
        widget=forms.Textarea(attrs={
            'rows': 8,
            'class': 'form-control',
            'placeholder': 'Masukkan teks berita yang ingin Anda periksa...',
            'style': 'resize: vertical; min-height: 150px;'
        }),
        max_length=5000,
        required=True
    )
    
    source_url = forms.URLField(
        label="URL Sumber (Opsional)",
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://example.com/berita'
        }),
        required=False
    )
    
    check_patterns = forms.BooleanField(
        label="Cek pola hoax umum",
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )

class BatchDetectionForm(forms.Form):
    """Form untuk deteksi batch"""
    
    file_upload = forms.FileField(
        label="Upload File",
        help_text="Upload file CSV atau TXT. Format CSV: kolom 'text' wajib ada",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.txt'
        })
    )
    
    file_type = forms.ChoiceField(
        label="Tipe File",
        choices=[
            ('csv', 'CSV File'),
            ('txt', 'TXT File (satu baris per berita)'),
        ],
        widget=forms.RadioSelect(attrs={
            'class': 'form-check-input'
        }),
        initial='csv'
    )

class URLDetectionForm(forms.Form):
    """Form untuk deteksi dari URL"""
    
    url = forms.URLField(
        label="URL Berita",
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://example.com/berita'
        })
    )
    
    extract_content = forms.BooleanField(
        label="Ekstrak konten otomatis",
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )

class FeedbackForm(forms.ModelForm):
    """Form untuk memberikan feedback"""
    
    class Meta:
        model = Feedback
        fields = ['is_correct', 'suggested_label', 'comments']
        widgets = {
            'is_correct': forms.RadioSelect(choices=[(True, 'Ya'), (False, 'Tidak')]),
            'suggested_label': forms.Select(attrs={'class': 'form-control'}),
            'comments': forms.Textarea(attrs={
                'rows': 3,
                'class': 'form-control',
                'placeholder': 'Berikan komentar atau saran...'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['suggested_label'].required = False

class CustomUserCreationForm(UserCreationForm):
    """Form untuk registrasi user"""
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({'class': 'form-control'})
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user