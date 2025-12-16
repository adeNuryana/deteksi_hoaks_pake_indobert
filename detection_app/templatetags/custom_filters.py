from django import template

register = template.Library()

# Contoh filter dasar yang mungkin Anda butuhkan:
@register.filter
def potong_teks(value, panjang=100):
    """Memotong teks jika terlalu panjang"""
    if len(value) > panjang:
        return value[:panjang] + '...'
    return value

@register.filter
def format_persen(value):
    """Memformat angka menjadi persentase"""
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return value

@register.filter
def ambil_item(dictionary, kunci):
    """Mengambil nilai dari dictionary berdasarkan kunci"""
    return dictionary.get(kunci, '')

@register.filter
def kapital_awal(value):
    """Membuat huruf pertama kapital"""
    if value:
        return value[0].upper() + value[1:]
    return value

# Tambahkan filter lain sesuai kebutuhan aplikasi NLP Anda
# detection_app/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiply the value by the arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value):
    """Convert float to percentage"""
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "0%"

@register.filter
def format_percent(value):
    """Format as percentage without % sign"""
    try:
        return f"{float(value) * 100:.1f}"
    except (ValueError, TypeError):
        return "0"

@register.filter
def format_float(value, decimals=2):
    """Format float with specific decimals"""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}"

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary"""
    return dictionary.get(key, '')

@register.filter
def truncate_words(value, num_words=20):
    """Truncate text to number of words"""
    words = value.split()
    if len(words) > num_words:
        return ' '.join(words[:num_words]) + '...'
    return value

@register.filter
def is_hoax(value):
    """Check if prediction is HOAX"""
    return value == 'HOAX'