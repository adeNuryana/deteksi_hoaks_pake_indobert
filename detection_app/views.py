# detection_app/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.utils import timezone
from django.db.models import Count, Q
from django.contrib.auth import login, authenticate
import json
import csv
import io
from datetime import datetime, timedelta

from .forms import (
    DetectionForm, BatchDetectionForm, 
    URLDetectionForm, FeedbackForm, CustomUserCreationForm
)
from .models import DetectionHistory, Feedback
from .utils.predictor import hoax_detector
from .utils.scraper import news_scraper
import logging

logger = logging.getLogger(__name__)

def home(request):
    """Home page"""
    context = {
        'title': 'Deteksi Berita Hoax - IndoBERT',
        'description': 'Sistem deteksi berita hoax menggunakan AI model IndoBERT',
        'stats': {
            'total_detections': DetectionHistory.objects.count(),
            'hoax_count': DetectionHistory.objects.filter(result='HOAX').count(),
            'valid_count': DetectionHistory.objects.filter(result='VALID').count(),
            'today_detections': DetectionHistory.objects.filter(
                created_at__date=timezone.now().date()
            ).count(),
        }
    }
    return render(request, 'detection_app/home.html', context)

def detect_news(request):
    """Halaman deteksi tunggal"""
    if request.method == 'POST':
        form = DetectionForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['news_text']
            source_url = form.cleaned_data['source_url']
            check_patterns = form.cleaned_data['check_patterns']
            
            # Lakukan deteksi
            result = hoax_detector.detect(text, check_patterns)
            
            # Simpan ke history
            detection = DetectionHistory.objects.create(
                user=request.user if request.user.is_authenticated else None,
                text=text[:2000],  # Potong jika terlalu panjang
                result=result['prediction'],
                confidence=result['confidence'],
                hoax_probability=result['hoax_probability'],
                valid_probability=result['valid_probability'],
                source_url=source_url,
                metadata={
                    'text_analysis': result.get('text_analysis', {}),
                    'patterns_detected': result.get('patterns_detected', []),
                    'risk_factors': result.get('risk_factors', []),
                },
                ip_address=request.META.get('REMOTE_ADDR'),
            )
            
            # Tambah ID deteksi ke result
            result['detection_id'] = detection.id
            
            context = {
                'form': form,
                'result': result,
                'detection': detection,
                'title': 'Hasil Deteksi'
            }
            
            return render(request, 'detection_app/result.html', context)
    else:
        form = DetectionForm()
    
    context = {
        'form': form,
        'title': 'Deteksi Berita Hoax'
    }
    return render(request, 'detection_app/detect.html', context)

def detect_from_url(request):
    """Deteksi hoax dari URL"""
    if request.method == 'POST':
        form = URLDetectionForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['url']
            extract_content = form.cleaned_data['extract_content']
            
            if extract_content:
                # Scrape konten dari URL
                scraped = news_scraper.extract_from_url(url)
                
                if scraped['success']:
                    text = f"{scraped['title']}\n\n{scraped['content']}"
                    source_info = f"Sumber: {scraped['source']} ({scraped['domain']})"
                else:
                    messages.error(request, f"Gagal mengambil konten: {scraped['error']}")
                    return render(request, 'detection_app/detect_url.html', {'form': form})
            else:
                # Gunakan URL sebagai referensi saja
                text = f"Berita dari: {url}"
                source_info = ""
            
            # Lakukan deteksi
            result = hoax_detector.detect(text, check_patterns=True)
            
            # Simpan ke history
            detection = DetectionHistory.objects.create(
                user=request.user if request.user.is_authenticated else None,
                text=text[:2000],
                result=result['prediction'],
                confidence=result['confidence'],
                hoax_probability=result['hoax_probability'],
                valid_probability=result['valid_probability'],
                source_url=url,
                metadata={
                    'scraped_info': scraped if extract_content else {},
                    'text_analysis': result.get('text_analysis', {}),
                    'source_info': source_info,
                },
                ip_address=request.META.get('REMOTE_ADDR'),
            )
            
            result['detection_id'] = detection.id
            result['scraped_info'] = scraped if extract_content else None
            
            context = {
                'form': form,
                'result': result,
                'detection': detection,
                'title': 'Hasil Deteksi dari URL'
            }
            
            return render(request, 'detection_app/result.html', context)
    else:
        form = URLDetectionForm()
    
    context = {
        'form': form,
        'title': 'Deteksi dari URL'
    }
    return render(request, 'detection_app/detect_url.html', context)

def batch_detect(request):
    """Deteksi batch dari file"""
    if request.method == 'POST':
        form = BatchDetectionForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file_upload']
            file_type = form.cleaned_data['file_type']
            
            try:
                if file_type == 'csv':
                    # Baca CSV
                    csv_file = io.TextIOWrapper(file, encoding='utf-8')
                    reader = csv.DictReader(csv_file)
                    
                    if 'text' not in reader.fieldnames:
                        messages.error(request, "File CSV harus memiliki kolom 'text'")
                        return redirect('batch_detect')
                    
                    texts = [row['text'] for row in reader if row.get('text')]
                
                else:  # TXT
                    content = file.read().decode('utf-8')
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
                
                if not texts:
                    messages.error(request, "File kosong atau tidak ada teks yang valid")
                    return redirect('batch_detect')
                
                if len(texts) > 100:  # Limit
                    texts = texts[:100]
                    messages.warning(request, f"Hanya memproses 100 teks pertama dari {len(texts)}")
                
                # Proses batch
                results = []
                for text in texts:
                    result = hoax_detector.detect(text, check_patterns=True)
                    results.append(result)
                
                # Statistik
                hoax_count = sum(1 for r in results if r['prediction'] == 'HOAX')
                valid_count = len(results) - hoax_count
                
                context = {
                    'form': form,
                    'results': results,
                    'total_count': len(results),
                    'hoax_count': hoax_count,
                    'valid_count': valid_count,
                    'title': 'Hasil Deteksi Batch'
                }
                
                return render(request, 'detection_app/batch_result.html', context)
                
            except Exception as e:
                logger.error(f"Error processing batch file: {e}")
                messages.error(request, f"Error memproses file: {str(e)}")
                return redirect('batch_detect')
    else:
        form = BatchDetectionForm()
    
    context = {
        'form': form,
        'title': 'Deteksi Batch'
    }
    return render(request, 'detection_app/batch_detect.html', context)

@login_required
def detection_history(request):
    """History deteksi pengguna"""
    detections = DetectionHistory.objects.filter(user=request.user).order_by('-created_at')
    
    # Filter
    result_filter = request.GET.get('result', 'all')
    date_filter = request.GET.get('date', 'all')
    
    if result_filter != 'all':
        detections = detections.filter(result=result_filter)
    
    if date_filter == 'today':
        detections = detections.filter(created_at__date=timezone.now().date())
    elif date_filter == 'week':
        week_ago = timezone.now() - timedelta(days=7)
        detections = detections.filter(created_at__gte=week_ago)
    elif date_filter == 'month':
        month_ago = timezone.now() - timedelta(days=30)
        detections = detections.filter(created_at__gte=month_ago)
    
    # Pagination
    paginator = Paginator(detections, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistik
    stats = {
        'total': detections.count(),
        'hoax': detections.filter(result='HOAX').count(),
        'valid': detections.filter(result='VALID').count(),
    }
    
    context = {
        'page_obj': page_obj,
        'stats': stats,
        'filters': {
            'result': result_filter,
            'date': date_filter,
        },
        'title': 'History Deteksi'
    }
    
    return render(request, 'detection_app/history.html', context)

@login_required
def detection_detail(request, detection_id):
    """Detail deteksi"""
    detection = get_object_or_404(DetectionHistory, id=detection_id, user=request.user)
    
    if request.method == 'POST':
        feedback_form = FeedbackForm(request.POST)
        if feedback_form.is_valid():
            feedback = feedback_form.save(commit=False)
            feedback.detection = detection
            feedback.user = request.user
            feedback.save()
            
            messages.success(request, "Feedback berhasil disimpan!")
            return redirect('detection_detail', detection_id=detection_id)
    else:
        feedback_form = FeedbackForm()
    
    context = {
        'detection': detection,
        'feedback_form': feedback_form,
        'title': f'Detail Deteksi #{detection.id}'
    }
    
    return render(request, 'detection_app/detection_detail.html', context)

def api_docs(request):
    """API documentation page"""
    return render(request, 'detection_app/api_docs.html', {
        'title': 'API Documentation'
    })

def about(request):
    """About page"""
    return render(request, 'detection_app/about.html', {
        'title': 'Tentang Kami',
        'description': 'Informasi tentang sistem deteksi hoax ini'
    })

def register(request):
    """User registration"""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Auto login
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, "Registrasi berhasil! Selamat datang.")
                return redirect('home')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'detection_app/register.html', {
        'form': form,
        'title': 'Registrasi'
    })

# API Views
@csrf_exempt
@require_POST
def api_detect(request):
    """API endpoint untuk deteksi"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return JsonResponse({
                'error': 'Text is required'
            }, status=400)
        
        # Deteksi
        result = hoax_detector.detect(text, check_patterns=True)
        
        # Hapus field yang tidak perlu untuk API
        result.pop('text', None)
        
        return JsonResponse({
            'success': True,
            'data': result
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        logger.error(f"API error: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)

@require_GET
def api_stats(request):
    """API endpoint untuk statistik"""
    stats = {
        'total_detections': DetectionHistory.objects.count(),
        'hoax_count': DetectionHistory.objects.filter(result='HOAX').count(),
        'valid_count': DetectionHistory.objects.filter(result='VALID').count(),
        'uncertain_count': DetectionHistory.objects.filter(result='UNCERTAIN').count(),
        'today_detections': DetectionHistory.objects.filter(
            created_at__date=timezone.now().date()
        ).count(),
        'avg_confidence': DetectionHistory.objects.aggregate(
            avg_conf=models.Avg('confidence')
        )['avg_conf'] or 0,
    }
    
    return JsonResponse({
        'success': True,
        'data': stats
    })

@login_required
def export_history(request, format_type='csv'):
    """Export history deteksi"""
    detections = DetectionHistory.objects.filter(user=request.user)
    
    if format_type == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="hoax_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['ID', 'Text', 'Result', 'Confidence', 'Hoax Probability', 
                        'Valid Probability', 'Source URL', 'Created At'])
        
        for detection in detections:
            writer.writerow([
                detection.id,
                detection.text[:200],
                detection.result,
                detection.confidence,
                detection.hoax_probability,
                detection.valid_probability,
                detection.source_url or '',
                detection.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return response
    
    elif format_type == 'json':
        data = list(detections.values(
            'id', 'text', 'result', 'confidence', 
            'hoax_probability', 'valid_probability', 
            'source_url', 'created_at'
        ))
        
        response = JsonResponse(data, safe=False)
        response['Content-Disposition'] = f'attachment; filename="hoax_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
        return response
    
    else:
        messages.error(request, "Format tidak didukung")
        return redirect('detection_history')