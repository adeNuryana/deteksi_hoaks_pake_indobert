# generate_dataset.py
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
import json
from fake_useragent import UserAgent
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate dataset hoax dan valid dari berbagai sumber"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def clean_text(self, text):
        """Bersihkan teks"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()
    
    def get_hoax_from_turnbackhoax(self, num_pages=10):
        """Ambil data hoax dari TurnBackHoax.id"""
        articles = []
        base_url = "https://turnbackhoax.id"
        
        for page in range(1, num_pages + 1):
            try:
                url = f"{base_url}/page/{page}/"
                logger.info(f"Scraping page {page}: {url}")
                
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Cari artikel
                posts = soup.find_all('article')
                
                for post in posts:
                    try:
                        title_elem = post.find('h2', class_='entry-title')
                        if title_elem:
                            link_elem = title_elem.find('a', href=True)
                            if link_elem:
                                title = title_elem.text.strip()
                                link = link_elem['href']
                                
                                # Ambil konten artikel
                                article_content = self._get_turnbackhoax_article(link)
                                
                                if article_content and len(article_content) > 100:
                                    articles.append({
                                        'text': f"{title}. {article_content}",
                                        'label': 1,  # Hoax
                                        'source': 'turnbackhoax',
                                        'url': link
                                    })
                    except Exception as e:
                        logger.error(f"Error processing post: {e}")
                        continue
                
                time.sleep(2)  # Delay untuk menghormati server
                
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                continue
        
        return articles
    
    def _get_turnbackhoax_article(self, url):
        """Ambil konten artikel dari TurnBackHoax"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_div = soup.find('div', class_='entry-content')
            if content_div:
                # Hapus elemen yang tidak perlu
                for element in content_div.find_all(['script', 'style', 'iframe', 'div']):
                    if 'sharedaddy' in str(element.get('class', '')):
                        element.decompose()
                
                # Ambil teks
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
                return self.clean_text(content)
        
        except Exception as e:
            logger.error(f"Error getting article {url}: {e}")
        
        return ""
    
    def get_valid_news(self):
        """Generate berita valid contoh"""
        valid_articles = [
            {
                'text': "Pemerintah Indonesia mengumumkan program vaksinasi COVID-19 tahap ketiga akan dimulai bulan depan. Program ini menargetkan 10 juta penerima vaksin.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Bank Indonesia mencatat inflasi berada pada level 3.5% pada bulan Desember 2023, sesuai dengan target yang ditetapkan.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Timnas Indonesia akan bertanding dalam kualifikasi Piala Dunia minggu depan di Stadion Gelora Bung Karno, Jakarta.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Kementerian Pendidikan dan Kebudayaan meluncurkan program beasiswa untuk 10.000 mahasiswa berprestasi dari keluarga kurang mampu.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Harga cabai rawit mengalami kenaikan 20% akibat cuaca ekstrem di daerah produsen utama seperti Brebes dan Cianjur.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "PT Kereta Api Indonesia menambah jumlah perjalanan kereta selama musim liburan Natal dan Tahun Baru untuk mengakomodasi lonjakan penumpang.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Badan Meteorologi, Klimatologi, dan Geofisika memprediksi hujan dengan intensitas sedang akan turun di wilayah Jakarta dan sekitarnya besok.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Ekspor komoditas kelapa sawit Indonesia meningkat 15% pada kuartal terakhir tahun 2023, didorong oleh permintaan global yang tinggi.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Pemerintah Kota Jakarta membangun tiga sekolah baru di daerah pinggiran untuk tahun ajaran 2024/2025.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
            {
                'text': "Indeks Harga Saham Gabungan ditutup menguat 0.5% pada perdagangan hari ini, didorong oleh kenaikan saham-saham sektor perbankan.",
                'label': 0,
                'source': 'sample',
                'url': ''
            },
        ]
        
        # Tambahkan lebih banyak contoh
        topics = [
            "ekonomi", "politik", "olahraga", "pendidikan", "kesehatan",
            "teknologi", "hiburan", "budaya", "lingkungan", "hukum"
        ]
        
        for topic in topics:
            valid_articles.append({
                'text': f"Konferensi tentang perkembangan {topic} akan diadakan di Jakarta minggu depan dengan menghadirkan para pakar dari berbagai negara.",
                'label': 0,
                'source': 'sample',
                'url': ''
            })
        
        return valid_articles
    
    def augment_data(self, articles, augmentation_factor=2):
        """Augment data untuk meningkatkan jumlah sampel"""
        augmented = []
        
        for article in articles:
            augmented.append(article)
            
            text = article['text']
            label = article['label']
            
            # Augmentasi: potong menjadi beberapa bagian (untuk artikel panjang)
            if len(text.split()) > 100:
                sentences = text.split('.')
                if len(sentences) > 3:
                    # Ambil bagian pertama
                    first_part = '.'.join(sentences[:len(sentences)//2]) + '.'
                    if len(first_part.split()) > 30:
                        augmented.append({
                            'text': first_part,
                            'label': label,
                            'source': article['source'] + '_augmented',
                            'url': article['url']
                        })
                    
                    # Ambil bagian kedua
                    second_part = '.'.join(sentences[len(sentences)//2:]) + '.'
                    if len(second_part.split()) > 30:
                        augmented.append({
                            'text': second_part,
                            'label': label,
                            'source': article['source'] + '_augmented',
                            'url': article['url']
                        })
        
        return augmented
    
    def create_dataset(self, save_path='hoax_dataset.csv'):
        """Buat dataset lengkap"""
        logger.info("Membuat dataset...")
        
        # Ambil data hoax dari TurnBackHoax
        logger.info("Mengambil data hoax dari TurnBackHoax...")
        hoax_articles = self.get_hoax_from_turnbackhoax(num_pages=5)
        
        # Generate berita valid
        logger.info("Membuat berita valid...")
        valid_articles = self.get_valid_news()
        
        # Gabungkan semua artikel
        all_articles = hoax_articles + valid_articles
        
        # Augment data
        logger.info("Melakukan augmentasi data...")
        all_articles = self.augment_data(all_articles)
        
        # Buat DataFrame
        df = pd.DataFrame(all_articles)
        
        # Hapus duplikat
        df = df.drop_duplicates(subset=['text'])
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Simpan
        df.to_csv(save_path, index=False, encoding='utf-8')
        
        # Statistik
        hoax_count = df['label'].sum()
        valid_count = len(df) - hoax_count
        
        logger.info(f"\nDataset created: {save_path}")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Hoax samples: {hoax_count}")
        logger.info(f"Valid samples: {valid_count}")
        
        return df

def main():
    """Generate dataset"""
    print("\n" + "="*70)
    print("HOAX DATASET GENERATOR")
    print("="*70)
    
    generator = DatasetGenerator()
    
    print("\n[1] Generating dataset...")
    dataset = generator.create_dataset('hoax_dataset.csv')
    
    print("\n[2] Dataset preview:")
    print(dataset.head())
    
    print("\n[3] Label distribution:")
    print(dataset['label'].value_counts())
    
    print("\n[4] Dataset saved to: hoax_dataset.csv")
    
    return dataset

if __name__ == "__main__":
    dataset = main()