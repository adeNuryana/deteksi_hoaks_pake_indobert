# detection_app/utils/scraper.py
import requests
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class NewsScraper:
    """Class untuk scraping berita dari berbagai website"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
        })
    
    def extract_from_url(self, url: str) -> dict:
        """
        Ekstrak konten berita dari URL
        
        Args:
            url (str): URL berita
        
        Returns:
            dict: Informasi yang diekstrak
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Hapus script dan style
            for element in soup(['script', 'style', 'iframe', 'nav', 'footer']):
                element.decompose()
            
            # Ekstrak berdasarkan domain
            domain = urlparse(url).netloc
            
            if 'kompas.com' in domain:
                return self._extract_kompas(soup, url)
            elif 'detik.com' in domain:
                return self._extract_detik(soup, url)
            elif 'antaranews.com' in domain:
                return self._extract_antara(soup, url)
            elif 'tribunnews.com' in domain:
                return self._extract_tribun(soup, url)
            elif 'cnnindonesia.com' in domain:
                return self._extract_cnn(soup, url)
            else:
                return self._extract_generic(soup, url)
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_kompas(self, soup, url):
        """Ekstrak dari Kompas.com"""
        title = soup.find('h1', class_='read__title')
        content = soup.find('div', class_='read__content')
        
        if title and content:
            return {
                'success': True,
                'title': title.text.strip(),
                'content': self._clean_text(content.text),
                'source': 'Kompas.com',
                'url': url,
                'domain': 'kompas.com'
            }
        return self._extract_generic(soup, url)
    
    def _extract_detik(self, soup, url):
        """Ekstrak dari Detik.com"""
        title = soup.find('h1', class_='detail__title')
        content = soup.find('div', class_='detail__body-text')
        
        if title and content:
            return {
                'success': True,
                'title': title.text.strip(),
                'content': self._clean_text(content.text),
                'source': 'Detik.com',
                'url': url,
                'domain': 'detik.com'
            }
        return self._extract_generic(soup, url)
    
    def _extract_antara(self, soup, url):
        """Ekstrak dari AntaraNews.com"""
        title = soup.find('h1', class_='post-title')
        content = soup.find('div', class_='post-content')
        
        if title and content:
            return {
                'success': True,
                'title': title.text.strip(),
                'content': self._clean_text(content.text),
                'source': 'Antara News',
                'url': url,
                'domain': 'antaranews.com'
            }
        return self._extract_generic(soup, url)
    
    def _extract_tribun(self, soup, url):
        """Ekstrak dari TribunNews.com"""
        title = soup.find('h1', id='arttitle')
        content = soup.find('div', id='article')
        
        if title and content:
            return {
                'success': True,
                'title': title.text.strip(),
                'content': self._clean_text(content.text),
                'source': 'Tribun News',
                'url': url,
                'domain': 'tribunnews.com'
            }
        return self._extract_generic(soup, url)
    
    def _extract_cnn(self, soup, url):
        """Ekstrak dari CNNIndonesia.com"""
        title = soup.find('h1', class_='title')
        content = soup.find('div', id='detikdetailtext')
        
        if title and content:
            return {
                'success': True,
                'title': title.text.strip(),
                'content': self._clean_text(content.text),
                'source': 'CNN Indonesia',
                'url': url,
                'domain': 'cnnindonesia.com'
            }
        return self._extract_generic(soup, url)
    
    def _extract_generic(self, soup, url):
        """Ekstrak generic dengan heuristic"""
        # Cari title
        title = soup.find('h1') or soup.find('title')
        title_text = title.text.strip() if title else ''
        
        # Cari konten utama
        article = soup.find('article') or soup.find('div', class_=re.compile(r'article|content|main'))
        
        if not article:
            # Coba cari div dengan banyak paragraf
            all_divs = soup.find_all('div')
            max_paragraphs = 0
            best_div = None
            
            for div in all_divs:
                paragraphs = div.find_all('p')
                if len(paragraphs) > max_paragraphs:
                    max_paragraphs = len(paragraphs)
                    best_div = div
            
            article = best_div
        
        content_text = ''
        if article:
            paragraphs = article.find_all('p')
            content_text = ' '.join([p.text.strip() for p in paragraphs])
        
        if not content_text:
            # Fallback: ambil semua teks
            content_text = soup.get_text()
        
        return {
            'success': True if content_text else False,
            'title': title_text[:500],
            'content': self._clean_text(content_text)[:5000],
            'source': 'Unknown',
            'url': url,
            'domain': urlparse(url).netloc
        }
    
    def _clean_text(self, text):
        """Bersihkan teks"""
        # Hapus extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Hapus karakter khusus
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Hapus multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

# Global instance
news_scraper = NewsScraper()