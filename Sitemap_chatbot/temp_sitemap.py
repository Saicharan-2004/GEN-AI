import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time

def generate_sitemap(base_url, output_file="sitemap_final.xml", max_pages=500):
    visited = set()
    to_visit = set([base_url])

    def is_valid(url):
        parsed = urlparse(url)
        return parsed.netloc == urlparse(base_url).netloc and parsed.scheme in ("http", "https")

    print(f"🚀 Starting sitemap generation for: {base_url}")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue

            visited.add(url)
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                new_url = urljoin(url, link["href"])
                if is_valid(new_url) and new_url not in visited:
                    to_visit.add(new_url)

            print(f"✅ Crawled: {url} ({len(visited)}/{max_pages})")

        except Exception as e:
            print(f"❌ Error visiting {url}: {str(e)}")
        
        time.sleep(0.5)  # Be gentle to servers!

    # Write to sitemap.xml
    print(f"📝 Writing {len(visited)} URLs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for url in visited:
            f.write(f"  <url>\n    <loc>{url}</loc>\n  </url>\n")
        f.write("</urlset>")

    print(f"🎉 Sitemap generated successfully: {output_file}")

# Example usage:
generate_sitemap("https://chaidocs.vercel.app/youtube/getting-started")
