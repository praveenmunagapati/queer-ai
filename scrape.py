import urllib2
import csv
import os.path
from bs4 import BeautifulSoup

PARSER = "html.parser"
DATA_FILE = "literotica.csv"
URL_TEMPLATE = "https://www.literotica.com/c/%s/%d-page"
GENRES = ["science-fiction-fantasy", "bdsm-stories", "erotic-couplings", "celebrity-stories", "gay-sex-stories", "lesbian-sex-stories", "transsexuals-crossdressers", "non-human-stories"]
PAGES_PER_GENRE = 2
data = {}

if os.path.isfile(DATA_FILE):
    with open(DATA_FILE, 'r+') as data_file:
        for line in csv.DictReader(data_file):
            data[line['url']] = line

def scrape_story(url):
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page, PARSER)
    story = soup.select_one(".b-story-body-x p").get_text().encode('utf-8').strip()
    data[url] = { "url": url, "story": story }

def scrape_list_page(page):
    soup = BeautifulSoup(page, PARSER)
    links = soup.select(".b-sl-item-r h3 a")

    urls = map(lambda l: l.get("href"), links)
    for u in urls:
        print "scraping %s" % u
        try:
            scrape_story(u)
        except urllib2.HTTPError as e:
            print(e)

def scrape_genre(genre_slug, page_limit = PAGES_PER_GENRE):
    count = 0
    while count < page_limit:
        count += 1
        url = URL_TEMPLATE % (genre_slug, count)
        print "scraping %s" % url
        try:
            page = urllib2.urlopen(url).read()
            scrape_list_page(page)
        except urllib2.HTTPError as e:
            print(e)
            break;

for g in GENRES:
    scrape_genre(g)

with open(DATA_FILE, 'w+') as data_file:
    writer = csv.DictWriter(data_file, fieldnames=["url", "story"])
    writer.writeheader()
    for row in data.itervalues():
        writer.writerow(row)
