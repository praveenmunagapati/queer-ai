# Queer AI


## Scraping Erotica
`scrape.py` is a python2 script using `BeautifulSoup` to scrape user-defined sections of literotica.com.

### Usage:
[Set up `pipenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if you don't already have it.

Feel free to edit `GENRES` and `PAGES_PER_GENRE`. `GENRES` is a list of url slugs for sections you want to include.

Run:
```
pipenv run python scrape.py  
```
