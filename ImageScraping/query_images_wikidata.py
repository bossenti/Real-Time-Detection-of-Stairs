import requests
from urllib.request import urlretrieve

"""
short script to scrape images of stairs from wikidata SPARQL endpoint
"""

# define SPARQL endpoint
url = 'https://query.wikidata.org/sparql'

# SPARQL query for stair images
query = "SELECT distinct ?Image WHERE {" \
        "  ?stair wdt:P31 wd:Q12511." \
        "  ?stair wdt:P18 ?Image." \
        "}"

# get results and format as JSON
r = requests.get(url, params={'format': 'json', 'query': query})
data = r.json()

# download images
for idx, row in enumerate(data['results']['bindings']):
    urlretrieve(row["Image"]["value"], "img/img_" + str(idx) + ".jpg")
