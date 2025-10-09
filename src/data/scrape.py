import requests
from bs4 import BeautifulSoup

def get_website(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()  # Ensure we got a successful response

    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("div", class_="article-content") or soup

    # Collect paragraph + list item text in order
    lines = []
    for elem in article.find_all(["p", "li"]):
        if elem.name == "li":
            lines.append("- " + elem.get_text(strip=True))
        else:
            lines.append(elem.get_text(strip=True))
    text = "\n".join(lines)

    print(text)
