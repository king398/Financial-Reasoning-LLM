"""Module for scraping web pages and extracting article text using BeautifulSoup."""
import requests
from bs4 import BeautifulSoup
import transformers
import trafilatura


tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
def get_website(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    title = downloaded
    token_length = len(tokenizer(text).input_ids)
    print(f"Token length: {token_length}")
    with open("trial.txt","w") as f:
        f.write(text)
    return text
