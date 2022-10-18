import requests

def download_file(url, filename):
  response = requests.get(url)
  open(filename, "wb").write(response.content)