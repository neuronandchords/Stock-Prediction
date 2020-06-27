import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'NO OF DAYS':5})

print(r.json())