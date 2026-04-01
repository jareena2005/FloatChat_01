import requests
qs=['compare temperature and salinity','salinity heatmap','show pacific floats']
for q in qs:
    r=requests.post('http://127.0.0.1:8080/get', json={'msg':q}, timeout=10)
    print('---',q,r.status_code)
    print(r.json())
