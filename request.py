import requests

url = 'http://localhost:5000/predict_api'
#r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

r = requests.post(url,json={'q1':2,'q2':3, 'q3':3, 'q4':4, 'q5':3, 'q6':4, 'q7':2, 'q8':4, 'q9':3, 'q10':5, 'q11':2, 'q12':3, 'q13':4, 'q14':3, 'q15':4, 'q16':2, 'q17':2, 'q18':5, 'q19':2, 'q20':3, 'q21':5, 'q22':2, 'q23':3, 'q24':4, 'q25':2, 'q26':5, 'q27':2, 'q28':4, 'q29':2, 'q30':4, 'q31':2, 'q32':5, 'q33':2, 'q34':5, 'q35':2, 'q36':1, 'q37':4, 'q38':2, 'q39':2, 'q40':4, 'q41':3, 'q42':4, 'q43':2, 'q44':4, 'q45':2, 'q46':4, 'q47':1, 'q48':4, 'q49':3, 'q50':3})
print(r.json())