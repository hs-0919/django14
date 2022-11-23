from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
# Create your views here.


def mainFunc(request):
    df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/patient.csv")
    print(df)
    table = df.head(3)
    df_x = df.drop(columns = ['STA','ID'])
    df_y = df['STA']
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.25, random_state=12)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (150, 10) (50, 10) (150, 1) (50, 1)
    
    model = RandomForestClassifier(n_estimators=200, criterion='entropy')
    model.fit(train_x, train_y)
    pickle.dump(model, open('patient_model.sav', 'wb'))    # 모델 저장
    
    pred = model.predict(test_x)
    print('예측값 :', pred[:5])
    print('실제값 :', np.array(test_y[:5]))
    
    # 정확도
    acc = accuracy_score(test_y, pred) # 0.84

    return render(request, 'main.html', {'table':table.to_html(), 'acc':acc})

def listFunc(request):
    
    # mymodel = joblib.load('pima_model_sav')
    model = pickle.load(open('patient_model.sav', 'rb'))
    
    age = request.POST.get('age')
    sex = request.POST.get('sex')
    race = request.POST.get('race')
    ser = request.POST.get('ser')
    can = request.POST.get('can')
    crn = request.POST.get('crn')
    inf = request.POST.get('inf')
    cpr = request.POST.get('cpr')
    hra = request.POST.get('hra')
    
    data = { age:[age],
            sex:[sex],
            race:[race],
            ser:[ser],
            can:[can],
            inf:[inf],
            cpr:[cpr],
            hra:[hra]
        }
    
    test_x = pd.DataFrame(data, columns=[age, sex, race, ser, can, crn, inf, cpr, hra])
    print(test_x)
    print(model)
    pred = model.predict(test_x)
    print(pred)
    
    return render(request, 'list.html', {'pred':pred[0]})

def showFunc(request):
    return render(request, 'show.html')