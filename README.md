# est_wassup_01
ESTSoft Wassup 1기 첫번째 프로젝트

팀 Popcorn [김종성, 김범찬, 박지연, 송승민, 이유성]

# <<Data Source>>
대구시 교통사고 예측 경진대회 https://dacon.io/competitions/official/236193/data

# <<Target>>
ECLO(Equivalent Casualty Loss Only, 인명피해 심각도)
= 사망자수*10 + 중상자수*5 + 경상자수*3 + 부상자수*1
사망자수, 중상자수, 경상자수, 부상자수를 각각 예측하는 multi-task learning model 설계

# <<Terminal>>
$ python train.py
기본적으로 config폴더의 config.py이 실행되도록 설정되어있음(MNN)
각종 경로와 hyper-prameter 등 설정 가능

$ python train.py -mode True
multi mode를 작동시키는 명령어(multi_config.py 파일들을 실행)
여러 config file을 만들어 놓고, 순차적으로 작동하도록 사용 가능

 
# <<Preprocess Summary>>
Select Numeric Data
Label Encoding
Merge External Data

# <<Metics>>
RMSE
R2 SCORE


# <<Pipelines>>
Directory
![Alt text](directory_img.png)