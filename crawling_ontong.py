import os, re, requests
import pandas as pd
import numpy as np
import copy

from bs4 import BeautifulSoup 
from urllib.request import urlopen

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

ymd = datetime.today().strftime("%y%m%d")
OUTPUT_DIR = os.path.join(os.getcwd(), "data")
CARD_DOC_PATH = os.path.join(OUTPUT_DIR, f"card_{ymd}.csv")
DB_DOC_PATH = os.path.join(OUTPUT_DIR, f"db_{ymd}.csv")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def get_ontong() -> pd.DataFrame:
    df = pd.DataFrame(columns = ['PolicyName', 'D-day', 'OrgName', 'PolicyType', 'Progress'])

    pol_names = []
    ddays = []
    org_names = []
    cates = []
    ongoings = []

    for i in range(1, 44):
        url = f"https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifList.do?plcyCmprInfo=&pageIndex={i}&plcyCmprInfo=&trgtJynEmp=&trgtJynEmp="

        a = requests.get(url)
        soup = BeautifulSoup(a.text, 'html.parser')
        for j in range(1, 13):
            answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.tit-wrap")
            pol_name = ""
            for a in answers:
                pol_name += a.get_text().strip()
            pol_names.append(pol_name)

            answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.organ-name > p.dday")
            dday = ""
            for a in answers:
                dday += a.get_text().strip().replace('D-day ', '').replace('일', '')
            ddays.append(dday)

            answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.organ-name > p:nth-child(1)")
            
            org_name = ""
            for a in answers:
                org_name += a.get_text().strip()
            org_names.append(org_name)

            answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.badge > span.cate")
            cate = ""
            for a in answers:
                cate += a.get_text().strip()
            cates.append(cate)
            
            answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.badge > span.label.green")
            if not answers:
                answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.badge > span.label.purple")
                if not answers:
                    answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.badge > span.label.red")
                    if not answers:
                        answers = soup.select(f"#srchFrm > div.sch-result-wrap > div.result-list-wrap > ul > li:nth-child({j}) > div.result-card-box > div.badge > span.label.blue")

            ongoing = ""
            for a in answers:
                ongoing += a.get_text().strip()
            ongoings.append(ongoing)

    df["PolicyName"] = pol_names
    df["D-day"] = ddays
    df["D-day"] = pd.to_numeric(df["D-day"], errors='coerce').fillna(0).astype(int) # int로 설정해 둬야 추후 마감일 기준 D-day 계산이 가능함
    df["OrgName"] = org_names
    df["PolicyType"] = cates
    df["Progress"] = ongoings

    df = df.drop_duplicates()

    ### ONTONG
    key = 'f93ad94f49a4829e7739cc9f' ### 여기에 신청한 본인의 key를 넣어야 합니다!
    page = 0
    outputNum = '100' # 출력건수, 기본 10 최대 100

    contents = ''
    while True:
        queryParams = 'pageIndex=' + str(page) + '&display=' + outputNum + '&openApiVlak=' + key
        url = 'https://www.youthcenter.go.kr/opi/youthPlcyList.do?'+ queryParams

        # url 불러오기
        response = requests.get(url)
        
        if len(response.text) == 130:
            break # 130으로 나오는 경우 해당 페이지에 출력할 정책이 없는 경우임, 즉 130 이전 page까지만 포함시키면 됨.

        print(len(response.text))

        #데이터 값 출력해보기
        contents += response.text
        page += 1

    soup = BeautifulSoup(contents, 'html.parser')

    YouthPolicyInfo = {}
    # attr 이름 모두 소문자로
    attr_to_find_list=['polybizsjnm', 'polyItcnCn', 'sporCn', 
                    'ageInfo', 'majrRqisCn', 'prcpCn', 'aditRscn', 'prcpLmttTrgtCn',
                    'rqutProcCn', 'pstnPaprCn', 'rqutUrla', 'rfcSiteUrla1', 'rfcSiteUrla2'] # 모두 소문자여야 함.

    for each_attr in attr_to_find_list:
        finded_attr=soup.find_all(each_attr.lower())
        YouthPolicyInfo[each_attr]=[x.text for x in finded_attr]

    df_ontong = pd.DataFrame(YouthPolicyInfo)
    alt_colnames = ['PolicyName', 'Age']

    df_ontong.rename(columns={'polybizsjnm': '정책명', 'polyitcncn': '정책소개', 'sporcn': '지원내용', 
                    'ageinfo': '연령정보', 'majrrqiscn': '전공요건내용', 'prcpcn': '거주지및소득조건내용', 'aditrscn': '추가단서사항내용', 'prcpLmtttrgtcn':'참여제한대상내용',
                    'rqutproccn':'신청절차내용', 'pstnpaprcn':'제출서류내용', 'rqutUrla':'신청사이트주소', 'rfcsiteurla1':'참고사이트1', 'rfcsiteurla2':'참고사이트2'})
        
        # 숫자를 찾아서 MinAge와 MaxAge를 할당하는 함수
    def assign_ages(row):
        # 현재 열에서 모든 숫자 찾기
        numbers = re.findall(r'\d+', row['연령정보'])
        
        # 숫자의 개수에 따라 조건 적용
        if len(numbers) == 2:
            return pd.Series([int(numbers[0]), int(numbers[1])])
        elif len(numbers) == 1:
            return pd.Series([int(numbers[0]), None])
        else:
            return pd.Series([-1, 200])

    # 새 열에 함수 적용
    df_ontong[['MinAge', 'MaxAge']] = df_ontong.apply(assign_ages, axis=1)

    df_ontong = df_ontong.drop_duplicates()

    merged_df = pd.merge(df, df_ontong, on='PolicyName', how='inner')
    merged_df = merged_df.drop(['연령정보', '정책소개', '지원내용', '전공요건내용', '거주지및소득조건내용', '추가단서사항내용', '참여제한대상내용', 
                                '신청절차내용', '제출서류내용', '신청사이트주소', '참고사이트1', '참고사이트2'], axis=1)
    merged_df.loc[merged_df['Progress'] == '상시', 'D-day'] = '∞'
    merged_df.loc[merged_df['Progress'] == '진행 예정', 'D-day'] = '?'
    merged_df = merged_df[merged_df['Progress'] != '신청 마감']
    merged_df.rename(columns={'정책명':'PolicyName'})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_csv(CARD_DOC_PATH, index=False)
    
    # for chroma db uploading data
    procedure = ['정책명', '신청절차내용', '제출서류내용', '신청사이트주소', '참고사이트1', '참고사이트2']
    qualification = ['정책명', '연령정보', '지원내용', '전공요건내용', '거주지및소득조건내용', '추가단서사항내용', '참여제한대상내용']
    df_db = copy.deepcopy(df_ontong)

    df_db['procedure'] = df[procedure].apply(lambda row : 'ᴥ'.join(row.values.astype(str)), axis=1)
    df_db['qualification'] = df[qualification].apply(lambda row : 'ᴥ'.join(row.values.astype(str)), axis=1)

    df_db.drop(['연령정보', '정책소개', '지원내용', '전공요건내용', '거주지및소득조건내용', '추가단서사항내용', '참여제한대상내용', 
                                '신청절차내용', '제출서류내용', '신청사이트주소', '참고사이트1', '참고사이트2'], axis=1)
    df_db.rename(columns={'정책명':'PolicyName'})
    df.to_csv(DB_DOC_PATH, index=False)




with DAG(
        dag_id='crawling_ontong',
        default_args=default_args,
        schedule_interval="0 0 * * *",  
        #schedule_interval="@once",
        catchup=False,
        tags=['assignment'],
) as dag:
    get_ontong_task = PythonOperator(
        task_id="get_ontong_task",
        python_callable=get_ontong,
    )

    get_ontong_task

