# -*- coding: utf-8 -*-
import os, re, csv, requests 
from datetime import datetime
from bs4 import BeautifulSoup 
from urllib.request import urlopen
import pandas as pd

## 1. 국민취업지원제도
# 1-1. 국민취업지원제도 사이트 QnA 크롤링
ymd = datetime.today().strftime("%y%m%d")
PATH = os.getcwd() # 원하는 폴더로 수정

filename = "kua_faq.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변", "source"]
writer.writerow(columns_title)

for i in range(1,4):
    url = f"https://www.kua.go.kr/uapae010/selectQstn.do?pageIndex={i}&pageUnit=10&srchSecd=A&srchTy=&srchKwrd="

    a = requests.get(url)
    soup = BeautifulSoup(a.text, 'html.parser')
    questions = soup.find_all('button')
    question_list = []
    url_list = []

    for q in questions:
        if 'Q.' in q.get_text():
            question_list.append(q.get_text().replace('  ', '').replace('Q.', '').replace(f'\n', '').replace(f'\t', '').strip(' '))
        url_list.append(url)
    answers = soup.find_all('div')
    answer_list = []

    for a in answers:
        if ('A.' in a.get_text()) & ('Q.' not in a.get_text()):
            answer_list.append(a.get_text().replace('A.', '').replace(f'\n', '').replace(f'\t', '').replace(f'\xa0', '').strip(' '))

    for pair in zip(question_list, answer_list, url_list):
        writer.writerow(list(pair))

f.close()

df1 = pd.read_csv(os.path.join(PATH, "kua_faq.csv"))
df_ontong = pd.read_csv(os.path.join(PATH, f"ontong_board_org_{ymd}.csv")) # web-crawling.py에서 생성한 csv
df_ontong_faq = pd.read_csv(os.path.join(PATH, f"ontong_faq_{ymd}.csv")) # web-crawling.py에서 생성한 csv

# 1-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('국민취업지원제도|국취제')]

for index, row in ontong_q.iterrows():
    df1.loc[len(df1)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('국민취업지원제도|국취제')]

for index, row in ontong_faq_q.iterrows():
    df1.loc[len(df1)] = [row['질문'], '', row['source']]

# 1-3. 고용노동부
kua_list = ['1000000702', '1000000703', '1000000704', '1000000705', '1000000706', 
            '1000000707', '1000000708', '1000000709', '1000000710', '1000000711', 
            '1000000712', '1000000713', '1000000714', '1000000715', '1000000716', 
            '1000000717', '1000000718', '1000000719', '1000000720', '1000000721', 
            '1000000722', '1000000723', '1000000724', '1000000725', '1000000726', 
            '1000000727', '1000000728', '1000000729', '1000000730', '1000000731', 
            '1000000732', '1000000978', '1000000979', '1000001040', '1000001041', 
            '1000001042', '1000001057', '1000001058', '1000001099', '1000001100', 
            '1000001115', '1000001233', '1000001234', '1000001235', '1000001236', 
            '1000001689', '1000001690', '1000001691', '1000001692', '1000001693', 
            '1000001839', '1000001840', '1000001841', '1000001842', '1000001843', 
            '1000001844', '1000001845', '1000001846', '1000001847', '1000001848'] 
# 국민취업지원 제도 url faq_idx 수동 수집
question_list = []
answer_list = []
url_list = []
for i in kua_list:
    url = f"https://1350.moel.go.kr/home/hp/data/faqView.do?faq_idx={i}"

    a = requests.get(url)
    soup = BeautifulSoup(a.text, 'html.parser')

    questions = soup.select("#Content > div > div > table > tbody > tr.p-table__subject > td > span")
    answers = soup.select("#Content > div > div > table > tbody > tr:nth-child(4) > td")

    question = ''
    answer = ''
    for q in questions:
        question += q.get_text().strip()
    for a in answers:
        answer += a.get_text().replace(f'\n', '').replace(f'\xa0', '').strip()

    question_list.append(question)
    answer_list.append(answer)
    url_list.append(url)

for pair in zip(question_list, answer_list, url_list):
    df1.loc[len(df1)] = list(pair)

df1['정책'] = '국민취업지원제도'
df1.to_csv(os.path.join(PATH, "kua_faq.csv"), encoding="utf-8-sig", index=True)
print('>>> 국민취업지원제도 완료')

## 2. 국민내일배움카드
# 2-1. 고용노동부
filename = "card_faq.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변", "source"]
writer.writerow(columns_title)

card_list = ['1000000691', '1000000692', '1000000693', '1000000694', '1000000695',
            '1000000696', '1000000697', '1000000698', '1000000699', '1000000700',
            '1000000701', '1000000963', '1000000964', '1000000965', '1000000966',
            '1000000967', '1000000968', '1000000969', '1000000970', '1000000971',
            '1000000972', '1000000973', '1000000974', '1000000975', '1000000977',
            '1000000980'] 
# 국민내일배움카드 url faq_idx 수동 수집
question_list = []
answer_list = []
url_list = []
for i in card_list:
    url = f"https://1350.moel.go.kr/home/hp/data/faqView.do?faq_idx={i}"

    a = requests.get(url)
    soup = BeautifulSoup(a.text, 'html.parser')

    questions = soup.select("#Content > div > div > table > tbody > tr.p-table__subject > td > span")
    answers = soup.select("#Content > div > div > table > tbody > tr:nth-child(4) > td")

    question = ''
    answer = ''
    for q in questions:
        question += q.get_text().strip()
    for a in answers:
        answer += a.get_text().replace(f'\n', '').replace(f'\xa0', '').strip()

    question_list.append(question)
    answer_list.append(answer)
    url_list.append(url)

for pair in zip(question_list, answer_list, url_list):
    writer.writerow(list(pair))

f.close()

df2 = pd.read_csv(os.path.join(PATH, "card_faq.csv"))

# 2-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('국민내일배움카드|내일배움카드')]

for index, row in ontong_q.iterrows():
    df2.loc[len(df2)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('국민내일배움카드|내일배움카드')]

for index, row in ontong_faq_q.iterrows():
    df2.loc[len(df2)] = [row['질문'], '', row['source']]

df2['정책'] = '국민내일배움카드'
df2.to_csv(os.path.join(PATH, "card_faq.csv"), encoding="utf-8-sig", index=True)

print('>>> 국민내일배움카드 완료')

## 3. 청년 주택드림 청약통장 -> 수동
filename = "bankbook_faq.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변"]
writer.writerow(columns_title)

question_list = ['이미 주택청약종합저축 또는 청년우대형 주택청약종합저축에 가입한 사람은 어떻게 가입하나요?',
                 '가입 요건 충족 여부는 어떤 서류로 확인을 하나요?',
                 '인터넷 뱅킹이나 모바일로 가입할 수 있나요?',
                 '군 복무 중인 현역 장병은 어떻게 가입하나요?',
                 '소득공제 혜택은 기존 주택청약종합저축과 동일하나요?',
                 '청년주택드림청약통장에 가입만 하면 이자소득 비과세 혜택을 받을 수 있나요?',
                 '청년주택드림청약통장 출시 전 사전청약에 당첨되고 출시 후 본 청약 하는 경우에도 청년주택드림대출을 받을 수 있나요?',
                 '청년주택드림대출은 언제 신청할 수 있나요?',
                 '청년도약계좌 또는 청년희망적금 만기 수령액을 청년주택드림청약통장 일시금으로 납부하려는데 어떻게 해야 하나요?']
answer_list = ['기존 청년우대형 주택청약종합저축에 가입한 사람은 별도 신청 없이 ‘청년주택드림청약통장’으로 자동 전환 됩니다. 기존 주택청약종합저축 가입자가 ‘청년주택드림청약통장’ 가입 조건을 갖추었다면, 은행을 방문하여 전환 신청이 가능합니다. 다만 기존 ‘주택청약종합저축’이 당첨 계좌인 경우 전환 대상에서 제외합니다.',
               '일반 주택청약종합저축은 누구나 가입 가능한 반면 청년주택드림청약통장은 일정 요건(나이, 소득, 무주택 등)을 충족 시 가입이 가능하여 이에 대한 확인이 필요합니다. ○ (무주택 여부) 가입 시 무주택확약서 등으로 확인하고, 해지 시 지방세 세목별 과세증명서 및 주택소유시스템 등으로 가입 기간에 대한 무주택 여부 확인 ○ (연령) 신분증 등으로 확인 ○ (소득) 소득확인증명서(청년우대형주택청약종합저축 가입 및 과세특레 신청용) 및 소득원천징수 영수증 등으로 직전년도 소득을 확인',
               '상반기 내 인터넷 뱅킹 또는 모바일로 가입할 수 있도록 시스템을 구축할 예정입니다.',
               '나라사랑포털 누리집에서 발급하는 ‘가입자격확인서’ 또는 소속부대에서 발급하는 ‘군복무 확인서’를 지참해 인근 은행을 방문해 가입 신청할 수 있습니다. ☞ (가입대상) 현역병, 상근예비역, 의무경찰, 해양의무경찰, 의무소방원, 사회복무요원, 대체복무요원',
               '청년주택드림청약통장은 주택청약종합저축의 일종으로 재형 기능을 강화하기 위해 우대금리와 이자소득 비과세 혜택을 제공하는 상품으로 주택청약종합저축의 하위 상품이라 할 수 있습니다. 따라서, 현재 주택청약종합저축에서 제공하는 있는 소득공제 조건(조세특례제한법 제87조)을 그대로 적용받게 되며, 연 소득 7천만 원 이하 무주택세대주로서 무주택확인서를 제출하는 경우 연간 납입액 300만 원 한도로 40%까지 소득공제가 가능합니다.',
               '비과세 혜택은 조세특례제한법 제87조에 따른 아래의 요건을 충족한 가입자가 별도의 서류(이자소득 비과세용 무주택확인서 등)를 은행에 가입 후 2년 내 제출하면 이자소득 비과세 혜택을 받을 수 있습니다. ○ (소득) 근로소득 3천6백만원 또는 사업소득 2천6백만원 이하 ○ (무주택) 가입 시 무주택 세대의 세대주 ☞ 청년주택드림통장에 가입한 자 중 소득초과자, 세대원이 주택을 소유한 경우, 직전년도 신고소득이 없는 경우 등은 이자소득 비과세 혜택에서는 제외될 수 있음',
               '사전청약 당첨자 중 기존 청년우대형청약저축 또는 일반 청약저축가입자로서 전환 가입하거나, 청년주택드림청약통장 가입 자격을 충족한 경우 대출 신청이 가능합니다.',
               '청약에 당첨된 주택의 소유권 이전 등기접수일로부터 3개월 이내에 신청 할 수 있습니다. 단, 청년주택드림청약저축 신규 또는 전환 가입 후 가입 기간이 1년 이상이면서, 1천만 원 이상 납부 실적이 있는 경우에 한합니다.',
               '청년도약계좌 또는 청년희망적금 만기 해지에 대한 증빙(계좌해지 은행에서 발급)을 지참해 해지일로부터 3개월 이내 인근 은행을 방문해 일시금으로 납부하면 됩니다.']

for pair in zip(question_list, answer_list):
    writer.writerow(list(pair))

f.close()

df3 = pd.read_csv(os.path.join(PATH, "bankbook_faq.csv"))

df3['source'] = 'https://blog.naver.com/mltmkr/223359281452'

# 3-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('청년 주택드림 청약통장|주택드림 청약통장|주택드림')]

for index, row in ontong_q.iterrows():
    df3.loc[len(df3)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('청년 주택드림 청약통장|주택드림 청약통장|주택드림')]

for index, row in ontong_faq_q.iterrows():
    df3.loc[len(df3)] = [row['질문'], '', row['source']]

df3['정책'] = '청년 주택드림 청약통장'
df3.to_csv(os.path.join(PATH, "bankbook_faq.csv"), encoding="utf-8-sig", index=True)

print('>>> 청년 주택드림 청약통장 완료')

## 4. 청년전용 버팀목전세자금
filename = "loan_faq.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변", "source"]
writer.writerow(columns_title)

for i in range(1, 5):
    url = f"https://nhuf.molit.go.kr/FP/FP05/FP0502/FP05020303.jsp?gotoPage={i}"

    a = requests.get(url)
    soup = BeautifulSoup(a.text, 'html.parser')

    question_list = []
    answer_list = []
    url_list = []

    questions = soup.select("#contArea > div.contents > div.boardTb > ul > li > div.q > div.txt")

    for q in questions:
        question_list.append(q.get_text().rstrip('열기'))
        url_list.append(url)

    answers = soup.select("#contArea > div.contents > div.boardTb > ul > li > div.a > div.txt")

    for a in answers:
        answer_list.append(a.get_text().replace(f'\n', '').replace(f'\xa0', '').replace(f'\r', ''))

    for pair in zip(question_list, answer_list, url_list):
        writer.writerow(list(pair))

f.close()

df4 = pd.read_csv(os.path.join(PATH, "loan_faq.csv"))

# 4-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('버팀목전세자금대출|버팀목대출|버팀목')]

for index, row in ontong_q.iterrows():
    df4.loc[len(df4)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('버팀목전세자금대출|버팀목대출|버팀목')]

for index, row in ontong_faq_q.iterrows():
    df4.loc[len(df4)] = [row['질문'], '', row['source']]

df4['정책'] = '청년전용 버팀목전세자금'
df4.to_csv(os.path.join(PATH, "loan_faq.csv"), encoding="utf-8-sig", index=True)

print('>>> 청년전용 버팀목전세자금 완료')

## 5. 청년도약계좌
filename = "account_faq.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변", "source"]
writer.writerow(columns_title)

question_list = []
answer_list = []
url_list = []
for i in range(1, 3):
    url = f"https://kinfa.or.kr/customerService/boardFAQ.do?currentPageNo={i}"

    a = requests.get(url)
    soup = BeautifulSoup(a.text, 'html.parser')

    for i in range(1, 16):
        questions = soup.select(f"#faq_con > ol > li:nth-child({i}) > div.title.row > a")
        for q in questions:
            question_list.append(q.get_text())
            url_list.append(url)

    for i in range(1, 16):
        answers = soup.select(f"#faq_con > ol > li:nth-child({i}) > div.con_inner")
        for a in answers:
            answer_list.append(a.get_text().replace(f'\n', ''))

for pair in zip(question_list, answer_list):
    writer.writerow(list(pair))

f.close()

df5 = pd.read_csv(os.path.join(PATH, "account_faq.csv"))

df5['source'] = 'https://blog.naver.com/mltmkr/223359281452'

# 5-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('청년도약계좌|도약계좌')]

for index, row in ontong_q.iterrows():
    df5.loc[len(df5)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('청년도약계좌|도약계좌')]

for index, row in ontong_faq_q.iterrows():
    df5.loc[len(df5)] = [row['질문'], '', row['source']]

df5['정책'] = '청년도약계좌'
df5.to_csv(os.path.join(PATH, "account_faq.csv"), encoding="utf-8-sig", index=True)

print('>>> 청년도약계좌 완료')

## 6. 기후동행카드
filename = "climate_card.csv"
f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["질문", "답변"]
writer.writerow(columns_title)

url = f"https://news.seoul.go.kr/traffic/archives/511551"

a = requests.get(url)
soup = BeautifulSoup(a.text, 'html.parser')

questions = soup.select("#post_content")

for q in questions:
    lines = q.get_text().strip().split('\n')

question_list = []
answer_list = []
temp_answers = ""

for line in lines:
    if line.startswith("Q"):
        # 새로운 질문이 시작되면 이전 답변을 answer_list에 추가
        if temp_answers:  # temp_answers가 비어있지 않다면
            answer_list.append(temp_answers.strip())
            temp_answers = ""  # 답변 임시 저장소 초기화
        question_list.append(re.sub(r'Q\d+\.\s', '', line))
    elif line.startswith("○"):
        # 답변을 temp_answers에 추가 (여러 줄의 답변을 하나의 문자열로 합치기)
        temp_answers += " " + line

# 마지막 답변 추가 (문서 끝에 도달했을 때)
if temp_answers:
    answer_list.append(temp_answers.strip())

for pair in zip(question_list, answer_list):
    writer.writerow(list(pair))

f.close()

df6 = pd.read_csv(os.path.join(PATH, "climate_card.csv"))

df6['source'] = url

# 6-2. 온통청년
ontong_q = df_ontong[df_ontong['질문'].str.contains('기후동행카드')]

for index, row in ontong_q.iterrows():
    df6.loc[len(df6)] = [row['질문'], '', row['source']]

ontong_faq_q = df_ontong_faq[df_ontong_faq['질문'].str.contains('기후동행카드')]

for index, row in ontong_faq_q.iterrows():
    df6.loc[len(df6)] = [row['질문'], '', row['source']]

df6['정책'] = '기후동행카드'
df6.to_csv(os.path.join(PATH, "climate_card.csv"), encoding="utf-8-sig", index=True)

print('>>> 기후동행카드 완료')

# df 통합
df_all = pd.concat([df1, df2, df3, df4, df5, df6])
df_all.to_csv(os.path.join(PATH, "faq.csv"), encoding="utf-8-sig", index=True)