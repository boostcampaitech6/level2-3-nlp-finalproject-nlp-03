# -*- coding: utf-8 -*-

import os, csv, requests 
from datetime import datetime
from bs4 import BeautifulSoup 
from urllib.request import urlopen
import pandas as pd

# 1. FAQ 크롤링
ymd = datetime.today().strftime("%y%m%d")
filename = f"ontong_faq_{ymd}.csv"
PATH = os.getcwd() # 원하는 폴더로 수정

f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["상담유형", "질문", "조회수", "source"]
writer.writerow(columns_title)

for i in range(1, 5): # 미리 사이트 방문하여 총 pageIndex 확인할 것
    url = f"https://www.youthcenter.go.kr/board/boardList.do?bbsNo=4&pageIndex={i}&pageUrl=board%2Fboard"

    a = urlopen(url)
    soup = BeautifulSoup(a.read(), 'html.parser')
    titles = soup.find_all('a', "ellipsis")
    type_nums = soup.find_all('td')

    url_list = []
    query = []
    for title in titles:
        query.append(title.text.strip('"').strip())
        url_list.append(url)
    
    type = []
    num = []
    for type_num in type_nums:
        if "상담유형" in type_num.text:
            type.append(type_num.text.replace("상담유형", '').strip())
        elif "조회" in type_num.text:
            num.append(int(type_num.text.replace("조회", '').strip()))

    for pair in zip(type, query, num, url_list):
        writer.writerow(list(pair))

f.close()   

print(">>> FAQ 크롤링 완료")

# 2. 게시판 상담 크롤링
filename = f"ontong_board_{ymd}.csv"

f = open(os.path.join(PATH, filename), "w", encoding="utf-8-sig", newline="")
writer = csv.writer(f)

columns_title = ["상담유형", "질문", "신청일자", "source"]
writer.writerow(columns_title)

for i in range(1, 219): # 먼저 사이트 들어가서 총 pageIndex 확인 후 수정할 것
    url = f"https://www.youthcenter.go.kr/jobConslt/jobConsltList.do?schAnswCslrId=&schConsKwrdCn=&schWrtMberId=&replyX=&replyN=&replyY=&schConsTycd=&schOtpbYn=&pageIndex={i}&pageUnit=10&pageSize=5"

    a = urlopen(url)
    soup = BeautifulSoup(a.read(), 'html.parser')
    titles = soup.find_all('div', "td-list-tit")
    type_nums = soup.find_all('td')

    url_list = []
    query = []
    for title in titles:
        if title.find("i"):
            query.append(title.text.replace("새 글", '').strip('"').strip())
        else:
            query.append(title.text.strip('"').strip())
        url_list.append(url)
    
    type = []
    num = []
    for type_num in type_nums:
        if "상담유형" in type_num.text:
            type.append(type_num.text.replace("상담유형", '').strip())
        elif "신청일자" in type_num.text:
            num.append(type_num.text.replace("신청일자", '').strip())

    for pair in zip(type, query, num, url_list):
        writer.writerow(list(pair))

f.close()   

df = pd.read_csv(os.path.join(PATH, f"ontong_board_{ymd}.csv")) 

df['신청일자'] = pd.to_datetime(df['신청일자'])

df_org = df.drop_duplicates(subset='질문', keep='first', inplace=False, ignore_index=False) # 중복 행 삭제

filename = f"ontong_board_org_{ymd}.csv"
df_org.to_csv(os.path.join(PATH, filename), index=None, encoding="utf-8-sig")

print(">>> 게시판 상담 크롤링 완료")