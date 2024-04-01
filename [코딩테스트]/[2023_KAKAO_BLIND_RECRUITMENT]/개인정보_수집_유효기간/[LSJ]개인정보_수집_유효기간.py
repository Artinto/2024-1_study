from datetime import datetime
from dateutil.relativedelta import relativedelta

def solution(today, terms, privacies):
    answer = []
    term_list = [item.split() for item in terms] # terms를 약관 종류와 약관별 보관기관 조합으로 리스트 수정 [A,6]
    term_dict = {} # 약관 별 보관 기간을 저장할 딕셔너리 선언
    distruction_dates = [] # 파기 날짜 저장할 리스트
    today_obj = datetime.strptime(today,"%Y.%m.%d") # 오늘 날짜를 문자열에서 객체로 변경
    for item in term_list: # 약관 관련 변수들을
        key = item[0]  # 약관별 
        value = int(item[1]) # 보관기관으로 엮어
        term_dict[key] = value # 딕셔너리에 저장
    for index, item in enumerate(privacies): # 인덱스와 item으로 분류
        date_str, char = item.split(' ') # 가입 날짜와 약관종류로 분리
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")  #  가입 날짜 문자열에서 객체화
        destruction_date_obj = date_obj + relativedelta(months=+term_dict[char]) # 가입 날짜 +약관별 보관기간 = 파기날짜
    
        if destruction_date_obj <= today_obj: # 파기날짜가 지났다면
            answer.append(index+1) # answer값 증가
    return answer
