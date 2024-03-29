from datetime import datetime
from dateutil.relativedelta import relativedelta

def solution(today, terms, privacies):
    answer = []
    term_list = [item.split() for item in terms]
    term_dict = {}
    distruction_dates = []
    today_obj = datetime.strptime(today,"%Y.%m.%d")
    for item in term_list:
        key = item[0]
        value = int(item[1])
        term_dict[key] = value
    for index, item in enumerate(privacies):
        date_str, char = item.split(' ')
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        destruction_date_obj = date_obj + relativedelta(months=+term_dict[char])
    
        if destruction_date_obj <= today_obj:
            answer.append(index+1)
    return answer
