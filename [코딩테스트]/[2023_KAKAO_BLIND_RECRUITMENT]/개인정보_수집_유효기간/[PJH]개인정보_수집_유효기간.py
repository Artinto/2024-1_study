from datetime import datetime
from dateutil.relativedelta import relativedelta

def solution(today, terms, privacies):
    answer = []
    today_date = datetime.strptime(today, '%Y.%m.%d')
    
    term_dict = {term.split()[0]: int(term.split()[1]) for term in terms}

    for i, privacy in enumerate(privacies):
        start_date_str, term_type = privacy.split()
        start_date = datetime.strptime(start_date_str, '%Y.%m.%d')

        expiry_date = start_date + relativedelta(months=+term_dict[term_type])
        
        if today_date >= expiry_date:
            answer.append(i + 1)
    
    return answer
