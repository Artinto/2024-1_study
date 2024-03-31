def solution(today, terms, privacies):
    term_dict = {term_type: int(term_valid) for term_type, term_valid in map(str.split, terms)}
    today_year, today_month, today_date = map(int, today.split('.'))
    today_days = today_year * 12 * 28 + today_month * 28 + today_date # 일로 처리.
    answer = []
    for idx, privacy in enumerate(privacies, start=1):
        date, term_type = privacy.split()
        year, month, day = map(int, date.split('.')) # 년,월,일 분리 후 int처리
        expiry_days = year * 12 * 28 + month * 28 + day + term_dict[term_type] * 28 - 1 # 만료일 계산,
        
        if expiry_days < today_days:
            answer.append(idx)
    
    return answer