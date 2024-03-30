def date2day(date):
    y,m,d = map(int,date.split('.'))
    return (y*28*12 + m*28 + d)
def solution(today, terms, privacies):
    today = date2day(today)
    terms = {t[0] : int(t[2:])*28 for t in terms}
    return [idx for idx,p in enumerate(privacies,start=1) if today-date2day(p[:10]) >= terms[p[-1]]]
