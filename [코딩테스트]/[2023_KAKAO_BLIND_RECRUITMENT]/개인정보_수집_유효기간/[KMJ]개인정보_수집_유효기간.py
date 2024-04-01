def solution(today, terms, privacies):
    answer = []    
    today = list(map(int, today.split('.')))
    today = today[0] * 28 * 12 + today[1] * 28 + today[2]
    
    terms = {i[0]: int(i[2:]) for i in terms}

    for index, value in enumerate(privacies):
        date, kind = value.split()
        month = terms[kind]
        date = list(map(int, date.split('.')))

        date = date[0] * 28 * 12 + date[1] * 28 + date[2] + month * 28 - 1
        if date < today:
            answer.append(index+1)
                    
    return answer
