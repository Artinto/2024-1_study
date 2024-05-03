def solution(phone_number):
    len_num = len(phone_number)
    answer = (len_num-4)*'*' + phone_number[len_num-4:len_num]

    return answer