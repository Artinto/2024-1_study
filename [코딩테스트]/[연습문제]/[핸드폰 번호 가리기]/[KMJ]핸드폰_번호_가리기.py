def solution(phone_number):
    star = '*'
    return (star*(len(phone_number)-4)+phone_number[-4:])
