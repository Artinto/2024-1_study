def solution(bandage, health, attacks):
    answer = 0
    initial_health = health # 최초의 체력 -> 회복 상한선
    attack_time = [attack[0] for attack in attacks] # attacks를 공격시간과
    attack_damage = [attack[1] for attack in attacks] # damage로 분리

    for i in range(len(attack_time)):
        if i > 0: # i가 0일때 빼고는 전부 체력 회복이 적용되므로 조건 먼저 적용
            time_diff = attack_time[i] - attack_time[i-1]
            if time_diff >= 2:
                health += (time_diff - 1) * bandage[1] # 시간차만큼 체력 회복

            while time_diff > bandage[0]: # 필요수치 이상 연속 회복하면
                health += bandage[2] # 추가 체력 증가 적용
                time_diff -= bandage[0] # 만약 2배 이상일 경우 감안

            if health > initial_health: # 최대 체력 넘지 않게
                health = initial_health # 회복 한계선 설정

        health -= attack_damage[i] # 체력이 0이하가 되면
        if health <= 0:
            answer = -1 # -1로 표현
            break

    if health > 0:
        answer = health

    return answer
