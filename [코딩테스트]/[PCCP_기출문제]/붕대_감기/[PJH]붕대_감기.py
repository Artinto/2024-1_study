def solution(bandage, health, attacks):
    t, x, y = bandage
    current_health = health
    last_attack_time = max(attacks, key=lambda a: a[0])[0]
    current_time = 0
    consecutive_success = 0 
    attack_index = 0
    
    while current_time <= last_attack_time:
        if attack_index < len(attacks) and attacks[attack_index][0] == current_time:
            current_health -= attacks[attack_index][1]
            if current_health <= 0:
                return -1
            consecutive_success = 0
            attack_index += 1
        else:
            if consecutive_success < t:
                current_health += x
                consecutive_success += 1
            if consecutive_success == t:
                current_health += y
                consecutive_success = 0 
            current_health = min(current_health, health)
        
        current_time += 1
    
    return current_health
