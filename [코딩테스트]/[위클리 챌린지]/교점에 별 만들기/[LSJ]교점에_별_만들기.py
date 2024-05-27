def solution(line):
    points = set()

    for i in range(len(line)):
        for j in range(i + 1, len(line)):
            a1, b1, c1 = line[i]
            a2, b2, c2 = line[j]

            denominator = a1 * b2 - a2 * b1
            if denominator != 0:
                x = (b1 * c2 - b2 * c1) / denominator
                y = (a2 * c1 - a1 * c2) / denominator
                if x.is_integer() and y.is_integer():
                    points.add((int(x), int(y)))
                  
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    width = max_x - min_x + 1
    height = max_y - min_y + 1
  
    matrix = [['.' for _ in range(width)] for _ in range(height)]
    for point in points:
        x, y = point
        matrix[max_y - y][x - min_x] = '*'
    matrix_str_list = [''.join(row) for row in matrix]
    return matrix_str_list
