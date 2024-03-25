from collections import defaultdict, deque
def solution(edges):
    def check_type(graph1, node):
        queue = deque([node])
        node_link = set()
        edge_count = -1 # 처음에는 간선이 없으므로
        while queue:
            current_node = queue.popleft()
            node_link.add(current_node)
            edge_count +=1
            if current_node in graph1:
                for next_onde in graph1[current_node]:
                    queue.append(next_onde)
                graph1.pop(current_node) # 현재 연결그래프에서 제거
        node_count = len(node_link)
        if node_count == edge_count:
            return 1
        elif node_count == edge_count + 1:
            return 2
        elif node_count == edge_count - 1:
            return 3 
        else:
            return False
    graph1 = defaultdict(set) # 나가는 그래프
    for edge in edges:
        graph1[edge[0]].add(edge[1]) 
    
    # 시작 노드 찾기
    answer = [0,0,0,0]
    max_length = max(len(s) for s in graph1.values())
    start = [k for k, v in graph1.items() if len(v) == max_length][0]
    answer[0] +=start
    for node in graph1.pop(start).copy(): # 시작 노드가 가지고 있는것만 돌기 + 딕셔너리 에러
        if len(graph1) != 0:
            graph_type = check_type(graph1, node) 
            answer[graph_type] += 1
    return answer