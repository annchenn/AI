import csv
import queue
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    edge={}
    with open(edgeFile, 'r') as file:
        now=csv.reader(file)
        next(now)
        for row in now:
            src=int(row[0])
            dest=int(row[1])
            dist=float(row[2])
            
            if src not in edge:
                edge[src]=[]
            edge[src].append((dest,dist))
    path=[]
    num_visited=0
    q=queue.Queue()
    q.put(start)
    distance={start:0}
    visited=set()
    visited.add(start)
    parent={start:None}
    
    while not q.empty():
        now=q.get()
        num_visited+=1
        if now == end:
            current=now
            while current is not None:
                path.append(current)
                current=parent[current]
            path.reverse()
            return path, distance[now], num_visited
        if now in edge:
            for node, dist in edge[now]:
                    if node not in visited:
                        q.put(node)
                        distance[node]=distance[now]+dist
                        visited.add(node)
                        parent[node]=now
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
