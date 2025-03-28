import csv
from queue import PriorityQueue 
edgeFile = 'edges.csv'


def ucs(start, end):
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
            
    pq=PriorityQueue()
    pq.put((0,start))
    num_visited=0
    dist={start:0}
    visited=set()
    parent={start:None}
    path=[]
    while not pq.empty():
        p, current=pq.get()
        if current in visited:
            continue
        num_visited+=1
        visited.add(current)
        if current==end:
            now=current
            while now is not None:
                path.append(now)
                now=parent[now]
            path.reverse()
            return path, dist[current], num_visited
        if current in edge:
            for node, d in edge[current]:
                new_dist=dist[current]+d
                if node not in visited and (node not in dist or new_dist < dist[node]):
                    dist[node]=new_dist
                    pq.put((new_dist,node))
                    parent[node]=current
    # End your code (Part 3)

if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
