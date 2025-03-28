import csv
from queue import PriorityQueue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'


def astar(start, end):
    # Begin your code (Part 4)
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
    h={}
    with open(heuristicFile,'r') as file:
        reader=csv.reader(file)
        header=next(reader)
        #print(header)
        dest=[int(header[i]) for i in range(1,len(header))]
        for i in range(0,len(header)-1):
            h[dest[i]]={}
        
        for row in reader:
            node=int(row[0])
            for i in range(0,len(header)-1):
                d=float(row[i+1])
                h[dest[i]][node]=d
                
    heuristic=h[end]
    pq=PriorityQueue()
    path=[]
    num_visited=0
    visited=set()
    pq.put((1,start))
    parent={start:None}
    dist={start:0}
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
                    priority=new_dist+heuristic[node]
                    pq.put((priority,node))
                    parent[node]=current
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
