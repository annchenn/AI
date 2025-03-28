import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
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
    num_visited=0
    path=[]
    parent={start:None}
    visited=set()
    dist={start:0}
    stack=[]
    stack.append(start)
    visited.add(start)
    while stack:
        current=stack.pop()
        num_visited+=1
        if current == end:
            now=current
            while now is not None:
                path.append(now)
                now=parent[now]
            path.reverse()
            return path, dist[current], num_visited
        
        if current in edge:
            for dest, d in edge[current]:
                if dest not in visited:
                    parent[dest]=current
                    dist[dest]=dist[current]+d
                    visited.add(dest)
                    stack.append(dest)
    return [], 0, num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
