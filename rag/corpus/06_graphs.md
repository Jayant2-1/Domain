# Graphs

## What is a Graph?
A graph G = (V, E) consists of a set of vertices (nodes) V and a set of edges E connecting pairs of vertices. Unlike trees, graphs can have cycles, disconnected components, and edges can be directed or undirected.

## Types of Graphs
- **Undirected Graph**: Edges have no direction. If (u, v) exists, so does (v, u).
- **Directed Graph (Digraph)**: Edges have direction. (u, v) means u → v.
- **Weighted Graph**: Edges have associated weights/costs.
- **Unweighted Graph**: All edges have equal weight (or weight = 1).
- **DAG (Directed Acyclic Graph)**: Directed graph with no cycles. Used for topological sorting.
- **Connected Graph**: There is a path between every pair of vertices.
- **Complete Graph**: Every pair of vertices is connected by an edge.

## Graph Representations

### Adjacency List (Most Common)
```python
# Using dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Using defaultdict
from collections import defaultdict
graph = defaultdict(list)
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # for undirected
```
Space: O(V + E). Best for sparse graphs.

### Adjacency Matrix
```python
# n x n matrix where matrix[i][j] = 1 if edge exists
n = 4
matrix = [[0] * n for _ in range(n)]
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
for u, v in edges:
    matrix[u][v] = 1
    matrix[v][u] = 1  # for undirected
```
Space: O(V^2). Best for dense graphs. O(1) edge lookup.

## Graph Traversals

### Breadth-First Search (BFS)
Explores all neighbors at current depth before moving deeper. Uses a queue.
```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
```
Time: O(V + E), Space: O(V)

**BFS for shortest path in unweighted graph:**
```python
def shortest_path_bfs(graph, start, end):
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []  # no path exists
```

### Depth-First Search (DFS)
Explores as far as possible along each branch before backtracking.
```python
# Recursive
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    result = [node]
    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    return result

# Iterative
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order
```
Time: O(V + E), Space: O(V)

## Classic Graph Algorithms

### Number of Connected Components
```python
def count_components(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    count = 0
    for node in range(n):
        if node not in visited:
            count += 1
            # BFS or DFS to visit all nodes in component
            queue = deque([node])
            visited.add(node)
            while queue:
                curr = queue.popleft()
                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    return count
```

### Cycle Detection

**Undirected graph (DFS):**
```python
def has_cycle_undirected(graph, n):
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in range(n):
        if node not in visited:
            if dfs(node, -1):
                return True
    return False
```

**Directed graph (3-color DFS):**
```python
def has_cycle_directed(graph, n):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # back edge = cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in range(n):
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False
```

### Topological Sort (DAG only)
Linear ordering of vertices such that for every edge (u, v), u comes before v.

**Kahn's Algorithm (BFS-based):**
```python
def topological_sort(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else []  # empty = cycle exists
```

### Dijkstra's Algorithm (Shortest Path, Non-negative Weights)
```python
import heapq

def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist
```
Time: O((V + E) log V), Space: O(V)

### Union-Find (Disjoint Set Union)
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```
Nearly O(1) amortized per operation with path compression and union by rank.

## Practice Tips
1. BFS = shortest path in unweighted graphs. Dijkstra = weighted graphs.
2. DFS is often simpler for connectivity, cycle detection, and path finding.
3. For "number of islands" type problems, use BFS/DFS from each unvisited cell.
4. Topological sort only works on DAGs. If the sort doesn't include all nodes, there's a cycle.
5. Union-Find is great for "connected components" and "is connected?" queries.
6. Always clarify: directed vs undirected, weighted vs unweighted, cyclic vs acyclic.
