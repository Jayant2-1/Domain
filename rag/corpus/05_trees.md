# Trees

## What is a Tree?
A tree is a hierarchical, non-linear data structure consisting of nodes connected by edges. It has a root node and every node (except root) has exactly one parent. Nodes with no children are called leaves.

## Binary Tree
Each node has at most two children (left and right).
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Types of Binary Trees
- **Full Binary Tree**: Every node has 0 or 2 children.
- **Complete Binary Tree**: All levels filled except possibly the last, which is filled left to right.
- **Perfect Binary Tree**: All internal nodes have 2 children, all leaves at same level.
- **Balanced Binary Tree**: Height of left and right subtrees differ by at most 1 at every node.
- **Degenerate/Skewed Tree**: Every node has only one child (essentially a linked list).

### Properties
- A binary tree with n nodes has exactly n-1 edges.
- Maximum nodes at level i: 2^i (root is level 0).
- Maximum nodes in a tree of height h: 2^(h+1) - 1.
- Minimum height of a tree with n nodes: floor(log2(n)).

## Tree Traversals

### Depth-First Search (DFS)

#### Inorder (Left, Root, Right) — gives sorted order for BST
```python
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# Iterative
def inorder_iterative(root):
    result, stack = [], []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result
```

#### Preorder (Root, Left, Right)
```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# Iterative
def preorder_iterative(root):
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

#### Postorder (Left, Right, Root)
```python
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### Breadth-First Search (BFS) — Level Order Traversal
```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```
Time: O(n), Space: O(n)

## Binary Search Tree (BST)
A BST maintains the property: for every node, all values in the left subtree are less, and all values in the right subtree are greater.

### Time Complexities
| Operation | Average   | Worst (skewed) |
|-----------|-----------|----------------|
| Search    | O(log n)  | O(n)           |
| Insert    | O(log n)  | O(n)           |
| Delete    | O(log n)  | O(n)           |

### BST Operations
```python
def search_bst(root, target):
    if not root or root.val == target:
        return root
    if target < root.val:
        return search_bst(root.left, target)
    return search_bst(root.right, target)

def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    elif val > root.val:
        root.right = insert_bst(root.right, val)
    return root

def delete_bst(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # Find inorder successor (smallest in right subtree)
        successor = root.right
        while successor.left:
            successor = successor.left
        root.val = successor.val
        root.right = delete_bst(root.right, successor.val)
    return root
```

### Validate BST
```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))
```

## Classic Tree Problems

### Maximum Depth of Binary Tree
```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### Check if Tree is Balanced
```python
def is_balanced(root):
    def check(node):
        if not node:
            return 0
        left = check(node.left)
        right = check(node.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return check(root) != -1
```

### Lowest Common Ancestor (LCA)
```python
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right
```

### Diameter of Binary Tree
```python
def diameter(root):
    max_d = [0]
    def depth(node):
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        max_d[0] = max(max_d[0], left + right)
        return 1 + max(left, right)
    depth(root)
    return max_d[0]
```

### Serialize and Deserialize Binary Tree
```python
def serialize(root):
    if not root:
        return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    values = iter(data.split(","))
    def build():
        val = next(values)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    return build()
```

## Self-Balancing Trees
- **AVL Tree**: Strictly balanced (height difference ≤ 1). Faster lookups.
- **Red-Black Tree**: Approximately balanced. Faster insertions/deletions. Used in Java TreeMap, C++ std::map.
- **B-Tree / B+ Tree**: Used in databases and file systems. Optimized for disk access.

## Practice Tips
1. Most tree problems can be solved with recursion. Think about the base case and the recursive relation.
2. For BST problems, leverage the ordering property.
3. BFS = level-order. DFS = preorder/inorder/postorder.
4. When calculating tree properties (height, diameter), think bottom-up.
5. "Path" problems often need a helper that returns values to the parent.
