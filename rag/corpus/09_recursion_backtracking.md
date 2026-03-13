# Recursion and Backtracking

## Recursion
Recursion is a technique where a function calls itself to solve smaller instances of the same problem. Every recursive solution needs:

1. **Base Case**: The simplest case that can be solved directly (stops recursion).
2. **Recursive Case**: Break the problem into smaller subproblems and call the function recursively.

### Anatomy of a Recursive Function
```python
def factorial(n):
    # Base case
    if n <= 1:
        return 1
    # Recursive case
    return n * factorial(n - 1)
```

### How Recursion Works
Each recursive call adds a frame to the call stack. When the base case is reached, frames are popped and results are combined.

**Call stack for factorial(4):**
```
factorial(4) → 4 * factorial(3)
                   → 3 * factorial(2)
                          → 2 * factorial(1)
                                 → returns 1
                          → returns 2 * 1 = 2
                   → returns 3 * 2 = 6
              → returns 4 * 6 = 24
```

### Recursion vs Iteration
| Aspect      | Recursion          | Iteration          |
|-------------|--------------------|--------------------|
| Readability | Often more elegant | Sometimes verbose  |
| Space       | O(n) stack space   | O(1) if no extra DS |
| Speed       | Function call overhead | Generally faster |
| Risk        | Stack overflow     | Infinite loop      |

### Tail Recursion
When the recursive call is the last operation. Some languages optimize this to avoid stack growth (Python does not).
```python
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, n * acc)
```

## Classic Recursive Problems

### Power Function
```python
def power(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = power(base, exp // 2)
        return half * half
    return base * power(base, exp - 1)
```
Time: O(log n) — fast exponentiation.

### Tower of Hanoi
Move n disks from source to target using an auxiliary peg.
```python
def hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, auxiliary, target, source)
```
Time: O(2^n). Minimum moves required: 2^n - 1.

## Backtracking
Backtracking is a systematic method for exploring all possible configurations of a search space. It builds a solution incrementally, abandoning a path ("backtracking") as soon as it determines the path cannot lead to a valid solution.

### Backtracking Template
```python
def backtrack(candidates, path, result):
    if is_solution(path):
        result.append(path[:])  # make a copy
        return

    for candidate in get_candidates(candidates, path):
        if is_valid(candidate, path):
            path.append(candidate)       # choose
            backtrack(candidates, path, result)  # explore
            path.pop()                   # un-choose (backtrack)
```

## Classic Backtracking Problems

### Subsets (Power Set)
Generate all subsets of a set.
```python
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result
```
Time: O(2^n), Space: O(n) for recursion depth.

### Permutations
Generate all permutations.
```python
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result
```
Time: O(n!), Space: O(n).

### Combination Sum
Find all combinations that sum to target (can reuse elements).
```python
def combination_sum(candidates, target):
    result = []
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    candidates.sort()
    backtrack(0, [], target)
    return result
```

### N-Queens
Place N queens on an N×N board so no two queens threaten each other.
```python
def solve_n_queens(n):
    result = []
    board = [['.' ] * n for _ in range(n)]

    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return result
```

### Sudoku Solver
```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True

    backtrack()
```

### Word Search
Find if a word exists in a grid.
```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
            return False
        temp = board[r][c]
        board[r][c] = '#'  # mark visited
        found = (backtrack(r+1, c, idx+1) or backtrack(r-1, c, idx+1) or
                 backtrack(r, c+1, idx+1) or backtrack(r, c-1, idx+1))
        board[r][c] = temp  # un-mark
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
```

## When to Use What
- **Recursion**: Natural for tree/graph traversal, divide and conquer (merge sort, binary search).
- **Backtracking**: Combinatorial problems (subsets, permutations, N-Queens), constraint satisfaction.
- **DP vs Backtracking**: If subproblems overlap → DP. If you need all solutions or pruning → Backtracking.

## Practice Tips
1. Always identify the base case first.
2. Trace through small examples by hand.
3. Watch for Python's default recursion limit (sys.setrecursionlimit).
4. In backtracking, the key optimization is pruning — skip invalid branches early.
5. Drawing the recursion tree helps visualize what's happening.
