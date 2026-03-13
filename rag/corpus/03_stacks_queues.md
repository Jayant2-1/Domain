# Stacks and Queues

## Stack
A stack is a Last-In-First-Out (LIFO) data structure. The last element added is the first one removed.

### Operations
| Operation | Time Complexity |
|-----------|----------------|
| push      | O(1)           |
| pop       | O(1)           |
| peek/top  | O(1)           |
| isEmpty   | O(1)           |
| size      | O(1)           |

### Implementation
```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
```

### Classic Stack Problems

#### Valid Parentheses
Check if a string of brackets is balanced.
```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return len(stack) == 0
```
Time: O(n), Space: O(n)

#### Min Stack
Stack that supports getMin() in O(1).
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val

    def getMin(self):
        return self.min_stack[-1]
```

#### Evaluate Reverse Polish Notation
```python
def eval_rpn(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    for token in tokens:
        if token in operators:
            b, a = stack.pop(), stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]
```

#### Monotonic Stack — Next Greater Element
```python
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []  # stores indices
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
```
Time: O(n), Space: O(n). Each element is pushed and popped at most once.

#### Largest Rectangle in Histogram
```python
def largest_rectangle(heights):
    stack = []
    max_area = 0
    heights.append(0)  # sentinel
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()
    return max_area
```
Time: O(n), Space: O(n)

## Queue
A queue is a First-In-First-Out (FIFO) data structure. The first element added is the first one removed.

### Operations
| Operation | Time Complexity |
|-----------|----------------|
| enqueue   | O(1)           |
| dequeue   | O(1)*          |
| front     | O(1)           |
| isEmpty   | O(1)           |
| size      | O(1)           |

*O(1) amortized using collections.deque; O(n) if using list.pop(0).

### Implementation
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()

    def front(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
```

### Classic Queue Problems

#### Implement Queue using Two Stacks
```python
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self._move()
        return self.out_stack.pop()

    def peek(self):
        self._move()
        return self.out_stack[-1]

    def _move(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def empty(self):
        return not self.in_stack and not self.out_stack
```
Amortized O(1) per operation.

#### Implement Stack using Two Queues
```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
```

## Priority Queue / Heap
A priority queue serves elements based on priority rather than insertion order. Commonly implemented using a binary heap.

### Python's heapq (min-heap)
```python
import heapq

nums = [3, 1, 4, 1, 5, 9]
heapq.heapify(nums)         # O(n)
heapq.heappush(nums, 2)     # O(log n)
smallest = heapq.heappop(nums)  # O(log n)
k_smallest = heapq.nsmallest(3, nums)  # O(n + k log n)
k_largest = heapq.nlargest(3, nums)    # O(n + k log n)
```

### Top K Elements Pattern
```python
def top_k_frequent(nums, k):
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

### Kth Largest Element
```python
def find_kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]
```
Time: O(n log k), Space: O(k)

## Deque (Double-Ended Queue)
Supports insertion and deletion at both ends in O(1).

### Sliding Window Maximum
```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # stores indices
    result = []
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```
Time: O(n), Space: O(k)

## Practice Tips
1. When you see "matching" or "nesting" — think Stack.
2. When you see "order processing" or "BFS" — think Queue.
3. For "top K" or "kth smallest/largest" — think Heap.
4. Monotonic stack is powerful for "next greater/smaller element" problems.
5. Always check: what happens when the stack/queue is empty?
