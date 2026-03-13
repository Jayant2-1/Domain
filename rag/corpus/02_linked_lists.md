# Linked Lists

## What is a Linked List?
A linked list is a linear data structure where elements (nodes) are stored in non-contiguous memory locations. Each node contains data and a reference (pointer) to the next node. Unlike arrays, linked lists do not support random access — you must traverse from the head.

## Types of Linked Lists

### Singly Linked List
Each node points to the next node. The last node points to None/null.
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Doubly Linked List
Each node has pointers to both next and previous nodes.
```python
class DListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### Circular Linked List
The last node points back to the first node, forming a cycle.

## Time Complexities
| Operation           | Singly | Doubly |
|---------------------|--------|--------|
| Access by index     | O(n)   | O(n)   |
| Search              | O(n)   | O(n)   |
| Insert at head      | O(1)   | O(1)   |
| Insert at tail      | O(n)*  | O(1)** |
| Insert at position  | O(n)   | O(n)   |
| Delete at head      | O(1)   | O(1)   |
| Delete at tail      | O(n)   | O(1)   |
| Delete at position  | O(n)   | O(n)   |

*O(1) if you maintain a tail pointer. **With tail pointer.

## Common Patterns

### Dummy Head Node
Use a sentinel/dummy node to simplify edge cases at the head.
```python
def remove_elements(head, val):
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next
    return dummy.next
```

### Fast and Slow Pointers (Floyd's Cycle Detection)
Use two pointers moving at different speeds.

**Detect cycle in a linked list:**
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

**Find middle of linked list:**
```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

**Find start of cycle:**
```python
def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

## Classic Problems

### Reverse a Linked List
```python
def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```
Time: O(n), Space: O(1)

**Recursive version:**
```python
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```
Time: O(n), Space: O(n) due to recursion stack

### Merge Two Sorted Lists
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 or l2
    return dummy.next
```
Time: O(n + m), Space: O(1)

### Remove Nth Node From End
```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```
Time: O(n), Space: O(1). Uses the gap technique with two pointers.

### LRU Cache (Doubly Linked List + Hash Map)
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = DListNode()  # dummy head
        self.tail = DListNode()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_front(node)
            return node.val
        return -1

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = DListNode(value)
        self.cache[key] = node
        self._add_to_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```
Time: O(1) for both get and put. Space: O(capacity).

## Practice Tips
1. Always consider using a dummy head node to handle edge cases.
2. Draw pictures! Linked list problems become much clearer when visualized.
3. Be careful with pointer manipulation order — changing the wrong pointer first can lose nodes.
4. For cycle-related problems, think Floyd's algorithm.
5. When asked "in-place" for linked lists, it means O(1) extra space (ignoring recursion stack).
