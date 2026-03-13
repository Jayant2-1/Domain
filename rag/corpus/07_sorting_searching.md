# Sorting Algorithms

## Overview
Sorting is the process of arranging elements in a specific order (ascending or descending). Sorting is fundamental in computer science and is a prerequisite for many algorithms (binary search, merge operations, etc.).

## Comparison of Sorting Algorithms
| Algorithm       | Best      | Average    | Worst      | Space  | Stable |
|----------------|-----------|------------|------------|--------|--------|
| Bubble Sort     | O(n)      | O(n²)     | O(n²)     | O(1)   | Yes    |
| Selection Sort  | O(n²)    | O(n²)     | O(n²)     | O(1)   | No     |
| Insertion Sort  | O(n)      | O(n²)     | O(n²)     | O(1)   | Yes    |
| Merge Sort      | O(n log n)| O(n log n)| O(n log n)| O(n)   | Yes    |
| Quick Sort      | O(n log n)| O(n log n)| O(n²)     | O(log n)| No    |
| Heap Sort       | O(n log n)| O(n log n)| O(n log n)| O(1)   | No     |
| Counting Sort   | O(n + k)  | O(n + k)  | O(n + k)  | O(k)   | Yes    |
| Radix Sort      | O(nk)     | O(nk)     | O(nk)     | O(n+k) | Yes    |

**Stable** means equal elements maintain their relative order.

## Elementary Sorts

### Bubble Sort
Repeatedly swap adjacent elements if they are in wrong order.
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
```

### Selection Sort
Find minimum element and place it at the beginning.
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

### Insertion Sort
Build the sorted array one element at a time by inserting each element in its correct position.
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```
Best for small arrays or nearly sorted data.

## Efficient Sorts

### Merge Sort
Divide and conquer. Split array in half, sort each half, merge them.
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
Time: O(n log n) always. Space: O(n). Stable. Excellent for linked lists (no random access needed).

### Quick Sort
Choose a pivot, partition array so elements < pivot go left, > pivot go right, recurse.
```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```
Average: O(n log n). Worst: O(n²) with bad pivot. In-place (O(log n) stack space).

**Randomized Quick Sort** avoids worst case:
```python
import random

def randomized_partition(arr, low, high):
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    return partition(arr, low, high)
```

### Heap Sort
Build a max-heap, repeatedly extract the maximum.
```python
def heap_sort(arr):
    n = len(arr)

    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr
```
Time: O(n log n) always. Space: O(1). Not stable.

## Non-Comparison Sorts

### Counting Sort
Works for integers in a known range.
```python
def counting_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for num in arr:
        count[num - min_val] += 1
    for i in range(1, range_val):
        count[i] += count[i - 1]
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1

    return output
```
Time: O(n + k) where k is the range. Space: O(n + k). Stable.

## Searching Algorithms

### Binary Search
Requires sorted array. Halves the search space each step.
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
Time: O(log n), Space: O(1)

### Binary Search Variants

**Find first occurrence:**
```python
def first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

**Find insertion position (bisect_left):**
```python
def search_insert(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

**Binary search on answer (minimize/maximize):**
```python
def min_capacity(weights, days):
    def can_ship(capacity):
        current_load = 0
        needed_days = 1
        for w in weights:
            if current_load + w > capacity:
                needed_days += 1
                current_load = 0
            current_load += w
        return needed_days <= days

    left, right = max(weights), sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

## Practice Tips
1. Know when to use which sort: small data → insertion sort; general purpose → merge/quick sort; integers in range → counting sort.
2. Python's built-in sort (Timsort) is O(n log n) and stable — use it in practice.
3. Binary search is not just for "find element" — it's a technique for searching over any monotonic function.
4. The "binary search on answer" pattern is extremely common in competitive programming.
5. For interviews, know merge sort and quick sort implementations by heart.
