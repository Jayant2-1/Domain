# Arrays

## What is an Array?
An array is a linear data structure that stores elements of the same type in contiguous memory locations. Each element can be accessed directly using its index, making random access O(1) time complexity.

## Key Properties
- **Fixed size** (in most languages): Once declared, the size cannot change (dynamic arrays like Python lists or Java ArrayList resize internally).
- **Contiguous memory**: Elements are stored next to each other in memory, enabling cache-friendly access.
- **Zero-indexed** (in most languages): The first element is at index 0.
- **Homogeneous**: All elements must be of the same data type (in statically-typed languages).

## Time Complexities
| Operation         | Average Case | Worst Case |
|-------------------|-------------|------------|
| Access by index   | O(1)        | O(1)       |
| Search (unsorted) | O(n)        | O(n)       |
| Search (sorted)   | O(log n)    | O(log n)   |
| Insert at end     | O(1)*       | O(n)*      |
| Insert at position| O(n)        | O(n)       |
| Delete at end     | O(1)        | O(1)       |
| Delete at position| O(n)        | O(n)       |

*Amortized O(1) for dynamic arrays; O(n) when resizing is needed.

## Common Patterns

### Two Pointer Technique
Use two pointers moving toward each other or in the same direction to solve problems efficiently.

**Example: Two Sum (sorted array)**
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]
```
Time: O(n), Space: O(1)

### Sliding Window
Maintain a window of elements and slide it across the array.

**Example: Maximum sum subarray of size k**
```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return -1
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```
Time: O(n), Space: O(1)

### Prefix Sum
Precompute cumulative sums for efficient range queries.

**Example: Range sum query**
```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    return prefix[right + 1] - prefix[left]
```
Time: O(n) build, O(1) per query. Space: O(n)

## Classic Problems

### Kadane's Algorithm — Maximum Subarray Sum
Find the contiguous subarray with the largest sum.
```python
def max_subarray_sum(arr):
    max_ending_here = max_so_far = arr[0]
    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
```
Time: O(n), Space: O(1). Uses dynamic programming.

### Dutch National Flag — Sort Colors
Sort an array of 0s, 1s, and 2s in-place.
```python
def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```
Time: O(n), Space: O(1)

### Merge Intervals
Given a list of intervals, merge overlapping ones.
```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```
Time: O(n log n), Space: O(n)

## Practice Tips
1. Always clarify: Is the array sorted? Are there duplicates? What are the constraints on values?
2. Think about edge cases: empty array, single element, all same elements.
3. Consider whether sorting the array first would simplify the problem.
4. For subarray problems, think about prefix sums or sliding window.
5. For pair/triplet problems, consider two pointers after sorting.
