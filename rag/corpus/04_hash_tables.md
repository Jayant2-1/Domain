# Hash Tables (Hash Maps)

## What is a Hash Table?
A hash table is a data structure that maps keys to values using a hash function. The hash function computes an index into an array of buckets, from which the desired value can be found. It provides average O(1) time for insert, delete, and lookup operations.

## How Hashing Works
1. **Hash Function**: Takes a key and returns an integer (hash code).
2. **Index Mapping**: hash_code % array_size gives the bucket index.
3. **Collision Handling**: When two keys map to the same index.

### Collision Resolution Strategies
- **Chaining**: Each bucket contains a linked list of entries.
- **Open Addressing**: Find another empty slot (linear probing, quadratic probing, double hashing).
- **Robin Hood Hashing**: Steal from rich (short probe distance) to give to poor (long probe distance).

## Time Complexities
| Operation | Average | Worst (poor hash) |
|-----------|---------|--------------------|
| Insert    | O(1)    | O(n)               |
| Delete    | O(1)    | O(n)               |
| Lookup    | O(1)    | O(n)               |
| Space     | O(n)    | O(n)               |

Worst case occurs when all keys hash to the same bucket (degenerate to linked list).

## Python Dictionaries and Sets
Python dict is a highly optimized hash table implementation.
```python
# Dictionary operations
d = {}
d['key'] = 'value'        # O(1) insert
val = d.get('key', None)   # O(1) lookup with default
del d['key']               # O(1) delete
'key' in d                 # O(1) membership test

# Set operations (hash set)
s = set()
s.add(1)                   # O(1)
s.remove(1)                # O(1), raises KeyError if missing
s.discard(1)               # O(1), no error if missing
1 in s                     # O(1)
```

## Common Patterns

### Frequency Counter
```python
from collections import Counter

def frequency_count(arr):
    count = Counter(arr)
    # or manually:
    count = {}
    for item in arr:
        count[item] = count.get(item, 0) + 1
    return count
```

### Two Sum (Unsorted Array)
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```
Time: O(n), Space: O(n). Single pass with hash map.

### Group Anagrams
```python
from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```
Time: O(n * k log k) where k is max string length. Space: O(n * k).

### Subarray Sum Equals K (Prefix Sum + Hash Map)
```python
def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_counts = {0: 1}
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_counts:
            count += prefix_counts[prefix_sum - k]
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1
    return count
```
Time: O(n), Space: O(n). Key insight: if prefix[j] - prefix[i] = k, then subarray [i+1..j] sums to k.

### Longest Substring Without Repeating Characters
```python
def length_of_longest_substring(s):
    char_index = {}
    max_length = 0
    start = 0
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    return max_length
```
Time: O(n), Space: O(min(n, alphabet_size))

### First Non-Repeating Character
```python
def first_unique_char(s):
    count = Counter(s)
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    return -1
```
Time: O(n), Space: O(1) — bounded by alphabet size.

### Isomorphic Strings
```python
def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    s_to_t = {}
    t_to_s = {}
    for cs, ct in zip(s, t):
        if cs in s_to_t and s_to_t[cs] != ct:
            return False
        if ct in t_to_s and t_to_s[ct] != cs:
            return False
        s_to_t[cs] = ct
        t_to_s[ct] = cs
    return True
```

## Advanced: Consistent Hashing
Used in distributed systems to minimize key redistribution when nodes are added/removed. Maps both keys and nodes onto a ring (0 to 2^32 - 1). Each key is assigned to the next node clockwise on the ring.

## Practice Tips
1. Hash maps are your go-to for "find pair/complement" problems.
2. Prefix sum + hash map is a powerful combination for subarray sum problems.
3. When you need O(1) lookup — think hash map/set.
4. Counter/defaultdict from collections simplifies many patterns.
5. Be aware of hash collisions in interview discussions.
6. For "group by" problems, the key design in the hash map is crucial.
