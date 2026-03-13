# String Algorithms

## String Basics
Strings are sequences of characters. In Python, strings are immutable — any modification creates a new string.

### Key Operations and Their Complexities
| Operation               | Time Complexity |
|------------------------|----------------|
| Access by index s[i]    | O(1)           |
| Slice s[i:j]           | O(j - i)       |
| Concatenation s1 + s2   | O(len(s1) + len(s2)) |
| Search (in)            | O(n * m)       |
| len(s)                 | O(1)           |
| s.split()              | O(n)           |
| ''.join(list)          | O(total length) |

### String Building
Avoid repeated concatenation (creates new string each time). Use list + join:
```python
# Bad: O(n^2)
result = ""
for char in chars:
    result += char

# Good: O(n)
parts = []
for char in chars:
    parts.append(char)
result = ''.join(parts)
```

## Common String Problems

### Reverse a String
```python
def reverse_string(s):
    return s[::-1]

# In-place (list of chars)
def reverse_in_place(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
```

### Check Palindrome
```python
def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

# Two pointer approach
def is_palindrome_two_pointer(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### Longest Palindromic Substring
Expand around center approach.
```python
def longest_palindrome(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    result = ""
    for i in range(len(s)):
        # Odd length palindromes
        odd = expand(i, i)
        if len(odd) > len(result):
            result = odd
        # Even length palindromes
        even = expand(i, i + 1)
        if len(even) > len(result):
            result = even
    return result
```
Time: O(n^2), Space: O(1).

### String Matching — KMP Algorithm
Find pattern in text efficiently.
```python
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    lps = build_lps(pattern)
    results = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return results
```
Time: O(n + m), Space: O(m). The LPS (Longest Proper Prefix which is also Suffix) array avoids re-scanning matched characters.

### Rabin-Karp Algorithm (Rolling Hash)
```python
def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    if m > n:
        return []
    BASE, MOD = 256, 10**9 + 7

    pattern_hash = 0
    text_hash = 0
    h = pow(BASE, m - 1, MOD)
    results = []

    for i in range(m):
        pattern_hash = (pattern_hash * BASE + ord(pattern[i])) % MOD
        text_hash = (text_hash * BASE + ord(text[i])) % MOD

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                results.append(i)
        if i < n - m:
            text_hash = (BASE * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % MOD
            if text_hash < 0:
                text_hash += MOD
    return results
```
Average: O(n + m), Worst: O(nm). Good for multiple pattern matching.

### Trie (Prefix Tree)
Efficient for prefix operations and autocomplete.
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```
Insert/Search/StartsWith: O(m) where m is word/prefix length.

### Anagram Detection
```python
from collections import Counter

def is_anagram(s, t):
    return Counter(s) == Counter(t)

def find_anagrams(s, p):
    """Find all start indices of p's anagrams in s."""
    result = []
    p_count = Counter(p)
    s_count = Counter()
    for i in range(len(s)):
        s_count[s[i]] += 1
        if i >= len(p):
            if s_count[s[i - len(p)]] == 1:
                del s_count[s[i - len(p)]]
            else:
                s_count[s[i - len(p)]] -= 1
        if s_count == p_count:
            result.append(i - len(p) + 1)
    return result
```

### Encode and Decode Strings
```python
def encode(strs):
    return ''.join(f'{len(s)}#{s}' for s in strs)

def decode(s):
    result = []
    i = 0
    while i < len(s):
        j = s.index('#', i)
        length = int(s[i:j])
        result.append(s[j + 1:j + 1 + length])
        i = j + 1 + length
    return result
```

## Practice Tips
1. Remember strings are immutable in Python. Use list operations for in-place modifications.
2. Sliding window + hash map is powerful for substring problems.
3. Two pointers work great for palindrome and reverse problems.
4. KMP is the go-to for efficient pattern matching.
5. Trie is ideal for autocomplete, spell-checking, and prefix-based operations.
6. For anagram problems, think frequency counting.
