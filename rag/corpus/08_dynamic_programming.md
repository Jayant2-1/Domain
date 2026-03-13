# Dynamic Programming

## What is Dynamic Programming?
Dynamic Programming (DP) is an optimization technique that solves complex problems by breaking them into overlapping subproblems and storing their solutions to avoid redundant computation. It is applicable when a problem has:

1. **Optimal Substructure**: The optimal solution can be constructed from optimal solutions of subproblems.
2. **Overlapping Subproblems**: The same subproblems are solved multiple times.

## Two Approaches

### Top-Down (Memoization)
Start from the original problem, recursively solve subproblems, cache results.
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

### Bottom-Up (Tabulation)
Build solutions from smallest subproblems up to the original problem.
```python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

### Space Optimization
Often you only need the last few values, not the entire table.
```python
def fib(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```

## Classic DP Problems

### Climbing Stairs
You can climb 1 or 2 steps at a time. How many distinct ways to reach the top?
```python
def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```
Time: O(n), Space: O(1). Recurrence: dp[i] = dp[i-1] + dp[i-2].

### 0/1 Knapsack
Given items with weights and values, find maximum value that fits in capacity W.
```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]  # don't take item i
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][W]
```
Time: O(n * W), Space: O(n * W). Can optimize to O(W) space.

**Space-optimized 0/1 Knapsack:**
```python
def knapsack_optimized(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # reverse order!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
```

### Longest Common Subsequence (LCS)
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```
Time: O(m * n), Space: O(m * n).

### Longest Increasing Subsequence (LIS)
```python
# O(n^2) DP
def lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# O(n log n) with binary search
import bisect

def lis_optimal(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### Edit Distance (Levenshtein Distance)
Minimum operations (insert, delete, replace) to convert word1 to word2.
```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1]   # replace
                )
    return dp[m][n]
```
Time: O(m * n), Space: O(m * n).

### Coin Change
Minimum coins to make amount.
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
    return dp[amount] if dp[amount] != float('inf') else -1
```
Time: O(amount * len(coins)), Space: O(amount).

### House Robber
Cannot rob two adjacent houses. Maximize total.
```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    return prev1
```
Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i]).

### Matrix Chain Multiplication / Burst Balloons
Interval DP pattern: try all possible split points.
```python
def burst_balloons(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(
                    dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left] * nums[k] * nums[right]
                )
    return dp[0][n - 1]
```

## DP Patterns Summary
1. **Linear DP**: Fibonacci, climbing stairs, house robber. dp[i] depends on dp[i-1], dp[i-2].
2. **Knapsack**: 0/1 knapsack, subset sum, coin change. Choose to include/exclude items.
3. **String DP**: LCS, edit distance, palindrome. Usually 2D table on two strings.
4. **Interval DP**: Matrix chain, burst balloons. Try all split points in an interval.
5. **Grid DP**: Unique paths, minimum path sum. dp[i][j] depends on dp[i-1][j] and dp[i][j-1].
6. **State Machine DP**: Best time to buy/sell stock. States represent current status.
7. **Tree DP**: Rob houses on tree, diameter. Post-order traversal with memoization.
8. **Bitmask DP**: Traveling salesman. Use bitmask to represent subset of items visited.

## Practice Tips
1. Identify the state: What information do you need to solve a subproblem?
2. Write the recurrence relation before coding.
3. Start with top-down (easier to think about), then convert to bottom-up if needed.
4. Look for space optimization opportunities (do you need the entire table?).
5. If the state space is too large, consider if greedy works instead.
6. Draw out small examples and trace through the recurrence.
