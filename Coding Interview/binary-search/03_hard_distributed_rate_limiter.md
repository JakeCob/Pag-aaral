# Hard: Distributed API Rate Limiter with Binary Search Optimization

## Problem Statement

You are building a distributed rate limiter for ELGO AI's multi-tenant SaaS platform. Each customer has a different rate limit, and you need to efficiently determine:

1. The maximum number of requests a client can make in a given time window
2. The earliest timestamp when a client can make their next request after being rate-limited
3. Given a sorted log of API timestamps and a rate limit configuration, find the maximum request rate (requests per second) that would NOT have triggered rate limiting

## Real-World Context

In production systems at ELGO AI:
- Multiple customers make concurrent API calls to LLM endpoints
- Each customer tier (Free, Pro, Enterprise) has different rate limits
- Rate limits can be per-second, per-minute, or sliding window
- You need to efficiently compute rate limit violations for logging and billing
- Historical data analysis helps optimize rate limit policies

This problem combines **sliding window rate limiting** with **binary search optimization** for efficient violation detection and threshold computation.

## Function Signatures

```python
def max_requests_in_window(
    timestamps: list[int],
    window_ms: int,
    max_requests: int
) -> int:
    """
    Find the maximum number of requests that occurred in any window.

    Args:
        timestamps: Sorted list of request timestamps in milliseconds
        window_ms: Sliding window duration in milliseconds
        max_requests: Rate limit threshold

    Returns:
        Maximum number of requests in any sliding window of size window_ms
    """
    pass


def earliest_next_request(
    timestamps: list[int],
    window_ms: int,
    max_requests: int,
    current_time: int
) -> int:
    """
    Find the earliest time a new request can be made without violating rate limit.

    Args:
        timestamps: Sorted list of recent request timestamps in milliseconds
        window_ms: Sliding window duration in milliseconds
        max_requests: Rate limit threshold
        current_time: Current timestamp in milliseconds

    Returns:
        Earliest timestamp for next allowed request (in milliseconds)
        Returns current_time if request can be made immediately
    """
    pass


def find_safe_rate_limit(
    timestamps: list[int],
    window_ms: int
) -> int:
    """
    Find the maximum rate limit that would NOT have caused any violations.

    Given historical request data, determine the safest rate limit that would
    have allowed all requests without rejections.

    Args:
        timestamps: Sorted list of request timestamps in milliseconds
        window_ms: Sliding window duration in milliseconds

    Returns:
        Maximum rate limit (requests per window) with zero violations
    """
    pass
```

## Examples

### Example 1: Maximum Requests in Window
```python
# Timestamps in milliseconds (10 requests over 5 seconds)
timestamps = [1000, 1200, 1500, 2000, 2200, 2800, 3100, 3500, 3900, 4200]
window_ms = 2000  # 2-second window
max_requests = 5

result = max_requests_in_window(timestamps, window_ms, max_requests)
print(result)
```

**Output:**
```
6
```

**Explanation:**
- Window [2000, 4000]: Contains timestamps [2000, 2200, 2800, 3100, 3500, 3900] = 6 requests
- Window [2200, 4200]: Contains timestamps [2200, 2800, 3100, 3500, 3900, 4200] = 6 requests
- Maximum is 6 requests (would violate limit of 5)

### Example 2: Find Next Available Request Time
```python
timestamps = [1000, 1100, 1200, 1300, 1400]  # 5 requests
window_ms = 1000  # 1-second window
max_requests = 5
current_time = 1400

result = earliest_next_request(timestamps, window_ms, max_requests, current_time)
print(result)
```

**Output:**
```
2000
```

**Explanation:**
- Current window [400, 1400] has 5 requests (at limit)
- For next request to be valid, oldest request (1000) must fall outside window
- Next valid timestamp: 1000 + window_ms = 2000

### Example 3: Find Safe Rate Limit
```python
timestamps = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
window_ms = 2000  # 2-second window

result = find_safe_rate_limit(timestamps, window_ms)
print(result)
```

**Output:**
```
4
```

**Explanation:**
Let's check each possible window:
- [1000, 3000]: timestamps [1000, 1500, 2000, 2500, 3000] = 5 requests
- [1500, 3500]: timestamps [1500, 2000, 2500, 3000, 3500] = 5 requests
- [2000, 4000]: timestamps [2000, 2500, 3000, 3500, 4000] = 5 requests
- [2500, 4500]: timestamps [2500, 3000, 3500, 4000, 4500] = 5 requests

Wait, let me recalculate...

Actually, we need to find the MAXIMUM number of requests in ANY 2-second window:
- Sliding window from 1000:
  - [1000, 3000): 1000, 1500, 2000, 2500 = 4 requests
  - [1500, 3500): 1500, 2000, 2500, 3000 = 4 requests
  - [2000, 4000): 2000, 2500, 3000, 3500 = 4 requests
  - [2500, 4500): 2500, 3000, 3500, 4000 = 4 requests
  - [3000, 5000): 3000, 3500, 4000, 4500 = 4 requests

Maximum requests in any window = 4, so safe rate limit is 4.

### Example 4: Complex Sliding Window Analysis
```python
# Timestamps over 10 seconds with burst traffic
timestamps = [
    1000, 1050, 1100,           # Burst: 3 requests in 100ms
    2000, 2500, 3000,           # Spread: 3 requests in 1s
    5000, 5100, 5200, 5300,     # Burst: 4 requests in 300ms
    7000, 8000, 9000
]
window_ms = 1000
max_requests = 4

result = max_requests_in_window(timestamps, window_ms, max_requests)
print(result)
```

**Output:**
```
4
```

**Explanation:**
- Window [5000, 6000]: Contains [5000, 5100, 5200, 5300] = 4 requests (at limit)
- This is the maximum across all windows

## Constraints

- `1 <= len(timestamps) <= 10^6`
- `0 <= timestamps[i] <= 10^12` (milliseconds since epoch)
- Timestamps are sorted in ascending order
- `1 <= window_ms <= 10^6`
- `1 <= max_requests <= 10^4`
- No duplicate timestamps (each request has unique timestamp)

## Key Concepts

- **Sliding Window**: Time-based window that moves continuously
- **Binary Search**: Efficiently find window boundaries in sorted array
- **Two Pointers**: Alternative O(n) approach for sliding window
- **Rate Limiting Algorithms**: Token bucket, leaky bucket, sliding window
- **Distributed Systems**: Handling rate limits across multiple servers

## Approach Hints

### For `max_requests_in_window`:
**Approach 1: Two Pointers (Optimal)**
- Use two pointers: left and right
- For each right pointer position, move left pointer until window is valid
- Track maximum window size seen
- Time: O(n), Space: O(1)

**Approach 2: Binary Search per Position**
- For each timestamp, binary search for leftmost timestamp in window
- Count requests in that window
- Time: O(n log n), Space: O(1)

### For `earliest_next_request`:
**Approach: Binary Search + Sliding Window**
1. Count requests in window [current_time - window_ms, current_time]
2. If count < max_requests, return current_time (can make request now)
3. Otherwise, find the oldest request in window
4. Return oldest_request_time + window_ms (when oldest falls out of window)
5. Use binary search to find window boundaries efficiently
- Time: O(log n), Space: O(1)

### For `find_safe_rate_limit`:
**Approach: Binary Search on Answer + Sliding Window Check**
1. Binary search on the rate limit value [1, len(timestamps)]
2. For each candidate limit, check if any window exceeds it
3. Use two-pointer sliding window to check all windows efficiently
4. Find maximum limit where no window exceeds it
- Time: O(n log n), Space: O(1)

**Alternative: Direct Calculation**
1. Use two-pointer technique to find maximum requests in any window
2. This value is the minimum safe rate limit
- Time: O(n), Space: O(1)

## Why This Matters for AI Engineering

At ELGO AI, rate limiting is critical for:

1. **Cost Control**: LLM API calls are expensive; prevent abuse
2. **Fair Usage**: Ensure no single customer monopolizes resources
3. **SLA Compliance**: Guarantee service availability to all tiers
4. **Billing Accuracy**: Track usage for metered billing
5. **Capacity Planning**: Analyze traffic patterns to optimize infrastructure
6. **DDoS Protection**: Detect and mitigate attack patterns
7. **Multi-tenancy**: Isolate customer workloads in shared infrastructure

Real production scenarios:
- Enterprise customer: 10,000 requests/minute
- Pro tier: 1,000 requests/minute
- Free tier: 100 requests/minute
- Burst allowance: 2x rate for 10 seconds

## Step-by-Step Example

Let's trace `max_requests_in_window` with two-pointer approach:

```python
timestamps = [1000, 1500, 2000, 2500, 3000]
window_ms = 1500
```

**Two-Pointer Sliding Window:**
```
left=0, right=0: window=[1000, 2500], count=?
- Check: 1000 + 1500 = 2500
- All timestamps in [1000, 2500): [1000, 1500, 2000] = 3 requests

left=0, right=1: window=[1000, 3000]
- Check: 1500 + 1500 = 3000
- Timestamps in [1000, 3000): [1000, 1500, 2000, 2500] = 4 requests

left=0, right=2: window=[1000, 3500]
- Check: 2000 + 1500 = 3500
- Timestamps in [1000, 3500): [1000, 1500, 2000, 2500, 3000] = 5 requests

left=1, right=2: window=[1500, 3500]
- Move left pointer: 1000 is outside window [2000-1500, 2000] = [500, 2000]
- Continue...

Maximum: 5 requests
```

## Expected Time Complexity

| Function | Optimal | Acceptable |
|----------|---------|------------|
| `max_requests_in_window` | O(n) two-pointer | O(n log n) binary search |
| `earliest_next_request` | O(log n) binary search | O(n) linear scan |
| `find_safe_rate_limit` | O(n) direct calculation | O(n log n) binary search on answer |

## Expected Space Complexity

All functions: **O(1)** - constant extra space (not counting input)

## Testing Your Solution

```python
# Test max_requests_in_window
timestamps1 = [1000, 1200, 1500, 2000, 2200, 2800, 3100]
assert max_requests_in_window(timestamps1, 2000, 5) == 5

timestamps2 = [1000, 2000, 3000, 4000, 5000]
assert max_requests_in_window(timestamps2, 1500, 3) == 1

timestamps3 = [1000, 1100, 1200, 1300, 1400]
assert max_requests_in_window(timestamps3, 500, 5) == 5

# Test earliest_next_request
timestamps4 = [1000, 1100, 1200, 1300, 1400]
assert earliest_next_request(timestamps4, 1000, 5, 1400) == 2000

timestamps5 = [1000, 1500, 2000]
assert earliest_next_request(timestamps5, 2000, 5, 2000) == 2000

# Test find_safe_rate_limit
timestamps6 = [1000, 1500, 2000, 2500, 3000, 3500]
assert find_safe_rate_limit(timestamps6, 2000) == 4

timestamps7 = [1000, 1001, 1002, 1003, 1004]
assert find_safe_rate_limit(timestamps7, 1000) == 5

print("✅ All tests passed!")
```

## Bonus Challenges

### Challenge 1: Multi-Window Rate Limiting
Implement rate limiting with multiple constraints:
- 10 requests per second
- 100 requests per minute
- 1000 requests per hour

```python
def check_multi_window_limit(
    timestamps: list[int],
    limits: list[tuple[int, int]],  # [(window_ms, max_requests), ...]
    current_time: int
) -> tuple[bool, int]:
    """
    Returns: (is_allowed, wait_time_ms)
    """
    pass
```

### Challenge 2: Distributed Rate Limiting
In a distributed system with N servers, each server has its own timestamp log. Find the global maximum request rate.

```python
def distributed_max_requests(
    server_timestamps: list[list[int]],  # List of timestamp lists (one per server)
    window_ms: int
) -> int:
    """
    Merge distributed logs and find global maximum.
    Hint: Use merge-sort approach or heap.
    """
    pass
```

### Challenge 3: Predictive Rate Limiting
Given historical timestamps, predict if the next request at time `t` would violate rate limit.

```python
def predict_rate_limit_violation(
    timestamps: list[int],
    window_ms: int,
    max_requests: int,
    next_request_time: int
) -> bool:
    """
    Returns: True if next request would violate limit
    """
    pass
```

### Challenge 4: Dynamic Rate Limit Adjustment
Implement adaptive rate limiting that increases limits during low-traffic periods and decreases during high traffic.

```python
def adaptive_rate_limit(
    timestamps: list[int],
    base_limit: int,
    window_ms: int,
    load_threshold: float  # 0.0 to 1.0
) -> int:
    """
    Returns: Adjusted rate limit based on current load
    """
    pass
```

## Common Pitfalls

❌ **Don't:**
- Use nested loops O(n²) when sliding window is O(n)
- Forget to handle edge cases (empty timestamps, single request)
- Ignore timestamp precision (milliseconds vs seconds)
- Assume uniform request distribution
- Forget about window boundary conditions (inclusive/exclusive)

✅ **Do:**
- Leverage sorted array property for binary search
- Use two-pointer technique for optimal O(n) solution
- Handle edge cases explicitly
- Consider off-by-one errors with window boundaries
- Test with burst traffic patterns
- Document time complexity analysis

## Real-World Extensions

In production, you'd also implement:

1. **Token Bucket Algorithm**: Allow burst traffic with token accumulation
2. **Leaky Bucket Algorithm**: Smooth out bursty traffic
3. **Redis-based Distributed Limiting**: Shared state across servers
4. **Rate Limit Headers**: Return `X-RateLimit-Remaining` and `X-RateLimit-Reset`
5. **Hierarchical Limits**: Per-user, per-organization, per-endpoint
6. **Cost-based Limiting**: Different weights for different operations

## Related LeetCode Problems

- Sliding Window Maximum (Hard)
- Time Based Key-Value Store (Medium)
- Logger Rate Limiter (Easy)
- Design Hit Counter (Medium)
- Find K Closest Elements (Medium)

## Interview Tips

When solving this in an interview:

1. **Clarify requirements**: Ask about window type (sliding vs fixed), timestamp format
2. **Start simple**: Implement brute force O(n²) first if needed
3. **Optimize iteratively**: Explain how to improve to O(n log n) then O(n)
4. **Discuss tradeoffs**: Time vs space, accuracy vs performance
5. **Mention distributed systems**: Show awareness of real-world constraints
6. **Test edge cases**: Empty input, single request, all requests in same window

Good luck! This problem tests your understanding of:
- Binary search optimization
- Sliding window technique
- Time-based algorithms
- Distributed systems concepts
- Production system design
