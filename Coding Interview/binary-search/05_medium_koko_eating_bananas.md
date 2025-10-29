# Medium: Koko Eating Bananas

## Problem Statement

Koko loves to eat bananas. There are `n` piles of bananas, the `i-th` pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses some pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, she eats all of them instead and will not eat any more bananas during that hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer `k` such that she can eat all the bananas within `h` hours.

## Function Signature

```python
def min_eating_speed(piles: list[int], h: int) -> int:
    """
    Find the minimum eating speed to finish all bananas in h hours.

    Args:
        piles: List of banana pile sizes
        h: Hours available to eat all bananas

    Returns:
        Minimum eating speed (bananas per hour)
    """
    pass
```

## Examples

### Example 1:
```python
piles = [3, 6, 7, 11]
h = 8

result = min_eating_speed(piles, h)
print(result)
```

**Output:**
```
4
```

**Explanation:**
- With speed k=4:
  - Pile 1 (3 bananas): 1 hour (eats 3)
  - Pile 2 (6 bananas): 2 hours (eats 4, then 2)
  - Pile 3 (7 bananas): 2 hours (eats 4, then 3)
  - Pile 4 (11 bananas): 3 hours (eats 4, 4, then 3)
  - Total: 1 + 2 + 2 + 3 = 8 hours ✓

- With speed k=3 (too slow):
  - Total: 1 + 2 + 3 + 4 = 10 hours ✗ (exceeds h=8)

### Example 2:
```python
piles = [30, 11, 23, 4, 20]
h = 5

result = min_eating_speed(piles, h)
print(result)
```

**Output:**
```
30
```

**Explanation:**
- With speed k=30:
  - Pile 1 (30 bananas): 1 hour
  - Pile 2 (11 bananas): 1 hour
  - Pile 3 (23 bananas): 1 hour
  - Pile 4 (4 bananas): 1 hour
  - Pile 5 (20 bananas): 1 hour
  - Total: 5 hours ✓

- Koko needs to finish in exactly 5 hours (one pile per hour), so she must eat at speed of the largest pile (30).

### Example 3:
```python
piles = [30, 11, 23, 4, 20]
h = 6

result = min_eating_speed(piles, h)
print(result)
```

**Output:**
```
23
```

**Explanation:**
- With speed k=23:
  - Pile 1 (30 bananas): 2 hours (eats 23, then 7)
  - Pile 2 (11 bananas): 1 hour
  - Pile 3 (23 bananas): 1 hour
  - Pile 4 (4 bananas): 1 hour
  - Pile 5 (20 bananas): 1 hour
  - Total: 2 + 1 + 1 + 1 + 1 = 6 hours ✓

### Example 4:
```python
piles = [1000000000]
h = 2

result = min_eating_speed(piles, h)
print(result)
```

**Output:**
```
500000000
```

**Explanation:** One pile of 1 billion bananas, 2 hours available. Speed = 1000000000 / 2 = 500000000.

## Constraints

- `1 <= piles.length <= 10^4`
- `piles.length <= h <= 10^9`
- `1 <= piles[i] <= 10^9`

## Key Concepts

- **Binary Search on Answer Space**: Search for minimum valid speed
- **Feasibility Check**: Verify if a given speed works
- **Math.ceil() for Division**: Calculate hours needed per pile
- **Minimization Problem**: Find smallest value satisfying condition

## Approach Hints

### Approach: Binary Search on Speed

**Key Insight:**
- Minimum possible speed: 1 banana/hour
- Maximum needed speed: max(piles) bananas/hour
- If speed `k` works, any speed > `k` also works (monotonic property)
- Use binary search to find the minimum `k` that works

**Algorithm:**

1. **Define search space**:
   - `left = 1` (minimum possible speed)
   - `right = max(piles)` (maximum needed speed)

2. **Binary search for minimum valid speed**:
   ```python
   while left < right:
       mid = (left + right) // 2
       if can_finish(piles, mid, h):
           right = mid  # Try smaller speed
       else:
           left = mid + 1  # Need faster speed
   return left
   ```

3. **Feasibility check** - Can finish at speed `k` within `h` hours?
   ```python
   def can_finish(piles, k, h):
       hours_needed = 0
       for pile in piles:
           hours_needed += (pile + k - 1) // k  # Ceiling division
       return hours_needed <= h
   ```

**Why ceiling division?**
- If pile = 7, k = 3: needs ceil(7/3) = 3 hours
- Formula: `ceil(a/b) = (a + b - 1) // b`
- Alternative: `import math; math.ceil(pile / k)`

**Time Complexity:** O(n log m) where n = len(piles), m = max(piles)
- Binary search: O(log m) iterations
- Each iteration checks all piles: O(n)

**Space Complexity:** O(1)

## Step-by-Step Trace

Let's trace Example 1:

```python
piles = [3, 6, 7, 11]
h = 8
```

**Binary Search Execution:**

```
Initial: left=1, right=11 (max pile)

Iteration 1:
  mid = (1 + 11) // 2 = 6
  can_finish(piles, k=6, h=8)?
    - Pile 3: ceil(3/6) = 1 hour
    - Pile 6: ceil(6/6) = 1 hour
    - Pile 7: ceil(7/6) = 2 hours
    - Pile 11: ceil(11/6) = 2 hours
    - Total: 1 + 1 + 2 + 2 = 6 hours <= 8 ✓
  Yes! Try slower speed
  right = 6

Iteration 2:
  left=1, right=6
  mid = (1 + 6) // 2 = 3
  can_finish(piles, k=3, h=8)?
    - Pile 3: ceil(3/3) = 1 hour
    - Pile 6: ceil(6/3) = 2 hours
    - Pile 7: ceil(7/3) = 3 hours
    - Pile 11: ceil(11/3) = 4 hours
    - Total: 1 + 2 + 3 + 4 = 10 hours > 8 ✗
  No! Need faster speed
  left = 4

Iteration 3:
  left=4, right=6
  mid = (4 + 6) // 2 = 5
  can_finish(piles, k=5, h=8)?
    - Pile 3: ceil(3/5) = 1 hour
    - Pile 6: ceil(6/5) = 2 hours
    - Pile 7: ceil(7/5) = 2 hours
    - Pile 11: ceil(11/5) = 3 hours
    - Total: 1 + 2 + 2 + 3 = 8 hours <= 8 ✓
  Yes! Try slower speed
  right = 5

Iteration 4:
  left=4, right=5
  mid = (4 + 5) // 2 = 4
  can_finish(piles, k=4, h=8)?
    - Pile 3: ceil(3/4) = 1 hour
    - Pile 6: ceil(6/4) = 2 hours
    - Pile 7: ceil(7/4) = 2 hours
    - Pile 11: ceil(11/4) = 3 hours
    - Total: 1 + 2 + 2 + 3 = 8 hours <= 8 ✓
  Yes! Try slower speed
  right = 4

Termination: left=4, right=4 → Exit
Return 4
```

**Answer: 4 bananas per hour** ✓

## Expected Time Complexity

- **Target:** O(n log m) where n = number of piles, m = max(piles)
- **Not Acceptable:** O(m × n) - trying every speed from 1 to max(piles)

## Expected Space Complexity

- **Target:** O(1) constant extra space

## Testing Your Solution

```python
import math

def can_finish(piles: list[int], k: int, h: int) -> bool:
    """Helper: Check if speed k allows finishing in h hours."""
    hours_needed = sum(math.ceil(pile / k) for pile in piles)
    return hours_needed <= h

def min_eating_speed(piles: list[int], h: int) -> int:
    """Find minimum eating speed."""
    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2
        if can_finish(piles, mid, h):
            right = mid
        else:
            left = mid + 1

    return left


# Test Cases
assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
assert min_eating_speed([1000000000], 2) == 500000000

# Edge Cases
assert min_eating_speed([1], 1) == 1
assert min_eating_speed([1, 1, 1, 1], 4) == 1
assert min_eating_speed([312884470], 312884469) == 2

# Large pile, many hours
assert min_eating_speed([100], 100) == 1
assert min_eating_speed([100], 50) == 2

print("✅ All tests passed!")
```

## Common Mistakes

❌ **Using floor division instead of ceiling:**
```python
# Wrong: This underestimates hours needed
hours = pile // k

# Correct: Use ceiling division
hours = math.ceil(pile / k)
# Or: hours = (pile + k - 1) // k
```

❌ **Wrong binary search bounds:**
```python
# Wrong: Starting from 0
left = 0

# Correct: Minimum speed is 1
left = 1
```

❌ **Wrong loop condition:**
```python
# Wrong: while left <= right (doesn't work for minimization)
while left <= right:
    ...

# Correct: while left < right
while left < right:
    ...
```

❌ **Not using math.ceil properly:**
```python
# Wrong: Integer division of sum
hours = sum(pile / k for pile in piles)  # Float result

# Correct: Ceil each pile individually
hours = sum(math.ceil(pile / k) for pile in piles)
```

❌ **Overflow in ceiling division formula:**
```python
# In Python this is fine, but in other languages:
hours = (pile + k - 1) // k  # Could overflow if pile is INT_MAX

# Safer alternative:
hours = pile // k + (1 if pile % k != 0 else 0)
```

## Why This Problem is Important

This problem teaches a critical pattern: **Binary Search on Answer Space**

Similar problems where you binary search for the answer:
- Capacity to Ship Packages Within D Days
- Split Array Largest Sum
- Minimize Max Distance to Gas Station
- Magnetic Force Between Two Balls
- Ugly Number III

The pattern:
1. Answer is in range [min, max]
2. If answer `x` works, all answers > `x` also work (or vice versa)
3. Binary search to find minimum/maximum valid answer
4. Each iteration checks feasibility in O(n) or O(n log n)

## Variations to Practice

### Variation 1: Maximum Eating Time
What if Koko wants to eat as slowly as possible (maximize time per pile)?

```python
def max_eating_time(piles: list[int], h: int) -> int:
    """
    Find maximum time Koko can spend on each pile.
    Note: This changes the problem significantly!
    """
    pass
```

### Variation 2: Minimum Piles
Given speed k and time h, what's the minimum number of piles needed?

```python
def min_piles_needed(total_bananas: int, k: int, h: int) -> int:
    """
    Given total bananas, speed, and hours, find min piles.
    """
    pass
```

### Variation 3: Variable Speed
Koko can change speed between piles. Find optimal strategy.

```python
def min_max_speed(piles: list[int], h: int) -> int:
    """
    Find the minimum possible maximum speed across all piles.
    This is a different optimization problem!
    """
    pass
```

## Real-World Applications

This problem models many real-world scenarios:

1. **Resource Allocation**: Distributing work across time with rate limits
2. **Manufacturing**: Production rate to meet deadline with minimal speed
3. **Data Processing**: Batch processing rate to finish within time window
4. **Network Bandwidth**: Minimum bandwidth to transfer files before deadline
5. **CPU Scheduling**: Minimum processor speed to complete tasks on time
6. **Video Streaming**: Minimum download speed for buffer-free playback

## Related Problems

**Similar Binary Search on Answer:**
- **875. Koko Eating Bananas** (This problem)
- **1011. Capacity To Ship Packages Within D Days** (Medium)
- **410. Split Array Largest Sum** (Hard)
- **774. Minimize Max Distance to Gas Station** (Hard)
- **1283. Find the Smallest Divisor Given a Threshold** (Medium)

**Similar Feasibility Check:**
- **1231. Divide Chocolate** (Hard)
- **1552. Magnetic Force Between Two Balls** (Medium)

## Interview Tips

When solving this in an interview:

1. **Recognize the pattern**:
   - "Minimum speed" → binary search on answer
   - "Within h hours" → feasibility constraint

2. **Start with examples**:
   - Work through small example by hand
   - Try different speeds to understand feasibility

3. **Explain your approach**:
   - "This is a binary search on answer space problem"
   - "I'll search for minimum valid speed between 1 and max(piles)"
   - "For each candidate speed, I'll check if we can finish in h hours"

4. **Discuss edge cases**:
   - Single pile: h must be >= pile size
   - h equals number of piles: need max(piles) speed
   - Very large pile values: mention integer overflow in other languages

5. **Complexity analysis**:
   - Clearly explain O(n log m) complexity
   - Mention m = max(piles), not length of array

6. **Optimize if needed**:
   - Could start right = sum(piles) // h (tighter upper bound)
   - Could use left = (sum(piles) + h - 1) // h (tighter lower bound)

Good luck! Master this pattern and you'll ace many medium-level binary search problems!
