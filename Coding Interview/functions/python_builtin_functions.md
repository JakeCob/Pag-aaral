# Essential Python Built-in Functions for Coding Interviews

## Table of Contents
- [Collections & Data Structures](#collections--data-structures)
- [String Methods](#string-methods)
- [List/Iterable Functions](#listiterable-functions)
- [Math Functions](#math-functions)
- [Set Operations](#set-operations)
- [Type Conversion](#type-conversion)
- [Range & Iteration](#range--iteration)
- [Built-in Constants](#built-in-constants)
- [Comprehensions](#comprehensions)

---

## Collections & Data Structures

### `collections.Counter`
Count element frequencies in an iterable.

```python
from collections import Counter

# Create counter from iterable
counter = Counter("hello")  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
counter = Counter([1, 1, 2, 3, 3, 3])  # Counter({3: 3, 1: 2, 2: 1})

# Useful methods
counter.most_common(2)      # [('l', 2), ('h', 1)] - top 2 elements
counter['x']                # Returns 0 for missing keys (no KeyError)
counter.elements()          # Iterator of elements repeating count times
counter1 + counter2         # Add counts
counter1 - counter2         # Subtract counts (keep only positive)
```

**Use cases**: Frequency counting, finding most common elements, anagram detection

---

### `collections.defaultdict`
Dictionary that provides default values for missing keys.

```python
from collections import defaultdict

# Common default types
dd = defaultdict(list)      # Creates empty list for missing keys
dd = defaultdict(int)       # Creates 0 for missing keys
dd = defaultdict(set)       # Creates empty set for missing keys

# Example usage
graph = defaultdict(list)
graph['a'].append('b')      # No KeyError, automatically creates list
graph['a'].append('c')      # graph = {'a': ['b', 'c']}

# Frequency counting
freq = defaultdict(int)
for char in "hello":
    freq[char] += 1         # No need to check if key exists
```

**Use cases**: Graphs (adjacency lists), grouping, frequency counting

---

### `collections.deque`
Double-ended queue with O(1) operations at both ends.

```python
from collections import deque

dq = deque([1, 2, 3])

# Add elements
dq.append(4)        # Right side: [1, 2, 3, 4]
dq.appendleft(0)    # Left side: [0, 1, 2, 3, 4]

# Remove elements
dq.pop()            # Remove from right: O(1)
dq.popleft()        # Remove from left: O(1) - much faster than list.pop(0)

# Other operations
dq.rotate(1)        # Rotate right: [4, 1, 2, 3]
dq.rotate(-1)       # Rotate left: [2, 3, 4, 1]
dq.extend([5, 6])   # Add multiple elements to right
dq.extendleft([0])  # Add multiple elements to left (reversed)

# Use as queue
dq.append(x)        # Enqueue
dq.popleft()        # Dequeue
```

**Use cases**: BFS, sliding window, queue implementation

**Time Complexity**:
- `append/appendleft/pop/popleft`: O(1)
- `list.pop(0)`: O(n) ❌ Use `deque.popleft()` instead ✅

---

### `heapq` - Min Heap
Priority queue implementation using min heap.

```python
import heapq

heap = []

# Add elements
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)     # heap maintains min-heap property

# Remove smallest element
smallest = heapq.heappop(heap)  # Returns 1, O(log n)

# Convert list to heap in-place
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)         # O(n)

# Get k largest/smallest elements
heapq.nlargest(3, [1, 2, 3, 4, 5])   # [5, 4, 3]
heapq.nsmallest(3, [1, 2, 3, 4, 5])  # [1, 2, 3]

# Peek at smallest without removing
smallest = heap[0]          # O(1)

# Max heap (use negative values)
max_heap = []
heapq.heappush(max_heap, -x)    # Push negative
max_val = -heapq.heappop(max_heap)  # Pop and negate

# Heap with custom objects (use tuples)
heapq.heappush(heap, (priority, value))
```

**Use cases**: Top K elements, merge sorted lists, Dijkstra's algorithm, median finding

**Time Complexity**:
- `heappush/heappop`: O(log n)
- `heapify`: O(n)
- `nlargest/nsmallest`: O(n log k) where k is count

---

## String Methods

### String Manipulation

```python
# Case conversion
s.lower()           # Convert to lowercase
s.upper()           # Convert to uppercase
s.title()           # "hello world" -> "Hello World"
s.capitalize()      # "hello" -> "Hello"
s.swapcase()        # Swap case of all characters

# Split and join
s.split()           # Split by whitespace: "a b c" -> ['a', 'b', 'c']
s.split(',')        # Split by delimiter: "a,b,c" -> ['a', 'b', 'c']
s.split(',', 1)     # Split with max splits: "a,b,c" -> ['a', 'b,c']
','.join(['a', 'b', 'c'])  # Join with delimiter: 'a,b,c'

# Strip whitespace
s.strip()           # Remove leading/trailing whitespace
s.lstrip()          # Remove leading whitespace
s.rstrip()          # Remove trailing whitespace
s.strip('.')        # Remove specific characters

# Search
s.find('x')         # Return index or -1 if not found
s.index('x')        # Return index or raise ValueError
s.count('x')        # Count occurrences
s.startswith('pre') # Check prefix
s.endswith('suf')   # Check suffix

# Replace
s.replace('old', 'new')      # Replace all occurrences
s.replace('old', 'new', 1)   # Replace first occurrence only
```

### String Checking Methods

```python
s.isalnum()         # True if all characters are alphanumeric
s.isalpha()         # True if all characters are alphabetic
s.isdigit()         # True if all characters are digits
s.isnumeric()       # True if all characters are numeric
s.islower()         # True if all cased characters are lowercase
s.isupper()         # True if all cased characters are uppercase
s.isspace()         # True if all characters are whitespace
```

**Use cases**: String parsing, validation, palindrome checking, anagram detection

---

## List/Iterable Functions

### `enumerate()`
Get index and value when iterating.

```python
for i, val in enumerate(['a', 'b', 'c']):
    print(i, val)
# 0 a
# 1 b
# 2 c

# Start index at different number
for i, val in enumerate(['a', 'b', 'c'], start=1):
    print(i, val)
# 1 a
# 2 b
# 3 c
```

---

### `zip()`
Combine multiple iterables element-wise.

```python
# Basic usage
list(zip([1, 2, 3], ['a', 'b', 'c']))
# [(1, 'a'), (2, 'b'), (3, 'c')]

# Multiple iterables
list(zip([1, 2], ['a', 'b'], [True, False]))
# [(1, 'a', True), (2, 'b', False)]

# Stops at shortest iterable
list(zip([1, 2, 3], ['a', 'b']))
# [(1, 'a'), (2, 'b')]

# Unzip using zip with *
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
nums, chars = zip(*pairs)
# nums = (1, 2, 3), chars = ('a', 'b', 'c')

# Create dictionary from two lists
dict(zip(['a', 'b', 'c'], [1, 2, 3]))
# {'a': 1, 'b': 2, 'c': 3}
```

**Use cases**: Parallel iteration, creating dictionaries, matrix transposition

---

### `map()`
Apply function to all elements.

```python
# Convert strings to integers
list(map(int, ['1', '2', '3']))  # [1, 2, 3]

# Apply function to multiple iterables
list(map(lambda x, y: x + y, [1, 2, 3], [4, 5, 6]))
# [5, 7, 9]

# Use with built-in functions
list(map(abs, [-1, -2, 3]))  # [1, 2, 3]
list(map(str.upper, ['a', 'b', 'c']))  # ['A', 'B', 'C']
```

---

### `filter()`
Filter elements based on condition.

```python
# Keep only positive numbers
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# Filter out None values
list(filter(None, [0, 1, False, 2, '', 3]))  # [1, 2, 3]

# Keep only even numbers
list(filter(lambda x: x % 2 == 0, range(10)))  # [0, 2, 4, 6, 8]
```

---

### `reversed()`
Return reversed iterator.

```python
list(reversed([1, 2, 3]))  # [3, 2, 1]
list(reversed('hello'))    # ['o', 'l', 'l', 'e', 'h']

# Reverse string
''.join(reversed('hello'))  # 'olleh'

# Note: Can also use slicing
[1, 2, 3][::-1]  # [3, 2, 1]
'hello'[::-1]    # 'olleh'
```

---

### `any()` / `all()`
Boolean checks on iterables.

```python
# any() - True if at least one element is True
any([False, False, True])   # True
any([False, False, False])  # False
any([])                     # False (empty iterable)

# all() - True if all elements are True
all([True, True, True])     # True
all([True, False, True])    # False
all([])                     # True (empty iterable)

# Common use cases
any(x > 0 for x in [-1, -2, 3])      # True
all(x > 0 for x in [1, 2, 3])        # True
all(c.isdigit() for c in "12345")    # True
```

---

## Math Functions

### Basic Math

```python
# Sum, min, max
sum([1, 2, 3])          # 6
sum([1, 2, 3], 10)      # 16 (with start value)
min([1, 2, 3])          # 1
max([1, 2, 3])          # 3
min('hello')            # 'e'
max('hello')            # 'o'

# With key function
min(['apple', 'pie', 'banana'], key=len)  # 'pie'
max([1, -5, 3], key=abs)                  # -5

# Absolute value
abs(-5)                 # 5
abs(-3.14)              # 3.14

# Power
pow(2, 3)               # 8 (2^3)
pow(2, 3, 5)            # 3 (2^3 % 5) - modular exponentiation
2 ** 3                  # 8 (alternative syntax)

# Division with quotient and remainder
divmod(10, 3)           # (3, 1) - quotient and remainder
q, r = divmod(10, 3)    # q=3, r=1

# Round
round(3.14159, 2)       # 3.14
round(2.5)              # 2 (banker's rounding - to nearest even)
round(3.5)              # 4
```

### Advanced Math (import math)

```python
import math

# Constants
math.pi                 # 3.141592653589793
math.e                  # 2.718281828459045

# Common functions
math.sqrt(16)           # 4.0
math.ceil(3.2)          # 4
math.floor(3.8)         # 3
math.factorial(5)       # 120
math.gcd(12, 18)        # 6
math.lcm(12, 18)        # 36 (Python 3.9+)

# Logarithms
math.log(8, 2)          # 3.0 (log base 2)
math.log10(100)         # 2.0
math.log(2.718281828)   # 1.0 (natural log)

# Trigonometry
math.sin(math.pi/2)     # 1.0
math.cos(0)             # 1.0
```

---

## Set Operations

```python
set1 = {1, 2, 3}
set2 = {2, 3, 4}

# Intersection (common elements)
set1 & set2             # {2, 3}
set1.intersection(set2) # {2, 3}

# Union (all elements)
set1 | set2             # {1, 2, 3, 4}
set1.union(set2)        # {1, 2, 3, 4}

# Difference (in set1 but not set2)
set1 - set2             # {1}
set1.difference(set2)   # {1}

# Symmetric difference (in either but not both)
set1 ^ set2             # {1, 4}
set1.symmetric_difference(set2)  # {1, 4}

# Subset/superset checks
{1, 2}.issubset({1, 2, 3})      # True
{1, 2, 3}.issuperset({1, 2})    # True

# Add/remove elements
s = {1, 2, 3}
s.add(4)                # {1, 2, 3, 4}
s.remove(2)             # {1, 3, 4} - raises KeyError if not found
s.discard(5)            # No error if not found
s.pop()                 # Remove and return arbitrary element
s.clear()               # Remove all elements
```

**Use cases**: Finding duplicates, unique elements, set operations

---

## Type Conversion

```python
# To integer
int('123')              # 123
int('1010', 2)          # 10 (binary to decimal)
int('FF', 16)           # 255 (hex to decimal)
int(3.14)               # 3 (truncates)

# To float
float('3.14')           # 3.14
float('inf')            # inf
float('-inf')           # -inf

# To string
str(123)                # '123'
str([1, 2, 3])          # '[1, 2, 3]'

# To list/tuple/set
list('hello')           # ['h', 'e', 'l', 'l', 'o']
tuple([1, 2, 3])        # (1, 2, 3)
set([1, 2, 2, 3])       # {1, 2, 3}

# To dictionary
dict([('a', 1), ('b', 2)])  # {'a': 1, 'b': 2}

# To boolean
bool(0)                 # False
bool(1)                 # True
bool([])                # False
bool([1])               # True
```

---

## Range & Iteration

### `range()`
Generate sequence of numbers.

```python
# range(stop)
list(range(5))          # [0, 1, 2, 3, 4]

# range(start, stop)
list(range(1, 5))       # [1, 2, 3, 4]

# range(start, stop, step)
list(range(0, 10, 2))   # [0, 2, 4, 6, 8]
list(range(10, 0, -1))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Common patterns
for i in range(len(arr)):           # Iterate with index
    print(i, arr[i])

for i in range(len(arr) - 1, -1, -1):  # Reverse iteration
    print(arr[i])
```

---

### `len()`
Get length of sequence.

```python
len([1, 2, 3])          # 3
len('hello')            # 5
len({1, 2, 3})          # 3
len({'a': 1, 'b': 2})   # 2
```

---

### `iter()` / `next()`
Manual iteration control.

```python
it = iter([1, 2, 3])
next(it)                # 1
next(it)                # 2
next(it)                # 3
next(it, 'done')        # 'done' (default value when exhausted)
```

---

## Built-in Constants

```python
# Infinity
float('inf')            # Positive infinity
float('-inf')           # Negative infinity

# Usage in comparisons
min_val = float('inf')
max_val = float('-inf')

# Check for infinity
import math
math.isinf(float('inf'))  # True

# Not a Number
float('nan')
math.isnan(float('nan'))  # True

# Boolean values
True                    # 1
False                   # 0

# None
None                    # null/nil equivalent
```

---

## Comprehensions

### List Comprehensions

```python
# Basic syntax: [expression for item in iterable if condition]

# Square numbers
[x**2 for x in range(5)]                    # [0, 1, 4, 9, 16]

# Filter with condition
[x for x in range(10) if x % 2 == 0]        # [0, 2, 4, 6, 8]

# Multiple conditions
[x for x in range(20) if x % 2 == 0 if x % 3 == 0]  # [0, 6, 12, 18]

# Nested loops
[(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# With if-else (ternary)
[x if x > 0 else 0 for x in [-1, 2, -3, 4]]  # [0, 2, 0, 4]

# Flatten nested list
matrix = [[1, 2], [3, 4], [5, 6]]
[val for row in matrix for val in row]      # [1, 2, 3, 4, 5, 6]
```

---

### Dictionary Comprehensions

```python
# Basic syntax: {key: value for item in iterable if condition}

# Square numbers
{x: x**2 for x in range(5)}                 # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Swap keys and values
original = {'a': 1, 'b': 2, 'c': 3}
{v: k for k, v in original.items()}         # {1: 'a', 2: 'b', 3: 'c'}

# Filter dictionary
{k: v for k, v in original.items() if v > 1}  # {'b': 2, 'c': 3}

# From two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
{k: v for k, v in zip(keys, values)}        # {'a': 1, 'b': 2, 'c': 3}
```

---

### Set Comprehensions

```python
# Basic syntax: {expression for item in iterable if condition}

{x**2 for x in range(5)}                    # {0, 1, 4, 9, 16}
{x % 3 for x in range(10)}                  # {0, 1, 2}
```

---

### Generator Expressions

```python
# Like list comprehension but with () - lazy evaluation
gen = (x**2 for x in range(5))              # Generator object
list(gen)                                    # [0, 1, 4, 9, 16]

# Memory efficient for large datasets
sum(x**2 for x in range(1000000))           # No list created in memory

# Can only iterate once
gen = (x for x in range(3))
list(gen)                                    # [0, 1, 2]
list(gen)                                    # [] - exhausted
```

---

## Other Useful Functions

### `ord()` / `chr()`
Convert between characters and ASCII values.

```python
ord('A')                # 65
ord('a')                # 97
chr(65)                 # 'A'
chr(97)                 # 'a'

# Use case: Caesar cipher, character arithmetic
ord('b') - ord('a')     # 1
chr(ord('a') + 1)       # 'b'
```

---

### `isinstance()` / `type()`
Check object type.

```python
isinstance(5, int)              # True
isinstance('hello', str)        # True
isinstance([1, 2], list)        # True
isinstance(5, (int, float))     # True (check multiple types)

type(5)                         # <class 'int'>
type([]) == list                # True
```

---

### `id()`
Get object's memory address (identity).

```python
a = [1, 2, 3]
b = a
id(a) == id(b)          # True - same object
c = [1, 2, 3]
id(a) == id(c)          # False - different objects
```

---

### `eval()` / `exec()`
Execute Python code from strings (use cautiously!).

```python
eval('2 + 3')           # 5
eval('[1, 2, 3]')       # [1, 2, 3]

# exec for statements
exec('x = 5')           # Creates variable x
```

**⚠️ Warning**: Never use with untrusted input!

---

## Quick Reference Summary

### Most Common for Interviews
1. `sorted()`, `reversed()`, `enumerate()`, `zip()`
2. `Counter`, `defaultdict`, `deque`
3. `heapq` for heap operations
4. `any()`, `all()`, `sum()`, `min()`, `max()`
5. String methods: `split()`, `join()`, `strip()`, `isalnum()`, `isdigit()`
6. List comprehensions and generator expressions
7. Set operations: `&`, `|`, `-`, `^`

### Time Complexities to Remember
- `list.append()`: O(1)
- `list.pop()`: O(1), but `list.pop(0)`: O(n)
- `deque.popleft()`: O(1) ✅
- `heappush/heappop`: O(log n)
- `sorted()`: O(n log n)
- `set` operations: O(1) average for add/remove/contains
- `dict` operations: O(1) average for get/set/delete
