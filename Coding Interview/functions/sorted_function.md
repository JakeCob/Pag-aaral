# Python `sorted()` Function

The `sorted()` function in Python is a built-in function that returns a new sorted list from any iterable.

## Basic Syntax
```python
sorted(iterable, key=None, reverse=False)
```

## Key Features

### Returns a new list
Unlike `list.sort()` which modifies in-place, `sorted()` creates a new sorted list:
```python
numbers = [3, 1, 4, 1, 5]
result = sorted(numbers)  # [1, 1, 3, 4, 5]
# Original list unchanged
```

### Works with any iterable
```python
sorted("hello")           # ['e', 'h', 'l', 'l', 'o']
sorted({3, 1, 4})        # [1, 3, 4]
sorted((5, 2, 8))        # [2, 5, 8]
```

## The `key` Parameter
The most powerful feature - lets you specify a custom sorting criterion:

```python
# Sort by string length
words = ["apple", "pie", "banana"]
sorted(words, key=len)  # ['pie', 'apple', 'banana']

# Sort by absolute value
numbers = [-4, -1, 3, -2]
sorted(numbers, key=abs)  # [-1, -2, 3, -4]

# Sort tuples by second element
pairs = [(1, 'b'), (2, 'a'), (3, 'c')]
sorted(pairs, key=lambda x: x[1])  # [(2, 'a'), (1, 'b'), (3, 'c')]
```

## Common Use Cases

### Sort dictionaries by values
```python
freq = {'a': 3, 'b': 1, 'c': 2}
sorted(freq.items(), key=lambda x: x[1])  # [('b', 1), ('c', 2), ('a', 3)]
```

### Sort strings alphabetically (useful for anagrams!)
```python
word = "listen"
sorted(word)  # ['e', 'i', 'l', 'n', 's', 't']
''.join(sorted(word))  # 'eilnst'

# Anagrams have identical sorted character sequences
sorted("listen")  # ['e', 'i', 'l', 'n', 's', 't']
sorted("silent")  # ['e', 'i', 'l', 'n', 's', 't']
```

### Multi-level sorting
```python
students = [('Alice', 25), ('Bob', 20), ('Charlie', 25)]
# Sort by age, then by name
sorted(students, key=lambda x: (x[1], x[0]))
```

### Reverse order
```python
sorted([3, 1, 4], reverse=True)  # [4, 3, 1]
```

## Sorting Dictionaries with Nested Values

### When values are dictionaries
```python
# Dictionary where values are dictionaries
students = {
    'alice': {'age': 25, 'grade': 90},
    'bob': {'age': 20, 'grade': 85},
    'charlie': {'age': 23, 'grade': 95}
}

# Sort by age
sorted(students.items(), key=lambda x: x[1]['age'])
# [('bob', {'age': 20, 'grade': 85}),
#  ('charlie', {'age': 23, 'grade': 95}),
#  ('alice', {'age': 25, 'grade': 90})]

# Sort by grade (descending)
sorted(students.items(), key=lambda x: x[1]['grade'], reverse=True)
# [('charlie', {'age': 23, 'grade': 95}),
#  ('alice', {'age': 25, 'grade': 90}),
#  ('bob', {'age': 20, 'grade': 85})]
```

### Multiple levels of nesting
```python
data = {
    'user1': {'info': {'score': 100, 'level': 5}},
    'user2': {'info': {'score': 150, 'level': 3}},
    'user3': {'info': {'score': 120, 'level': 4}}
}

# Sort by deeply nested score
sorted(data.items(), key=lambda x: x[1]['info']['score'])
# [('user1', ...), ('user3', ...), ('user2', ...)]
```

### Multi-criteria sorting with nested dictionaries
```python
students = {
    'alice': {'age': 25, 'grade': 90},
    'bob': {'age': 25, 'grade': 85},
    'charlie': {'age': 23, 'grade': 95}
}

# Sort by age first, then by grade
sorted(students.items(), key=lambda x: (x[1]['age'], x[1]['grade']))
# [('charlie', ...), ('bob', ...), ('alice', ...)]
```

### Converting back to dictionary
```python
# If you want a dictionary result instead of list of tuples
result = dict(sorted(students.items(), key=lambda x: x[1]['grade']))
```

**Key pattern**: `lambda x: x[1]['nested_key']` where `x[1]` is the value (the nested dict) and `['nested_key']` accesses the specific field you want to sort by.

## Complexity
- **Time**: O(n log n) - uses Timsort algorithm
- **Space**: O(n) - creates new list
