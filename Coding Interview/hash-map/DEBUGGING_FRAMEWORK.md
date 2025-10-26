# Debugging Framework: How to Find and Fix Your Mistakes

## Why Your Anagram Solution Failed (Detailed Analysis)

Let's use your solution as a learning example.

---

## 🔍 Step-by-Step Debugging Process

### Step 1: Identify the Bug (Test Case Analysis)

**Input:** `["eat", "tea", "tan", "ate", "nat", "bat"]`

**Your Output:** `[['eat', 'tea', 'ate'], ['tan'], ['nat'], ['bat']]`
**Expected:** `[['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]`

**What went wrong?** `'tan'` and `'nat'` are in separate groups but should be together.

---

### Step 2: Trace Through Your Logic

Let's manually trace what your code does:

```python
# Your code structure:
start = 0
not_done = [0]
done = []
anagrams = []

# Iteration 1: start = 0, current_str = "eat"
current_str = "eat"
letters = ['e', 'a', 't']
anagrams = [["eat"]]  # Start new group

# Check other strings:
for end in range(1, 6):  # Check "tea", "tan", "ate", "nat", "bat"

    # end = 1, other_str = "tea"
    for letter in ['e', 'a', 't']:
        if 'e' not in "tea": False, continue
        if 'a' not in "tea": False, continue
        if 't' not in "tea": False, continue
    # All letters found, but your logic says:
    # "if letter not in other_str ... add to not_done"
    # Since all letters WERE found, nothing happens
    # So "tea" gets added to anagrams[0]

    # end = 2, other_str = "tan"
    for letter in ['e', 'a', 't']:
        if 'e' not in "tan": True! → Add 2 to not_done, break
    # not_done = [0, 2]
    # "tan" is NOT added to anagrams[0]
```

**The Bug:** Your logic is inverted!
- You add to `not_done` when letters are NOT found
- You DON'T add to group when letters ARE found
- But you also don't have logic to ADD when all letters match!

---

### Step 3: What Should Happen

For "tan" and "nat" to be grouped:

```python
# When checking if "tan" and "nat" are anagrams:
# 1. Sort both: "tan" → "ant", "nat" → "ant"
# 2. Same sorted form → They're anagrams!
# 3. Add both to same group

# OR using character count:
# "tan" → {t:1, a:1, n:1}
# "nat" → {n:1, a:1, t:1}
# Same counts → They're anagrams!
```

**What you tried to do:** Check if all letters exist
**Problem:**
- `"tab"` has letters 't', 'a', 'b'
- `"tan"` has letters 't', 'a', 'n'
- `'t'` in "tan"? Yes
- `'a'` in "tan"? Yes
- `'b'` in "tan"? No

Your code would think they're NOT anagrams (correct), BUT your logic is backwards and doesn't check character COUNTS.

---

### Step 4: The Correct Approach

```python
def group_anagrams(strs):
    from collections import defaultdict

    groups = defaultdict(list)

    for s in strs:
        # Create a unique key for all anagrams
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

**Why this works:**
```
"eat" → sorted → "aet" → groups["aet"] = ["eat"]
"tea" → sorted → "aet" → groups["aet"] = ["eat", "tea"]
"tan" → sorted → "ant" → groups["ant"] = ["tan"]
"ate" → sorted → "aet" → groups["aet"] = ["eat", "tea", "ate"]
"nat" → sorted → "ant" → groups["ant"] = ["tan", "nat"]
"bat" → sorted → "abt" → groups["abt"] = ["bat"]

Result: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

---

## 🎯 General Debugging Framework

### When Your Solution Fails:

#### 1. Write Down Expected Behavior
```
For input ["tan", "nat"]:
Expected: They should be in the same group
Reason: They have the same letters (t, a, n)
```

#### 2. Trace Through Your Code Manually
```
Step 1: current_str = "tan"
Step 2: letters = ['t', 'a', 'n']
Step 3: Check "nat"
  - 't' in "nat"? Yes
  - 'a' in "nat"? Yes
  - 'n' in "nat"? Yes
Step 4: What should happen? Add "nat" to group
Step 5: What actually happens? [TRACE YOUR CODE HERE]
```

#### 3. Identify the Gap
```
Expected: "nat" added to group with "tan"
Actual: "nat" NOT added
Gap: Missing logic to add when all letters match
```

#### 4. Check Your Logic
```
Your condition:
  if letter not in other_str:
    not_done.append(end)
    break

This says: "If ANY letter is missing, mark as not_done"

Problem: What happens when ALL letters are present?
→ Nothing! No code to handle this case!
```

#### 5. Simplify and Rebuild
```
Instead of tracking what's done/not done:
→ For each string, compute a "signature"
→ Group all strings with same signature
→ Signature = sorted string (or character count)
```

---

## 🧪 Testing Strategy

### Always Test These Cases:

#### 1. Empty Input
```python
assert group_anagrams([]) == []
```

#### 2. Single Element
```python
assert group_anagrams(["a"]) == [["a"]]
```

#### 3. No Anagrams (All Different)
```python
input = ["a", "b", "c"]
result = group_anagrams(input)
assert len(result) == 3  # Three separate groups
```

#### 4. All Anagrams (All Same)
```python
input = ["abc", "bca", "cab"]
result = group_anagrams(input)
assert len(result) == 1  # One group
assert len(result[0]) == 3  # Three strings in group
```

#### 5. Mixed
```python
input = ["eat", "tea", "tan", "ate", "nat", "bat"]
result = group_anagrams(input)
# Should have 3 groups
# "eat"/"tea"/"ate" together
# "tan"/"nat" together
# "bat" alone
```

#### 6. Duplicates
```python
input = ["eat", "eat", "tea"]
result = group_anagrams(input)
assert len(result) == 1
assert len(result[0]) == 3  # All three in one group
```

#### 7. Empty String
```python
assert group_anagrams([""]) == [[""]]
```

---

## 🚨 Common Logic Errors

### Error 1: Inverted Conditions
```python
# ❌ WRONG
if letter not in other_str:
    # Mark as different

# But then forgot to handle: if letter IS in other_str!
```

**Fix:** Think through BOTH branches
```python
# ✅ CORRECT
if condition:
    # Handle true case
else:
    # Handle false case
```

### Error 2: Missing Count Verification
```python
# ❌ WRONG: Just checks existence
if 'a' in "aab" and 'b' in "aab":
    # Thinks "ab" and "aab" are anagrams!

# ✅ CORRECT: Check counts
Counter("ab") == Counter("aab")  # False
```

### Error 3: Complex State Management
```python
# ❌ WRONG: Too many tracking variables
not_done = []
done = []
anagram_index = {}
i = 0
start = 0

# ✅ CORRECT: Simple data structure
groups = defaultdict(list)
```

**Rule:** If you need 3+ tracking variables, you're overcomplicating.

### Error 4: Using Exceptions for Control Flow
```python
# ❌ WRONG
try:
    start = not_done[i]
except Exception:
    break

# ✅ CORRECT
if i < len(not_done):
    start = not_done[i]
else:
    break
```

### Error 5: Not Testing Edge Cases
```python
# Always test:
- Empty: []
- Single: ["a"]
- Duplicates: ["a", "a"]
- All same: ["abc", "bca", "cab"]
- All different: ["a", "b", "c"]
```

---

## 🔧 How to Prevent These Mistakes

### 1. Start with Examples
Before coding, manually trace through an example:
```
Input: ["eat", "tea", "tan"]

What should happen?
- "eat" → sorted → "aet"
- "tea" → sorted → "aet"  (same as "eat"!)
- "tan" → sorted → "ant"  (different!)

Groups: {
  "aet": ["eat", "tea"],
  "ant": ["tan"]
}
```

### 2. Write Pseudocode First
```
1. Create empty hash map (key → group)
2. For each string:
   a. Compute signature (sorted string)
   b. Add string to group with that signature
3. Return all groups
```

### 3. Implement Incrementally
```python
# Step 1: Just create groups (don't worry about logic yet)
groups = defaultdict(list)

# Step 2: Add strings to groups (hardcode keys for now)
groups["aet"].append("eat")
groups["aet"].append("tea")

# Step 3: Now implement key computation
for s in strs:
    key = ''.join(sorted(s))
    groups[key].append(s)

# Step 4: Return result
return list(groups.values())
```

### 4. Test After Each Step
```python
# After Step 1
print(groups)  # Should be empty defaultdict

# After Step 2
print(groups)  # Should show hardcoded groups

# After Step 3
print(groups)  # Should show computed groups
```

### 5. Use Assertions
```python
# Verify assumptions
assert len(strs) > 0, "Input should not be empty"
assert all(isinstance(s, str) for s in strs), "All elements should be strings"

# Verify intermediate results
key = ''.join(sorted("eat"))
assert key == "aet", f"Expected 'aet', got {key}"
```

---

## 📊 Complexity Analysis Checklist

Before submitting, verify:

### Time Complexity
```
□ What's the outer loop? O(?)
□ What's the inner loop? O(?)
□ What operations are inside loops? O(?)
□ Are there any library calls? (sorted, max, etc.) O(?)
□ Total time: O(?)
```

For group_anagrams:
```
□ Outer loop: n strings → O(n)
□ Sorting each string: k log k → O(k log k)
□ Total: O(n * k log k) ✅
```

### Space Complexity
```
□ What data structures are used?
□ How much space does each use?
□ Total space: O(?)
```

For group_anagrams:
```
□ Hash map: stores n strings → O(n * k)
□ Total: O(n * k) ✅
```

---

## 🎓 Learning from Mistakes

### Your Anagram Solution Taught You:

1. ✅ **Pattern Recognition:** This is a grouping problem (use hash map!)
2. ✅ **Key Design:** Anagrams → same sorted string
3. ✅ **Simplicity:** 5 lines > 30 lines with bugs
4. ✅ **Testing:** Always test with examples BEFORE submitting
5. ✅ **Data Structures:** defaultdict simplifies code

### Next Time You See a Similar Problem:

1. ✅ **Recognize the pattern:** "Group items" → Hash map grouping
2. ✅ **Find the invariant:** What makes items "equivalent"?
3. ✅ **Design the key:** How to represent equivalence?
4. ✅ **Implement simply:** Use defaultdict, avoid manual tracking
5. ✅ **Test thoroughly:** Edge cases before submitting

---

## 🚀 Debugging Checklist

When your solution doesn't work:

### Step 1: Verify Understanding
- [ ] Do I understand the problem correctly?
- [ ] Do I understand what the output should be?
- [ ] Have I considered all edge cases?

### Step 2: Trace Manually
- [ ] Pick a simple example
- [ ] Write down expected result
- [ ] Trace through my code line by line
- [ ] Compare actual vs expected at each step

### Step 3: Find the Gap
- [ ] Where does actual diverge from expected?
- [ ] What's the first incorrect result?
- [ ] Why did my logic produce this?

### Step 4: Identify Root Cause
- [ ] Is my condition wrong?
- [ ] Am I missing a case?
- [ ] Is my data structure wrong?
- [ ] Am I overcomplicating?

### Step 5: Fix and Re-test
- [ ] Make minimal fix
- [ ] Test with original example
- [ ] Test with edge cases
- [ ] Verify complexity

---

## 💡 Quick Reference: "Is My Solution Right?"

### Good Signs ✅
- Code is simple (5-15 lines of logic)
- Using appropriate data structure (hash map for grouping)
- No nested loops (unless necessary)
- Handles edge cases
- Time complexity matches expected

### Bad Signs ❌
- Code is complex (30+ lines)
- Many tracking variables (3+)
- Try-except for control flow
- Fails basic test cases
- Can't explain how it works

### If You're Stuck (15+ minutes):
1. ✅ Look at the pattern guide
2. ✅ Read one solution explanation
3. ✅ Understand WHY it works
4. ✅ Close the solution
5. ✅ Implement from memory
6. ✅ Come back tomorrow and do it again

---

## 📝 Post-Mortem Template

After every mistake, fill this out:

```
Problem: Group Anagrams

What went wrong?
- Used inverted logic (checked when letters NOT found)
- Didn't check character counts
- Overcomplicated with manual tracking

What should I have done?
- Recognize grouping pattern → hash map
- Find invariant: anagrams have same sorted string
- Use sorted string as key

Key lesson learned:
- Grouping problems → hash map with creative key
- Simpler is better
- Test with examples before coding

Similar problems to practice:
- Group Shifted Strings
- Group People by Size
- Group Transactions by Date
```

---

Remember: **Mistakes are learning opportunities.** Every bug you fix teaches you something new. Keep debugging, keep learning! 🔍🐛→✅
