# 5-DAY EMERGENCY INTERVIEW PREP PLAN
## Intensive Hash Map Crash Course

---

## ⚠️ WARNING: This is AGGRESSIVE

- **Time Required:** 3-4 hours per day (minimum)
- **Difficulty:** High intensity
- **Success Rate:** 60-70% (vs. 90%+ with 4 weeks)
- **Strategy:** Focus on most common patterns only

---

## 🎯 The Strategy

### What We're Optimizing For:
- ✅ **Breadth over depth** - Cover common patterns quickly
- ✅ **Pattern recognition** - Recognize problem types fast
- ✅ **Communication skills** - Explain clearly even if stuck
- ✅ **Partial credit** - Get 70% of solution right

### What We're Sacrificing:
- ❌ Deep understanding of advanced patterns
- ❌ Speed (you'll be slower than optimal)
- ❌ Hard problems (focus on Easy/Medium)
- ❌ Comprehensive coverage

---

## 📅 DAY 1: Foundation + Pattern 1 & 2

### Morning (1.5 hours)

#### 1. Speed-read STUDY_GUIDE.md (30 min)
Focus on:
- [ ] Hash map basics
- [ ] Pattern 1: Frequency Counting
- [ ] Pattern 2: Grouping by Property

#### 2. Watch/Read Pattern Explanations (30 min)
- [ ] Two Sum (Easy) - explanation
- [ ] Group Anagrams (Medium) - explanation

#### 3. Copy Templates to Notes (30 min)
```python
# Template 1: Frequency Counting
from collections import Counter
def pattern_frequency(arr):
    return Counter(arr)

# Template 2: Grouping
from collections import defaultdict
def pattern_grouping(items, key_function):
    groups = defaultdict(list)
    for item in items:
        key = key_function(item)
        groups[key].append(item)
    return list(groups.values())
```

### Afternoon (1.5 hours)

#### 4. Solve These Problems (1.5 hours, 30 min each)
- [ ] **Two Sum** (Easy) - Most common interview question!
- [ ] **Valid Anagram** (Easy)
- [ ] **Group Anagrams** (Medium) - Re-attempt with template

**Rules:**
- Set timer for 25 minutes
- If stuck at 15 min, look at hints
- If stuck at 20 min, look at solution
- Understand WHY it works, then close solution
- Re-implement from memory

### Evening (1 hour)

#### 5. Review & Solidify (1 hour)
- [ ] Re-do all 3 problems from memory (20 min each)
- [ ] Write down key insights for each
- [ ] Practice explaining solutions out loud

**Key Insights to Note:**
```
Two Sum:
- Pattern: Complement search
- Key insight: Store seen numbers in hash map
- Time: O(n) vs O(n²) brute force

Valid Anagram:
- Pattern: Frequency counting
- Key insight: Anagrams have same character counts
- Time: O(n) with Counter

Group Anagrams:
- Pattern: Grouping by property
- Key insight: Sorted string is the key
- Time: O(n * k log k)
```

### End of Day 1 Goal:
- ✅ Understand frequency counting
- ✅ Understand grouping pattern
- ✅ Solved 3 problems confidently
- ✅ Can explain solutions clearly

---

## 📅 DAY 2: Pattern 3 (Two-Sum) + Pattern 4 (Prefix Sum)

### Morning (1.5 hours)

#### 1. Study Patterns (30 min)
- [ ] Read Pattern 3: Two-Sum/Complement Search
- [ ] Read Pattern 4: Prefix Sum

#### 2. Copy Templates (15 min)
```python
# Template 3: Two Sum Pattern
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Template 4: Prefix Sum
def subarray_sum_k(nums, k):
    count = 0
    current_sum = 0
    prefix_sums = {0: 1}  # IMPORTANT: Base case

    for num in nums:
        current_sum += num
        if (current_sum - k) in prefix_sums:
            count += prefix_sums[current_sum - k]
        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1

    return count
```

#### 3. Watch Solution Videos (45 min)
- [ ] Two Sum - watch explanation
- [ ] Subarray Sum Equals K - watch explanation

### Afternoon (1.5 hours)

#### 4. Solve These Problems (30 min each)
- [ ] **Two Sum** (Easy) - If not done yesterday
- [ ] **Contains Duplicate** (Easy)
- [ ] **Subarray Sum Equals K** (Hard) - The one you struggled with!

**Special Focus: Subarray Sum Equals K**
This is THE problem you need to understand. Spend extra time here.

Key insights:
```
Why {0: 1} as base case?
- Handles subarrays starting from index 0
- Example: nums=[1,2,3], k=3
  - At index 1: current_sum=3, need prefix_sum=0
  - {0:1} gives us that match

Why current_sum - k?
- We want: sum[i:j] = k
- We know: sum[i:j] = prefix[j] - prefix[i]
- So: prefix[j] - prefix[i] = k
- Rearrange: prefix[i] = prefix[j] - k
- We're at j, so look for (current_sum - k)
```

### Evening (1 hour)

#### 5. Practice & Review (1 hour)
- [ ] Re-do Subarray Sum from memory (30 min)
- [ ] Trace through example step-by-step (15 min)
- [ ] Explain to rubber duck/friend (15 min)

### End of Day 2 Goal:
- ✅ Understand Two-Sum complement pattern
- ✅ Understand Prefix Sum (WHY it works)
- ✅ Can solve Subarray Sum Equals K
- ✅ Solved 6 total problems

---

## 📅 DAY 3: Pattern 5 (Sliding Window) + Speed Practice

### Morning (1.5 hours)

#### 1. Study Pattern 5 (30 min)
- [ ] Read: Sliding Window with Hash Map
- [ ] Understand: When to expand, when to shrink window

#### 2. Copy Template (15 min)
```python
# Template 5: Sliding Window
def longest_substring_without_repeating(s):
    char_index = {}
    max_len = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)

    return max_len
```

#### 3. Watch Video (45 min)
- [ ] Longest Substring Without Repeating Characters

### Afternoon (1.5 hours)

#### 4. Solve These Problems (30 min each)
- [ ] **Longest Substring Without Repeating Characters** (Medium)
- [ ] **First Unique Character in String** (Easy)
- [ ] **Intersection of Two Arrays** (Easy)

### Evening (1 hour)

#### 5. SPEED PRACTICE (1 hour)
Go back and re-solve ALL previous problems as fast as possible:
- [ ] Two Sum (Target: 10 min)
- [ ] Valid Anagram (Target: 8 min)
- [ ] Group Anagrams (Target: 15 min)
- [ ] Subarray Sum Equals K (Target: 20 min)

**Goal:** Build muscle memory and speed

### End of Day 3 Goal:
- ✅ Understand sliding window basics
- ✅ Faster at previous patterns
- ✅ Solved 9 total problems
- ✅ Can code solutions without looking

---

## 📅 DAY 4: Pattern 6 (Hash Set) + Interview Communication

### Morning (1.5 hours)

#### 1. Study Pattern 6 (30 min)
- [ ] Read: Hash Set for O(1) Lookups
- [ ] Understand: Longest Consecutive Sequence trick

#### 2. Copy Template (15 min)
```python
# Template 6: Hash Set for Sequence
def longest_consecutive(nums):
    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only process sequence starts
        if num - 1 not in num_set:
            current = num
            streak = 1

            while current + 1 in num_set:
                current += 1
                streak += 1

            longest = max(longest, streak)

    return longest
```

#### 3. Watch Video (45 min)
- [ ] Longest Consecutive Sequence

### Afternoon (1.5 hours)

#### 4. Solve These Problems (30 min each)
- [ ] **Longest Consecutive Sequence** (Hard)
- [ ] **Contains Duplicate II** (Easy)
- [ ] **Happy Number** (Easy)

### Evening (1 hour)

#### 5. COMMUNICATION PRACTICE (1 hour)

**Practice explaining solutions OUT LOUD:**

For each problem, practice this script:
```
1. "I recognize this as a [PATTERN] problem"
2. "I'll use [DATA STRUCTURE] because [REASON]"
3. "The key insight is [INSIGHT]"
4. [Code while explaining]
5. "Time complexity is O(...) because..."
6. "Let me trace through an example..."
```

Record yourself or explain to someone. This is CRITICAL for interview performance.

### End of Day 4 Goal:
- ✅ Understand Hash Set optimization
- ✅ Can explain solutions clearly
- ✅ Solved 12 total problems
- ✅ Comfortable with communication

---

## 📅 DAY 5: Mock Interview + Final Review

### Morning (2 hours)

#### 1. Mock Interview Simulation (1 hour)

**Setup:**
- Set timer for 45 minutes
- Pick 2 problems (one Easy, one Medium)
- Solve AS IF in interview
- Talk out loud the entire time

**Problems:**
- [ ] **Top K Frequent Elements** (Medium)
- [ ] **Ransom Note** (Easy)

**Evaluation:**
- Did you recognize the pattern?
- Did you communicate clearly?
- Did you test your solution?
- How long did it take?

#### 2. Review & Fix (1 hour)
- [ ] Watch solutions for problems you struggled with
- [ ] Note what you missed
- [ ] Re-implement correctly

### Afternoon (1.5 hours)

#### 3. RAPID PATTERN REVIEW (1.5 hours)

Go through all patterns and solve one problem each:

- [ ] **Pattern 1 (Frequency):** Jewels and Stones (Easy) - 10 min
- [ ] **Pattern 2 (Grouping):** Group Anagrams - 15 min
- [ ] **Pattern 3 (Two-Sum):** Two Sum - 10 min
- [ ] **Pattern 4 (Prefix Sum):** Subarray Sum = K - 20 min
- [ ] **Pattern 5 (Sliding Window):** Longest Substring - 15 min
- [ ] **Pattern 6 (Hash Set):** Longest Consecutive - 20 min

### Evening (30 min)

#### 4. Final Prep
- [ ] Read MINDSET_AND_CONFIDENCE.md (15 min)
- [ ] Write down your pattern cheat sheet (15 min)

**Pattern Recognition Cheat Sheet:**
```
See "count/frequency" → Frequency Counting (Counter)
See "group items" → Grouping (defaultdict + key function)
See "find pairs/sum" → Two-Sum (complement in hash map)
See "subarray sum" → Prefix Sum ({0:1} base case)
See "substring/window" → Sliding Window (track with hash map)
See "consecutive sequence" → Hash Set (only process starts)
```

### Night Before Interview

#### 5. Light Review & Rest (30 min + sleep)
- [ ] Skim pattern templates (10 min)
- [ ] Do ONE Easy problem (10 min)
- [ ] Read affirmations (5 min)
- [ ] Get good sleep (8 hours)

**Affirmations:**
- "I've learned 6 patterns in 5 days"
- "I've solved 15+ problems"
- "I can recognize common patterns"
- "I'll communicate clearly"
- "I've done my best to prepare"

### End of Day 5 Goal:
- ✅ Done full mock interview
- ✅ Reviewed all 6 patterns
- ✅ Solved 15-18 total problems
- ✅ Ready to give it your best shot

---

## 📊 5-Day Progress Tracker

### Daily Checklist:

**Day 1:**
- [ ] Learned patterns 1 & 2
- [ ] Solved 3 problems
- [ ] Can explain solutions

**Day 2:**
- [ ] Learned patterns 3 & 4
- [ ] Solved 3 problems
- [ ] Understand prefix sum

**Day 3:**
- [ ] Learned pattern 5
- [ ] Solved 3 problems
- [ ] Improved speed

**Day 4:**
- [ ] Learned pattern 6
- [ ] Solved 3 problems
- [ ] Practiced communication

**Day 5:**
- [ ] Mock interview
- [ ] Reviewed all patterns
- [ ] Final prep

**Total:** 15-18 problems solved, 6 patterns learned

---

## 🎯 Interview Day Strategy

### Before Interview (30 min before)
- ✅ Review pattern cheat sheet (5 min)
- ✅ Do 1 Easy problem as warm-up (10 min)
- ✅ Deep breathing (5 min)
- ✅ Positive affirmations (5 min)

### During Interview

#### Phase 1: Listen & Understand (2 min)
```
- Listen carefully to the problem
- Restate in your own words
- Ask clarifying questions
- Note constraints
```

#### Phase 2: Pattern Recognition (1-2 min)
```
- "This looks like a [PATTERN] problem"
- Think: What pattern fits?
- If unsure, ask: "Can I have a moment to think about the approach?"
```

#### Phase 3: Explain Approach (2 min)
```
- "I'll use [DATA STRUCTURE] because [REASON]"
- Explain high-level approach
- Mention time/space complexity
- Get interviewer buy-in
```

#### Phase 4: Code (15 min)
```
- Talk while coding
- Explain key decisions
- Use meaningful variable names
- Write clean, readable code
```

#### Phase 5: Test (3 min)
```
- Trace through given example
- Test edge cases
- Fix any bugs
```

#### Phase 6: Discuss (2 min)
```
- Explain complexity
- Discuss trade-offs
- Mention optimizations
```

### If You Get Stuck:

**DON'T:**
- ❌ Panic silently
- ❌ Give up
- ❌ Stop communicating

**DO:**
```
✅ "Let me think through this systematically..."
✅ "I'm considering approach A vs B..."
✅ "Can you give me a hint about [SPECIFIC THING]?"
✅ "Let me trace through an example to debug..."
```

### Partial Credit Strategy:

**If you can't solve perfectly:**
```
1. Get the brute force solution working first
2. Explain: "This is O(n²), but it works"
3. Then optimize: "To improve, I could use a hash map..."
4. Even partial optimization shows understanding
```

**Remember:** 70% correct with good communication > 100% correct with silence

---

## 🎓 What You CAN Realistically Expect

### After 5 Days of Intensive Prep:

**You WILL be able to:**
- ✅ Recognize 6 common patterns
- ✅ Solve Easy problems (~80% success)
- ✅ Solve Medium problems (~50-60% success)
- ✅ Explain your approach clearly
- ✅ Debug systematically
- ✅ Get partial credit even when stuck

**You will NOT be able to:**
- ❌ Solve all Hard problems
- ❌ Be as fast as someone with 2+ months practice
- ❌ Handle every curveball question
- ❌ Guarantee perfect performance

**But that's OKAY:**
- Companies know not everyone is a LeetCode expert
- Good communication can compensate for imperfect solutions
- Showing problem-solving process matters
- You might get lucky with familiar patterns

---

## 🚨 Honest Success Probability

### With This 5-Day Plan:

**Best Case Scenario:** 70-80% success
- You get questions matching patterns you learned
- You recognize them quickly
- You communicate well
- **Outcome:** Pass interview

**Likely Scenario:** 50-60% success
- You get some familiar patterns, some unfamiliar
- You solve Easy/Medium with hints
- You struggle but show process
- **Outcome:** Depends on interviewer standards

**Worst Case Scenario:** 30-40% success
- You get patterns you didn't study (LRU Cache, etc.)
- Time pressure causes mistakes
- Nerves affect performance
- **Outcome:** Might not pass, but learned a ton

### Reality Check:
**5 days is NOT enough to master this.** But it's enough to:
- Learn the basics
- Improve significantly
- Give yourself a chance
- Perform way better than going in blind

---

## 💪 Mindset for 5-Day Prep

### Things to Remember:

1. **You're learning, not mastering**
   - Goal: Get competent, not perfect
   - Focus: Common patterns only

2. **Progress over perfection**
   - 70% right is better than 0%
   - Partial credit counts

3. **Communication is key**
   - Even if stuck, talk through thinking
   - Show problem-solving ability
   - Ask good questions

4. **You're still a great engineer**
   - This doesn't measure your worth
   - You have 2 years of real experience
   - Interview is just one skill

5. **You'll do your best**
   - Whatever happens, you prepared hard
   - You'll learn from this
   - Next interview will be easier

---

## 📞 Final Checklist

### 5 Days Before Interview:
- [ ] Read this plan thoroughly
- [ ] Block 3-4 hours per day
- [ ] Clear your schedule

### 4 Days Before:
- [ ] Complete Day 1 tasks
- [ ] Solved first 3 problems

### 3 Days Before:
- [ ] Complete Day 2 tasks
- [ ] Understand prefix sum

### 2 Days Before:
- [ ] Complete Day 3 tasks
- [ ] Improved speed

### 1 Day Before:
- [ ] Complete Day 4 tasks
- [ ] Practiced communication

### Interview Day:
- [ ] Complete Day 5 morning
- [ ] Light review
- [ ] Get good sleep
- [ ] BELIEVE IN YOURSELF

---

## 🎯 Success Metrics (5-Day Version)

### End of Day 1:
- [ ] Solved 3 problems
- [ ] Understand 2 patterns

### End of Day 3:
- [ ] Solved 9 problems
- [ ] Understand 5 patterns
- [ ] Getting faster

### End of Day 5:
- [ ] Solved 15-18 problems
- [ ] Understand 6 patterns
- [ ] Can communicate clearly
- [ ] READY TO INTERVIEW

---

## 🚀 You've Got a Fighting Chance

**The truth:**
- 5 days is tough, but not impossible
- You'll improve dramatically
- You might get lucky with patterns
- At minimum, you'll learn for next time

**Your job:**
- Give it everything for 5 days
- Focus on common patterns
- Practice communication
- Do your best

**Remember:**
Even if you don't ace this interview, you're building skills for the next one. Every problem you solve, every pattern you learn, compounds.

**You can do this. Go ALL IN for 5 days.** 💪🔥

---

*"Success is not final, failure is not fatal: it is the courage to continue that counts."*

Give it your absolute best. That's all anyone can ask. 🚀
