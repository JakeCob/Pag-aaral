'''
Requirements= arting indices of substring(s) in `s` that is a concatenation of each word in `words` exactly once
Expected Output = longest consecutive sequence of response time values

Input = tring `s` and an array of strings `words` of the same length
'''
def findSubstring(s: str, words: list[str]) -> list[int]:
    if not s or not words:
        return []

    target_count = {}
    for word in words:
        target_count[word] = target_count.get(word, 0) + 1
    l = 0

    # target_formed = sum(target_count.values())
    formed = 0
    word_count = {}
    substring_index = []
    start_index = 0
    while l < len(s):
        r = l + len(words[0])
        
        window_word = s[l:r]
        current_word_count = word_count.get(window_word, 0)
        required_word_count = target_count.get(window_word, 0)
        if window_word in words and current_word_count < required_word_count:
            word_count[window_word] = current_word_count + 1
            formed += 1
            
            if formed == 1:
                start_index = l
            if formed == len(words):
                substring_index.append(start_index)
                formed = 0
                word_count.clear()
                l = start_index + 1
            else:
                l = r
        else:
            formed = 0
            if word_count:
                word_count.clear()
            l = start_index + 1
            start_index = l

    return substring_index

# Example 1
s = "barfoothefoobarman"
words = ["foo","bar"]
result = findSubstring(s, words)
print(result)

# Example 2
s = "wordgoodgoodgoodbestword"
words = ["word","good","best","word"]
result = findSubstring(s, words)
print(result)

# Example 3
s = "barfoofoobarthefoobarman"
words = ["bar","foo","the"]
result = findSubstring(s, words)
print(result)

# Example 4
s = "lingmindraboofooowingdingbarrwingmonkeypoundcake"
words = ["fooo","barr","wing","ding","wing"]
result = findSubstring(s, words)
print(result)