# Challenge 03: Circuit Breaker & Retry Logic (Advanced)

**Difficulty**: Advanced
**Time Estimate**: 50-60 minutes
**Interview Section**: Section 3 - Part C + Extensions

---

## 📋 Challenge Description

Implement **production-grade fault tolerance** for multi-agent systems using:
1. **Circuit Breaker Pattern**: Stop calling failing agents temporarily
2. **Exponential Backoff Retry**: Retry with increasing delays
3. **Fallback Agents**: Use backup agents when primary fails

---

## 🎯 Requirements

### Part A: Circuit Breaker (20 min)

**3 States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Too many failures, requests blocked
- **HALF_OPEN**: Testing if service recovered

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,  # Open after 3 failures
        recovery_timeout: int = 30,  # Try recovery after 30s
        success_threshold: int = 2   # Close after 2 successes in half-open
    ):
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
```

### Part B: Retry with Exponential Backoff (15 min)

```python
async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0  # seconds
):
    """
    Retry pattern: 1s, 2s, 4s, 8s...

    Example:
    - Attempt 1: Fail → wait 1s
    - Attempt 2: Fail → wait 2s
    - Attempt 3: Fail → wait 4s
    - Attempt 4: Success!
    """
```

### Part C: Fallback Agent Pattern (20 min)

```python
class FallbackOrchestrator:
    async def execute_with_fallback(
        self,
        query: str,
        primary_agent: BaseAgent,
        fallback_agents: List[BaseAgent]
    ) -> Dict[str, Any]:
        """
        Try primary agent first.
        If fails, try fallback agents in order.
        """
```

---

## 📊 Example Output

```
=== Circuit Breaker Test ===

Attempt 1: ❌ FAILED (failures: 1/3)
Attempt 2: ❌ FAILED (failures: 2/3)
Attempt 3: ❌ FAILED (failures: 3/3)
>>> Circuit OPEN - blocking requests

Attempt 4: ⛔ BLOCKED (circuit is open)
Attempt 5: ⛔ BLOCKED (circuit is open)

[30 seconds later...]
>>> Circuit HALF_OPEN - testing recovery

Attempt 6: ✅ SUCCESS (successes: 1/2 in half-open)
Attempt 7: ✅ SUCCESS (successes: 2/2 in half-open)
>>> Circuit CLOSED - back to normal

---

=== Retry with Exponential Backoff ===

Attempt 1: Failed → Retry in 1.0s
Attempt 2: Failed → Retry in 2.0s
Attempt 3: Success! ✅

---

=== Fallback Agent Test ===

Query: "Analyze customer sentiment"

Primary (GPT4Agent): ❌ FAILED (timeout)
Fallback 1 (GPT35Agent): ❌ FAILED (rate limit)
Fallback 2 (LocalLLM): ✅ SUCCESS

Result: "Sentiment: Positive (confidence: 0.78)"
Agent used: LocalLLM (fallback level 2)
```

---

## 💡 Implementation Tips

### Circuit Breaker State Machine

```python
class CircuitBreaker:
    async def call(self, func, *args, **kwargs):
        # Check current state
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Handle success
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0

            return result

        except Exception as e:
            self._record_failure()
            raise
```

### Exponential Backoff

```python
for attempt in range(max_retries):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if attempt == max_retries - 1:
            raise  # Last attempt, give up

        delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s, 8s...
        await asyncio.sleep(delay)
```

---

## 🎓 Key Concepts

1. **Circuit Breaker**: Prevent cascading failures
2. **Exponential Backoff**: Give failing services time to recover
3. **Fallback Pattern**: Graceful degradation
4. **State Machine**: Manage circuit breaker states
5. **Jitter**: Add randomness to avoid thundering herd

---

**Time Allocation**:
- Circuit Breaker: 20 min
- Retry Logic: 15 min
- Fallback Pattern: 15 min
- Testing: 10 min
- **Total**: 60 min

**Good luck!** 🎯
