# Challenge 02: DAG Workflow Orchestration (Intermediate)

**Difficulty**: Intermediate
**Time Estimate**: 45-60 minutes
**Interview Section**: Section 3 - Part B

---

## 📋 Challenge Description

Build a **DAG (Directed Acyclic Graph) workflow orchestrator** that executes agents in a specific order based on dependencies. This is crucial for complex multi-step workflows where some agents depend on outputs from others.

### Example Use Case

Processing a research paper:
1. **ExtractAgent**: Extract text from PDF → outputs text
2. **SummarizeAgent**: Summarize text (depends on ExtractAgent) → outputs summary
3. **TranslateAgent**: Translate summary (depends on SummarizeAgent) → outputs translated text
4. **EmailAgent**: Email results (depends on TranslateAgent)

Execution order: Extract → Summarize → Translate → Email

---

## 🎯 Requirements

### Part A: Task Graph Definition (15 min)

```python
from typing import List, Dict, Set

class Task:
    def __init__(self, agent_name: str, depends_on: List[str] = None):
        self.agent_name = agent_name
        self.depends_on = depends_on or []

# Example usage
tasks = [
    Task("Extract", depends_on=[]),
    Task("Summarize", depends_on=["Extract"]),
    Task("Translate", depends_on=["Summarize"]),
    Task("Email", depends_on=["Translate"])
]
```

### Part B: Topological Sort (20 min)

Implement Kahn's algorithm for topological sorting:

1. Find all nodes with no dependencies (in-degree = 0)
2. Process these nodes
3. Remove them and their edges
4. Repeat until all nodes processed

### Part C: Workflow Executor (20 min)

```python
class WorkflowOrchestrator:
    async def execute_workflow(
        self,
        tasks: List[Task],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tasks in topological order.

        Returns:
            Results from all task executions
        """
```

---

## 📊 Expected Output

```
=== Workflow Execution ===

Topological Order: ['Extract', 'Summarize', 'Translate', 'Email']

Step 1: Executing Extract
  Input: {'document': 'research_paper.pdf'}
  Output: {'text': 'Extracted content...'}

Step 2: Executing Summarize
  Input: {'text': 'Extracted content...'}
  Output: {'summary': 'Brief summary...'}

Step 3: Executing Translate
  Input: {'summary': 'Brief summary...'}
  Output: {'translated': 'Résumé traduit...'}

Step 4: Executing Email
  Input: {'translated': 'Résumé traduit...'}
  Output: {'sent': True, 'recipient': 'user@example.com'}

✅ Workflow completed successfully!
```

---

## 💡 Implementation Tips

### Topological Sort (Kahn's Algorithm)

```python
def topological_sort(self, tasks: List[Task]) -> List[str]:
    """Sort tasks in execution order using Kahn's algorithm"""
    # Build in-degree map
    in_degree = {task.agent_name: 0 for task in tasks}
    adj_list = {task.agent_name: [] for task in tasks}

    for task in tasks:
        for dep in task.depends_on:
            adj_list[dep].append(task.agent_name)
            in_degree[task.agent_name] += 1

    # Find all nodes with in-degree 0
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        # Process node
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree of neighbors
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task graph!")

    return result
```

### Parallel Execution (Advanced)

For tasks with no dependencies between them, execute in parallel:

```python
# Sequential
Step 1: A
Step 2: B (depends on A)
Step 3: C (depends on A)
Step 4: D (depends on B and C)

# Parallel optimization
Step 1: A
Step 2: B and C in parallel (both depend on A)
Step 3: D (depends on B and C)
```

---

## 🎓 Key Concepts

1. **DAG**: Directed Acyclic Graph - no cycles allowed
2. **Topological Sort**: Linear ordering respecting dependencies
3. **Kahn's Algorithm**: O(V + E) complexity for topological sort
4. **Dependency Resolution**: Passing outputs between tasks
5. **Cycle Detection**: Ensuring workflow is executable

---

**Time Allocation**:
- Task Graph: 15 min
- Topological Sort: 20 min
- Workflow Executor: 20 min
- Testing: 5 min
- **Total**: 60 min

**Good luck!** 🎯
