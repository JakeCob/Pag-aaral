"""
Breadth-First Search (BFS) Tree Implementation with Simulation
Demonstrates BFS traversal on the tree structure from your image
"""

from collections import deque
from typing import Optional, List, Any
import time

class TreeNode:
    """Binary Tree Node class"""
    def __init__(self, value: int):
        self.value = value
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.children: List[TreeNode] = []  # For generic tree support
    
    def add_children(self, *children):
        """Add children to the node (for binary tree, max 2)"""
        for child in children:
            if isinstance(child, TreeNode):
                self.children.append(child)
                if self.left is None:
                    self.left = child
                elif self.right is None:
                    self.right = child
        return self

class BFSTreeSearch:
    """BFS implementation with step-by-step simulation"""
    
    def __init__(self, visualize: bool = True):
        self.visualize = visualize
        self.steps = []
        
    def bfs(self, root: TreeNode, target: int = None) -> tuple:
        """
        Breadth-First Search implementation
        Returns: (found_node or None, traversal_path, all_steps)
        """
        if not root:
            return None, [], []
        
        queue = deque([root])
        traversal_path = []
        self.steps = []
        step_num = 0
        
        # Helper function to check if we found the target
        def is_goal(node: TreeNode) -> bool:
            return node.value == target if target is not None else False
        
        while len(queue) > 0:
            # Record current state
            queue_values = [n.value for n in queue]
            
            # Pop from left (FIFO - First In First Out)
            node = queue.popleft()
            traversal_path.append(node.value)
            
            # Record this step
            step_info = {
                'step': step_num,
                'queue_before': queue_values.copy(),
                'current_node': node.value,
                'queue_after': [n.value for n in queue],
                'traversal_so_far': traversal_path.copy()
            }
            
            # Check if this is the goal
            if is_goal(node):
                step_info['result'] = f'FOUND({node.value})'
                self.steps.append(step_info)
                if self.visualize:
                    self._print_step(step_info)
                return node, traversal_path, self.steps
            
            # Add children to queue
            children_added = []
            for child in node.children:
                if child:
                    queue.append(child)
                    children_added.append(child.value)
            
            step_info['children_added'] = children_added
            step_info['queue_after'] = [n.value for n in queue]
            self.steps.append(step_info)
            
            if self.visualize:
                self._print_step(step_info)
            
            step_num += 1
        
        # Not found
        return None, traversal_path, self.steps
    
    def _print_step(self, step_info: dict):
        """Pretty print each step of the BFS"""
        print(f"\n{'='*60}")
        print(f"Step {step_info['step']}")
        print(f"{'='*60}")
        print(f"Queue before pop: {step_info['queue_before']}")
        print(f"Current node (popped): {step_info['current_node']}")
        
        if 'children_added' in step_info:
            if step_info['children_added']:
                print(f"Children added to queue: {step_info['children_added']}")
            else:
                print("No children to add")
        
        print(f"Queue after: {step_info['queue_after']}")
        print(f"Traversal path: {step_info['traversal_so_far']}")
        
        if 'result' in step_info:
            print(f"\n🎯 Result: {step_info['result']}")

def create_sample_tree() -> TreeNode:
    """Create the exact tree from your image"""
    # Create nodes
    root = TreeNode(20)
    node14 = TreeNode(14)
    node24 = TreeNode(24)
    node11 = TreeNode(11)
    node16 = TreeNode(16)
    node27 = TreeNode(27)
    node29 = TreeNode(29)
    
    # Build tree structure
    root.add_children(node14, node24)
    node14.add_children(node11, node16)
    node24.add_children(node27, node29)
    
    return root

def print_tree(node: TreeNode, level: int = 0, prefix: str = "Root: "):
    """Print tree structure"""
    if node:
        print(" " * (level * 4) + prefix + str(node.value))
        if node.left or node.right:
            if node.left:
                print_tree(node.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")
            if node.right:
                print_tree(node.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")

def run_simulation():
    """Run the complete BFS simulation"""
    print("\n" + "="*60)
    print("BFS TREE TRAVERSAL SIMULATION")
    print("="*60)
    
    # Create the tree
    root = create_sample_tree()
    
    # Display the tree structure
    print("\n📊 TREE STRUCTURE:")
    print("-" * 40)
    print_tree(root)
    
    # Initialize BFS
    bfs_search = BFSTreeSearch(visualize=True)
    
    # Example 1: Search for a value that exists
    print("\n\n" + "🔍 " + "="*56)
    print("EXAMPLE 1: Searching for value 27")
    print("="*60)
    
    found_node, path, steps = bfs_search.bfs(root, target=27)
    
    print("\n\n📝 SUMMARY:")
    print("-" * 40)
    print(f"Target: 27")
    print(f"Found: {'Yes ✅' if found_node else 'No ❌'}")
    print(f"Complete traversal path: {path}")
    print(f"Number of nodes visited: {len(path)}")
    
    # Example 2: Search for a value that doesn't exist
    print("\n\n" + "🔍 " + "="*56)
    print("EXAMPLE 2: Searching for value 100 (doesn't exist)")
    print("="*60)
    
    bfs_search2 = BFSTreeSearch(visualize=True)
    found_node2, path2, steps2 = bfs_search2.bfs(root, target=100)
    
    print("\n\n📝 SUMMARY:")
    print("-" * 40)
    print(f"Target: 100")
    print(f"Found: {'Yes ✅' if found_node2 else 'No ❌'}")
    print(f"Complete traversal path: {path2}")
    print(f"Number of nodes visited: {len(path2)}")
    
    # Example 3: Complete traversal without target
    print("\n\n" + "🔍 " + "="*56)
    print("EXAMPLE 3: Complete BFS traversal (no specific target)")
    print("="*60)
    
    bfs_search3 = BFSTreeSearch(visualize=False)
    _, path3, _ = bfs_search3.bfs(root)
    
    print(f"\nComplete BFS traversal order: {path3}")
    print("\nBFS visits nodes level by level (breadth-first):")
    print("Level 0 (root): [20]")
    print("Level 1: [14, 24]")
    print("Level 2: [11, 16, 27, 29]")

def compare_with_dfs():
    """Compare BFS with DFS to show the difference"""
    print("\n\n" + "⚡ " + "="*56)
    print("BFS vs DFS COMPARISON")
    print("="*60)
    
    root = create_sample_tree()
    
    # BFS traversal
    bfs_search = BFSTreeSearch(visualize=False)
    _, bfs_path, _ = bfs_search.bfs(root)
    
    # Simple DFS implementation for comparison
    def dfs(node: TreeNode) -> List[int]:
        if not node:
            return []
        result = [node.value]
        for child in node.children:
            result.extend(dfs(child))
        return result
    
    dfs_path = dfs(root)
    
    print(f"\nBFS traversal (level by level): {bfs_path}")
    print(f"DFS traversal (depth first):    {dfs_path}")
    
    print("\n📊 Key Differences:")
    print("-" * 40)
    print("BFS: Uses a queue (FIFO) - explores all nodes at current level before moving deeper")
    print("DFS: Uses a stack (LIFO) - explores as deep as possible before backtracking")
    print("\nBFS is optimal for finding shortest path in unweighted graphs!")

def interactive_mode():
    """Interactive mode to search for custom values"""
    print("\n\n" + "🎮 " + "="*56)
    print("INTERACTIVE MODE")
    print("="*60)
    
    root = create_sample_tree()
    print("\nAvailable values in tree: [20, 14, 24, 11, 16, 27, 29]")
    
    while True:
        try:
            user_input = input("\nEnter a value to search (or 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                print("Exiting interactive mode...")
                break
            
            target = int(user_input)
            print(f"\n🔍 Searching for {target}...")
            print("-" * 40)
            
            bfs_search = BFSTreeSearch(visualize=False)
            found_node, path, steps = bfs_search.bfs(root, target=target)
            
            if found_node:
                print(f"✅ Found {target}!")
                print(f"Nodes visited in order: {path}")
                print(f"Number of steps: {len(steps)}")
            else:
                print(f"❌ Value {target} not found in tree")
                print(f"All nodes visited: {path}")
                
        except ValueError:
            print("Please enter a valid integer or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

if __name__ == "__main__":
    # Run the main simulation
    run_simulation()
    
    # Show BFS vs DFS comparison
    compare_with_dfs()
    
    # Optional: Uncomment for interactive mode
    # interactive_mode()
    
    print("\n\n" + "="*60)
    print("✨ SIMULATION COMPLETE!")
    print("="*60)