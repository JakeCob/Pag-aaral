"""
Breadth-First Search (BFS) for Graphs - Implementation with Simulation
Demonstrates BFS traversal on the graph structure from your image
Key difference from trees: Graphs can have cycles, so we need a 'visited' set
"""

from collections import deque
from typing import Dict, List, Set, Optional
import time

class Graph:
    """Graph representation using adjacency list"""
    
    def __init__(self):
        self.adjacency_list: Dict[int, List[int]] = {}
    
    def add_edge(self, node1: int, node2: int):
        """Add bidirectional edge between two nodes"""
        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = []
        if node2 not in self.adjacency_list:
            self.adjacency_list[node2] = []
        
        # Add bidirectional connection
        if node2 not in self.adjacency_list[node1]:
            self.adjacency_list[node1].append(node2)
        if node1 not in self.adjacency_list[node2]:
            self.adjacency_list[node2].append(node1)
    
    def get_neighbors(self, node: int) -> List[int]:
        """Get all neighbors of a node"""
        return self.adjacency_list.get(node, [])
    
    def display(self):
        """Display the graph structure"""
        print("\nGraph Structure (Adjacency List):")
        print("-" * 40)
        for node in sorted(self.adjacency_list.keys()):
            neighbors = sorted(self.adjacency_list[node])
            print(f"Node {node}: {neighbors}")

class BFSGraphSearch:
    """BFS implementation for graphs with step-by-step simulation"""
    
    def __init__(self, graph: Graph, visualize: bool = True):
        self.graph = graph
        self.visualize = visualize
        self.steps = []
    
    def bfs(self, root: int, target: Optional[int] = None) -> tuple:
        """
        Breadth-First Search implementation for graphs
        Exact implementation from your image
        
        Returns: (found_target, traversal_order, all_steps)
        """
        queue = deque([root])
        visited = set([root])
        traversal_order = []
        self.steps = []
        step_num = 0
        
        while len(queue) > 0:
            # Record current state
            queue_list = list(queue)
            visited_list = sorted(list(visited))
            
            # Pop from left (FIFO)
            node = queue.popleft()
            traversal_order.append(node)
            
            # Record this step
            step_info = {
                'step': step_num,
                'queue_before': queue_list.copy(),
                'visited_before': visited_list.copy(),
                'current_node': node,
                'queue_after': list(queue),
                'visited_after': sorted(list(visited)),
                'traversal_so_far': traversal_order.copy()
            }
            
            # Check if we found the target
            if target and node == target:
                step_info['result'] = f'FOUND({node})'
                self.steps.append(step_info)
                if self.visualize:
                    self._print_step(step_info)
                return True, traversal_order, self.steps
            
            # Process neighbors
            neighbors_added = []
            neighbors_skipped = []
            
            for neighbor in self.graph.get_neighbors(node):
                if neighbor in visited:
                    neighbors_skipped.append(neighbor)
                    continue  # Skip already visited nodes
                queue.append(neighbor)
                visited.add(neighbor)
                neighbors_added.append(neighbor)
            
            step_info['neighbors_checked'] = self.graph.get_neighbors(node)
            step_info['neighbors_added'] = neighbors_added
            step_info['neighbors_skipped'] = neighbors_skipped
            step_info['queue_after'] = list(queue)
            step_info['visited_after'] = sorted(list(visited))
            
            self.steps.append(step_info)
            
            if self.visualize:
                self._print_step(step_info)
            
            step_num += 1
        
        return False, traversal_order, self.steps
    
    def _print_step(self, step_info: dict):
        """Pretty print each step of BFS"""
        print(f"\n{'='*60}")
        print(f"Step {step_info['step']}")
        print(f"{'='*60}")
        print(f"Queue before pop: {step_info['queue_before']}")
        print(f"Visited set: {step_info['visited_before']}")
        print(f"Current node (popped): {step_info['current_node']}")
        
        if 'neighbors_checked' in step_info:
            print(f"\nNeighbors of {step_info['current_node']}: {step_info['neighbors_checked']}")
            
            for neighbor in step_info['neighbors_checked']:
                if neighbor in step_info['neighbors_skipped']:
                    print(f"  - {neighbor}: Already visited (skip)")
                elif neighbor in step_info['neighbors_added']:
                    print(f"  - {neighbor}: Not visited (add to queue)")
        
        print(f"\nQueue after: {step_info['queue_after']}")
        print(f"Visited after: {step_info['visited_after']}")
        print(f"Traversal path: {step_info['traversal_so_far']}")
        
        if 'result' in step_info:
            print(f"\n🎯 Result: {step_info['result']}")

def create_sample_graph() -> Graph:
    """Create the exact graph from your image"""
    graph = Graph()
    
    # Add edges based on the visual
    # Node 1 connects to Node 2
    graph.add_edge(1, 2)
    
    # Node 2 connects to Nodes 1, 3, 4
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    
    # Node 3 connects to Nodes 2, 5
    graph.add_edge(3, 5)
    
    # Node 4 connects to Nodes 2, 6
    graph.add_edge(4, 6)
    
    # Node 5 connects to Nodes 3, 6
    graph.add_edge(5, 6)
    
    # Node 6 connects to Nodes 4, 5
    # (already added through bidirectional edges above)
    
    return graph

def visualize_graph():
    """ASCII art visualization of the graph"""
    print("\n" + "="*60)
    print("GRAPH VISUALIZATION")
    print("="*60)
    print("""
    Graph Structure:
    
        3 ——— 5
       /       \\
    1 — 2       6
         \\     /
          4 ———
    
    Edges:
    - 1-2: Node 1 connects to Node 2
    - 2-3: Node 2 connects to Node 3
    - 2-4: Node 2 connects to Node 4
    - 3-5: Node 3 connects to Node 5
    - 4-6: Node 4 connects to Node 6
    - 5-6: Node 5 connects to Node 6
    """)

def run_simulation():
    """Run the complete BFS simulation"""
    print("\n" + "="*60)
    print("BFS GRAPH TRAVERSAL SIMULATION")
    print("="*60)
    
    # Create and display the graph
    graph = create_sample_graph()
    visualize_graph()
    graph.display()
    
    # Example 1: BFS starting from node 1 (as shown in image)
    print("\n\n" + "🔍 " + "="*56)
    print("EXAMPLE 1: BFS starting from Node 1")
    print("="*60)
    
    bfs_search = BFSGraphSearch(graph, visualize=True)
    found, path, steps = bfs_search.bfs(root=1)
    
    print("\n\n📝 SUMMARY:")
    print("-" * 40)
    print(f"Starting node: 1")
    print(f"Complete traversal order: {path}")
    print(f"Number of nodes visited: {len(path)}")
    print(f"Total steps: {len(steps)}")
    
    # Example 2: Search for a specific node
    print("\n\n" + "🔍 " + "="*56)
    print("EXAMPLE 2: Search for Node 6 starting from Node 1")
    print("="*60)
    
    bfs_search2 = BFSGraphSearch(graph, visualize=True)
    found, path2, steps2 = bfs_search2.bfs(root=1, target=6)
    
    print("\n\n📝 SUMMARY:")
    print("-" * 40)
    print(f"Target: 6")
    print(f"Found: {'Yes ✅' if found else 'No ❌'}")
    print(f"Path traversed: {path2}")
    print(f"Found after visiting {len(path2)} nodes")

def compare_different_starts():
    """Show how BFS order changes with different starting nodes"""
    print("\n\n" + "⚡ " + "="*56)
    print("BFS FROM DIFFERENT STARTING NODES")
    print("="*60)
    
    graph = create_sample_graph()
    
    for start_node in range(1, 7):
        bfs_search = BFSGraphSearch(graph, visualize=False)
        _, path, _ = bfs_search.bfs(root=start_node)
        print(f"Starting from Node {start_node}: {path}")
    
    print("\nNotice how the traversal order changes based on starting node!")

def demonstrate_visited_importance():
    """Show why the visited set is crucial in graphs"""
    print("\n\n" + "⚠️ " + "="*56)
    print("WHY 'VISITED' SET IS CRUCIAL IN GRAPHS")
    print("="*60)
    
    print("""
    Without a 'visited' set, BFS would get stuck in cycles!
    
    Example: Starting at Node 1
    - Visit 1, add neighbor 2 to queue
    - Visit 2, add neighbors 1, 3, 4 to queue
    - Visit 1 again (cycle!), add 2 to queue
    - Visit 3, add neighbors...
    - Visit 4, add neighbors...
    - Visit 2 again (cycle!)...
    
    This would continue forever!
    
    The 'visited' set prevents this by:
    1. Marking nodes as visited when first discovered
    2. Skipping already-visited nodes
    3. Ensuring each node is processed exactly once
    """)

def shortest_path_demo():
    """Demonstrate that BFS finds shortest path in unweighted graphs"""
    print("\n\n" + "🏆 " + "="*56)
    print("BFS FINDS SHORTEST PATH (UNWEIGHTED GRAPHS)")
    print("="*60)
    
    graph = create_sample_graph()
    
    # Modified BFS to track parent nodes for path reconstruction
    def bfs_with_path(graph: Graph, start: int, target: int) -> List[int]:
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            node = queue.popleft()
            
            if node == target:
                # Reconstruct path
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = node
        
        return []
    
    # Find shortest paths between different nodes
    test_pairs = [(1, 6), (1, 5), (3, 4), (2, 6)]
    
    for start, end in test_pairs:
        path = bfs_with_path(graph, start, end)
        print(f"Shortest path from {start} to {end}: {path} (length: {len(path)-1} edges)")
    
    print("\nBFS guarantees the shortest path because it explores")
    print("all nodes at distance k before exploring nodes at distance k+1!")

# Simplified version matching exactly your image
def bfs_exact_from_image(root: int, graph: Graph):
    """Exact implementation from your image"""
    queue = deque([root])
    visited = set([root])
    
    print(f"\nExact implementation from image:")
    print(f"Starting with root = {root}")
    print(f"Initial: queue = [{root}], visited = {{{root}}}")
    
    step = 1
    while len(queue) > 0:
        node = queue.popleft()
        print(f"\nStep {step}: Processing node {node}")
        
        for neighbor in graph.get_neighbors(node):
            if neighbor in visited:
                print(f"  - Neighbor {neighbor}: already visited, continue")
                continue
            print(f"  - Neighbor {neighbor}: not visited, adding to queue")
            queue.append(neighbor)
            visited.add(neighbor)
        
        print(f"  After: queue = {list(queue)}, visited = {sorted(visited)}")
        step += 1
    
    print(f"\nFinal visited order reflects BFS traversal")
    return visited

if __name__ == "__main__":
    # Run main simulation
    run_simulation()
    
    # Show different starting points
    compare_different_starts()
    
    # Explain importance of visited set
    demonstrate_visited_importance()
    
    # Show shortest path property
    shortest_path_demo()
    
    # Run exact implementation from image
    print("\n\n" + "📋 " + "="*56)
    print("EXACT IMPLEMENTATION FROM YOUR IMAGE")
    print("="*60)
    
    graph = create_sample_graph()
    bfs_exact_from_image(1, graph)
    
    print("\n\n" + "="*60)
    print("✨ SIMULATION COMPLETE!")
    print("="*60)