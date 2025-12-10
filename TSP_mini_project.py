import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import math
import time
import itertools
import random
from typing import List, Tuple, Callable, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment

# ==========================================
# Core TSP Algorithms
# ==========================================

def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix from an (n,2) ndarray of points."""
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = math.sqrt((points[i][0] - points[j][0])**2 + 
                                      (points[i][1] - points[j][1])**2)
    return dist

def brute_force_tsp(distances: np.ndarray, callback: Optional[Callable] = None) -> Tuple[List[int], float, float]:
    """
    Brute force TSP solver - tries all permutations.
    WARNING: Exponential time complexity O(n!)
    No city limit - runs for any number of cities
    """
    start_time = time.time()
    n = len(distances)
    cities = list(range(n))
    
    best_path = None
    best_dist = float('inf')
    
    # Generate all permutations starting from city 0
    for perm in itertools.permutations(cities[1:]):
        path = [0] + list(perm)
        dist = sum(distances[path[i], path[i+1]] for i in range(n-1))
        dist += distances[path[-1], path[0]]  # Return to start
        
        if dist < best_dist:
            best_dist = dist
            best_path = path
            if callback:
                callback(path, dist)
    
    elapsed = time.time() - start_time
    return best_path, best_dist, elapsed

def dynamic_programming_tsp(distances: np.ndarray, callback: Optional[Callable] = None) -> Tuple[List[int], float, float]:
    """
    Dynamic Programming TSP solver using Held-Karp algorithm.
    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)
    Much faster than brute force for n > 10, but still exponential.
    """
    start_time = time.time()
    n = len(distances)
    
    # dp[mask][i] = minimum cost to visit all cities in mask ending at city i
    # mask is a bitmask representing visited cities
    dp = {}
    parent = {}
    
    # Base case: start from city 0
    for i in range(1, n):
        mask = (1 << 0) | (1 << i)  # Visited city 0 and i
        dp[(mask, i)] = distances[0][i]
        parent[(mask, i)] = 0
    
    # Fill DP table
    for mask_size in range(3, n + 1):
        # Generate all subsets of size mask_size that include city 0
        for subset in itertools.combinations(range(1, n), mask_size - 1):
            mask = (1 << 0)  # Start with city 0
            for city in subset:
                mask |= (1 << city)
            
            # Try ending at each city in the subset
            for last in subset:
                prev_mask = mask & ~(1 << last)  # Remove last city from mask
                min_dist = float('inf')
                best_prev = -1
                
                # Try all possible previous cities
                for prev in subset:
                    if prev == last:
                        continue
                    if (prev_mask, prev) in dp:
                        dist = dp[(prev_mask, prev)] + distances[prev][last]
                        if dist < min_dist:
                            min_dist = dist
                            best_prev = prev
                
                if best_prev != -1:
                    dp[(mask, last)] = min_dist
                    parent[(mask, last)] = best_prev
    
    # Find the best path back to city 0
    full_mask = (1 << n) - 1  # All cities visited
    min_dist = float('inf')
    best_last = -1
    
    for i in range(1, n):
        if (full_mask, i) in dp:
            dist = dp[(full_mask, i)] + distances[i][0]
            if dist < min_dist:
                min_dist = dist
                best_last = i
    
    # Reconstruct path
    path = []
    mask = full_mask
    current = best_last
    
    while current != -1:
        path.append(current)
        if (mask, current) not in parent:
            break
        next_city = parent[(mask, current)]
        mask &= ~(1 << current)
        current = next_city
    
    path.reverse()
    
    if callback:
        callback(path, min_dist)
    
    elapsed = time.time() - start_time
    return path, min_dist, elapsed

def nearest_neighbor_tsp(distances: np.ndarray, callback: Optional[Callable] = None) -> Tuple[List[int], float, float]:
    """Greedy nearest neighbor heuristic."""
    start_time = time.time()
    n = len(distances)
    unvisited = set(range(1, n))
    path = [0]
    total_dist = 0.0
    
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: distances[current][x])
        total_dist += distances[current][nearest]
        path.append(nearest)
        current = nearest
        unvisited.remove(nearest)
        if callback:
            callback(path[:], total_dist)
    
    total_dist += distances[path[-1]][path[0]]
    elapsed = time.time() - start_time
    return path, total_dist, elapsed

def simulated_annealing_tsp(distances: np.ndarray, callback: Optional[Callable] = None,
                           max_iterations: int = 10000, initial_temp: float = 100.0,
                           cooling_rate: float = 0.995) -> Tuple[List[int], float, float]:
    """Simulated annealing metaheuristic."""
    start_time = time.time()
    n = len(distances)
    
    def path_distance(path):
        dist = sum(distances[path[i], path[i+1]] for i in range(n-1))
        dist += distances[path[-1], path[0]]
        return dist
    
    current_path = list(range(n))
    random.shuffle(current_path[1:])
    current_dist = path_distance(current_path)
    
    best_path = current_path[:]
    best_dist = current_dist
    
    temp = initial_temp
    for iteration in range(max_iterations):
        i, j = sorted(random.sample(range(1, n), 2))
        new_path = current_path[:]
        new_path[i:j+1] = reversed(new_path[i:j+1])
        new_dist = path_distance(new_path)
        
        delta = new_dist - current_dist
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_path = new_path
            current_dist = new_dist
            
            if current_dist < best_dist:
                best_path = current_path[:]
                best_dist = current_dist
                if callback and iteration % 100 == 0:
                    callback(best_path, best_dist)
        
        temp *= cooling_rate
    
    elapsed = time.time() - start_time
    return best_path, best_dist, elapsed

def linear_programming_tsp(distances: np.ndarray, callback: Optional[Callable] = None) -> Tuple[List[int], float, float]:
    """
    Linear Programming TSP solver using Assignment Problem relaxation with subtour elimination.
    
    This approach uses the Hungarian Algorithm (scipy's linear_sum_assignment) to solve
    the assignment problem, then iteratively eliminates subtours until a valid tour is found.
    
    Method:
    1. Solve assignment problem (find minimum cost perfect matching)
    2. Extract subtours from the solution
    3. If multiple subtours exist, add constraints to eliminate them
    4. Repeat until a single tour is found
    
    Time complexity: Polynomial per iteration, but may need multiple iterations
    This is a heuristic approach that often finds optimal or near-optimal solutions.
    """
    start_time = time.time()
    n = len(distances)
    
    def find_subtours(assignment: np.ndarray) -> List[List[int]]:
        """
        Find all subtours in an assignment solution.
        Returns a list of subtours, where each subtour is a list of city indices.
        """
        visited = set()
        subtours = []
        
        for start in range(n):
            if start in visited:
                continue
            
            # Trace the subtour starting from this city
            tour = []
            current = start
            while current not in visited:
                visited.add(current)
                tour.append(current)
                # Find where current city goes to
                current = int(assignment[current])
            
            if tour:
                subtours.append(tour)
        
        return subtours
    
    def solve_assignment_with_forbidden(forbidden_edges: Set[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        """
        Solve the assignment problem with certain edges forbidden.
        Returns the assignment and its total cost.
        """
        # Create a modified distance matrix
        modified_dist = distances.copy()
        
        # Set forbidden edges to a very high cost
        for i, j in forbidden_edges:
            modified_dist[i, j] = 1e10
        
        # Solve using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(modified_dist)
        
        # Calculate total cost (using original distances)
        total_cost = sum(distances[i, j] for i, j in zip(row_ind, col_ind))
        
        return col_ind, total_cost
    
    # Initial solution without any forbidden edges
    forbidden_edges: Set[Tuple[int, int]] = set()
    best_tour = None
    best_cost = float('inf')
    max_iterations = 100  # Prevent infinite loops
    
    for iteration in range(max_iterations):
        # Solve assignment problem
        assignment, cost = solve_assignment_with_forbidden(forbidden_edges)
        
        # Find subtours in the solution
        subtours = find_subtours(assignment)
        
        # If we have a single tour, we're done!
        if len(subtours) == 1:
            tour = subtours[0]
            total_dist = sum(distances[tour[i]][tour[(i+1) % n]] for i in range(n))
            
            if callback:
                callback(tour, total_dist)
            
            elapsed = time.time() - start_time
            return tour, total_dist, elapsed
        
        # Multiple subtours - need to eliminate them
        # Strategy: forbid one edge from the smallest subtour
        smallest_subtour = min(subtours, key=len)
        
        # Forbid the first edge of the smallest subtour
        if len(smallest_subtour) >= 2:
            i = smallest_subtour[0]
            j = smallest_subtour[1]
            forbidden_edges.add((i, j))
        
        # Keep track of best solution found so far (even if it has subtours)
        if cost < best_cost and cost < 1e9:  # Ignore solutions with forbidden edges
            best_cost = cost
            best_tour = subtours
    
    # If we couldn't find a single tour, construct one from the best subtours found
    # This is a fallback - connect subtours greedily
    if best_tour and len(best_tour) > 1:
        # Flatten all subtours into a single tour by connecting them
        tour = []
        for subtour in best_tour:
            tour.extend(subtour)
        
        total_dist = sum(distances[tour[i]][tour[(i+1) % n]] for i in range(n))
        
        if callback:
            callback(tour, total_dist)
        
        elapsed = time.time() - start_time
        return tour, total_dist, elapsed
    
    # Ultimate fallback: use nearest neighbor
    return nearest_neighbor_tsp(distances, callback)

# ==========================================
# GUI Application
# ==========================================

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - Algorithm Comparison (Parallel)")
        self.root.geometry("1600x900")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.points = []
        self.running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # Left panel - Canvas (Larger)
        left_frame = ctk.CTkFrame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(left_frame, text="TSP Visualization", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.canvas = tk.Canvas(left_frame, bg="white", width=800, height=700)
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.add_city)
        
        # Control buttons
        btn_frame = ctk.CTkFrame(left_frame)
        btn_frame.pack(pady=5)
        
        ctk.CTkButton(btn_frame, text="Clear All", command=self.clear_all, width=100).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btn_frame, text="Random Cities", command=self.add_random_cities, width=120).pack(side=tk.LEFT, padx=5)
        
        self.animate_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(btn_frame, text="Animate", variable=self.animate_var).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results (Smaller)
        right_frame = ctk.CTkFrame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="Algorithm Results", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.result_text = tk.Text(right_frame, height=30, width=45, font=("Courier", 10))
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
        
        # Individual Algorithm Buttons
        algo_label = ctk.CTkLabel(right_frame, text="Run Individual Algorithms:", 
                                  font=("Arial", 12, "bold"))
        algo_label.pack(pady=(5, 5), padx=10)
        
        # Create frame for individual algorithm buttons
        algo_btn_frame = ctk.CTkFrame(right_frame)
        algo_btn_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.bf_btn = ctk.CTkButton(algo_btn_frame, text="Brute Force", 
                                    command=lambda: self.run_single_algorithm(brute_force_tsp, "Brute Force"),
                                    width=140, height=30, fg_color="#e74c3c")
        self.bf_btn.grid(row=0, column=0, padx=5, pady=3)
        
        self.dp_btn = ctk.CTkButton(algo_btn_frame, text="Dynamic Prog.", 
                                    command=lambda: self.run_single_algorithm(dynamic_programming_tsp, "Dynamic Programming"),
                                    width=140, height=30, fg_color="#3498db")
        self.dp_btn.grid(row=0, column=1, padx=5, pady=3)
        
        self.nn_btn = ctk.CTkButton(algo_btn_frame, text="Nearest Neighbor", 
                                    command=lambda: self.run_single_algorithm(nearest_neighbor_tsp, "Nearest Neighbor"),
                                    width=140, height=30, fg_color="#2ecc71")
        self.nn_btn.grid(row=1, column=0, padx=5, pady=3)
        
        self.sa_btn = ctk.CTkButton(algo_btn_frame, text="Simulated Annealing", 
                                    command=lambda: self.run_single_algorithm(simulated_annealing_tsp, "Simulated Annealing"),
                                    width=140, height=30, fg_color="#f39c12")
        self.sa_btn.grid(row=1, column=1, padx=5, pady=3)
        
        # Linear Programming button (using scipy - always available)
        self.lp_btn = ctk.CTkButton(algo_btn_frame, text="Linear Programming", 
                                    command=lambda: self.run_single_algorithm(linear_programming_tsp, "Linear Programming"),
                                    width=140, height=30, fg_color="#9b59b6")
        self.lp_btn.grid(row=2, column=0, padx=5, pady=3)
        
        # Run All button (Parallel)
        self.run_btn = ctk.CTkButton(right_frame, text="🚀 Run All Algorithms (Parallel)", 
                                     command=self.run_comparison, height=40,
                                     font=("Arial", 14, "bold"), fg_color="#9b59b6")
        self.run_btn.pack(pady=10, padx=10, fill=tk.X)
    
    def add_city(self, event):
        if self.running:
            return
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="black", width=2)
        self.canvas.create_text(x, y-15, text=str(len(self.points)-1), font=("Arial", 10, "bold"))
    
    def clear_all(self):
        if self.running:
            return
        self.points = []
        self.canvas.delete("all")
        self.result_text.delete(1.0, tk.END)
    
    def add_random_cities(self):
        if self.running:
            return
        self.clear_all()
        n = random.randint(5, 12)
        for _ in range(n):
            x = random.randint(50, 750)
            y = random.randint(50, 650)
            self.points.append((x, y))
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="black", width=2)
            self.canvas.create_text(x, y-15, text=str(len(self.points)-1), font=("Arial", 10, "bold"))
    
    def draw_path(self, path, color="blue", width=2, show_dist=False):
        if not path:
            return
        for i in range(len(path)):
            p1 = self.points[path[i]]
            p2 = self.points[path[(i+1) % len(path)]]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=width, arrow=tk.LAST)
    
    def animation_callback(self, path, dist):
        self.canvas.delete("path")
        self.draw_path(path, color="lightblue", width=2)
        self.root.update()
        time.sleep(0.01)
    
    def set_running(self, state):
        self.running = state
        self.run_btn.configure(state="disabled" if state else "normal")
        self.bf_btn.configure(state="disabled" if state else "normal")
        self.dp_btn.configure(state="disabled" if state else "normal")
        self.nn_btn.configure(state="disabled" if state else "normal")
        self.sa_btn.configure(state="disabled" if state else "normal")
        self.lp_btn.configure(state="disabled" if state else "normal")
    
    def format_path_string(self, path):
        return " → ".join(map(str, path)) + f" → {path[0]}"
    
    def run_single_algorithm(self, algo_func, algo_name):
        """Run a single algorithm individually"""
        if not self.points:
            messagebox.showwarning("Warning", "Please add cities first!")
            return
            
        self.set_running(True)
        
        points_array = np.array(self.points)
        distances = compute_distance_matrix(points_array)
        n = len(self.points)
        
        self.result_text.delete(1.0, tk.END)
        self.log_result(f"{'='*50}")
        self.log_result(f"RUNNING: {algo_name}")
        self.log_result(f"Number of Cities: {n}")
        self.log_result(f"{'='*50}\n")
        
        callback = self.animation_callback if self.animate_var.get() else None
        
        try:
            self.log_result(f"Executing {algo_name}...")
            self.root.update()
            
            p, d, t = algo_func(distances, callback=callback)
            
            path_str = self.format_path_string(p)
            self.log_result(f"\n✓ Completed!\n")
            self.log_result(f"Path: {path_str}")
            self.log_result(f"Distance: {d:.2f}")
            self.log_result(f"Time: {t:.4f}s")
            self.log_result(f"{'='*50}")
            
            self.draw_path(p, color="#2ecc71", width=4, show_dist=True)
        except Exception as e:
            self.log_result(f"\n✗ Error: {str(e)}")
            self.log_result(f"{'='*50}")
        
        self.set_running(False)
    
    def run_comparison(self):
        if not self.points:
            messagebox.showwarning("Warning", "Please add cities first!")
            return
            
        self.set_running(True)
        
        points_array = np.array(self.points)
        distances = compute_distance_matrix(points_array)
        n = len(self.points)
        
        self.result_text.delete(1.0, tk.END)
        self.log_result(f"{'='*50}")
        self.log_result(f"ALGORITHM COMPARISON (PARALLEL)")
        self.log_result(f"Number of Cities: {n}")
        self.log_result(f"{'='*50}\n")
        
        best_dist = float('inf')
        best_path = []
        best_name = ""
        best_time = float('inf')
        fastest_name = ""
        
        # Disable callback for parallel execution (GUI updates not thread-safe)
        callback = None
        
        # Define algorithms to run
        algorithms = [
            (brute_force_tsp, "Brute Force"),
            (dynamic_programming_tsp, "Dynamic Programming (Held-Karp)"),
            (nearest_neighbor_tsp, "Nearest Neighbor"),
            (simulated_annealing_tsp, "Simulated Annealing"),
            (linear_programming_tsp, "Linear Programming (Hungarian)")
        ]
        
        results = {}
        
        # Run algorithms in parallel using ThreadPoolExecutor
        self.log_result("⚡ Running all algorithms in parallel...\n")
        self.root.update()
        
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_algo = {
                executor.submit(func, distances, callback): name 
                for func, name in algorithms
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_algo):
                name = future_to_algo[future]
                try:
                    p, d, t = future.result()
                    results[name] = (p, d, t)
                    self.log_result(f"✓ {name} completed in {t:.4f}s")
                    self.root.update()
                except Exception as e:
                    self.log_result(f"✗ {name} failed: {str(e)}")
                    self.root.update()
        
        overall_time = time.time() - overall_start
        
        self.log_result(f"\n⏱️  Total parallel execution time: {overall_time:.4f}s\n")
        self.log_result(f"{'='*50}\n")
        
        # Display detailed results
        for name in [algo[1] for algo in algorithms]:
            if name in results:
                p, d, t = results[name]
                path_str = self.format_path_string(p)
                self.log_result(f"{name}:")
                self.log_result(f"  Path: {path_str}")
                self.log_result(f"  Distance: {d:.2f}")
                self.log_result(f"  Time: {t:.4f}s\n")
                
                # Track shortest distance
                if d < best_dist:
                    best_dist = d
                    best_path = p
                    best_name = name
                
                # Track fastest algorithm
                if t < best_time:
                    best_time = t
                    fastest_name = name
        
        self.log_result(f"{'='*50}")
        self.log_result(f"🏆 BEST RESULT (Fastest): {fastest_name}")
        self.log_result(f"   Time: {best_time:.4f}s")
        self.log_result(f"\n📏 SHORTEST DISTANCE: {best_name}")
        self.log_result(f"   Distance: {best_dist:.2f}")
        self.log_result(f"{'='*50}")
        
        self.draw_path(best_path, color="#2ecc71", width=4, show_dist=True)
        self.set_running(False)

    def log_result(self, text):
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)

if __name__ == "__main__":
    root = ctk.CTk()
    app = TSPApp(root)
    root.mainloop()