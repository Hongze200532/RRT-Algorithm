#The below import are all that should be used in this assignment. Additional libraries are not allowed.
import numpy as np
import math
import scipy.sparse.csgraph
import matplotlib.pyplot as plt
import random
import argparse
import collections
import sys
from scipy.sparse import csr_array
'''
==============================
The code below here is for your occupancy grid solution
==============================
'''
def read_map_from_file(filename):
"""
This functions reads a csv file describing a map and returns the map data
Inputs:
- filename (string): name of the file to read
Outputs:
- map (tuple): A map is a tuple of the form (grid_size, start_pos, goal_pos, [obstacles])
grid_size is a tuple (length, height) representing the size of the map
start_pos is a tuple (x, y) representing the x,y coordinates of the start position
goal_pos is a tuple (x, y) representing the x,y coordinate of the goal position
obstacles is a list of tuples. Each tuple represents a single circular obstacle and is of the form (x, y, radius).
x is an integer representing the x coordinate of the obstacle
y is an integer representing the y coordinate of the obstacle
radius is an integer representing the radius of the obstacle
"""
#Your code goes here
fp = open(filename)
txt = fp.read()# Read the file content and store it in the txt string
fp.close()
lines = txt.split('\n')# Split the file content by lines and store the result in the list lines
x_dimension, y_dimension = map(int, lines[0].split(','))# Split the first line into x/y dimensions (map size information)
start_x, start_y = map(int, lines[1].split(','))# The second line contains the start position coordinates
goal_x, goal_y = map(int, lines[2].split(','))# The third line contains the goal position coordinates
obstacles = []# Create an empty obstacles list
for line in lines[3:]:
obstacles.append(tuple(map(int, line.split(','))))# From the fourth line onward, each line describes one obstacle; iterate through these lines
return (x_dimension, y_dimension), (start_x, start_y), (goal_x, goal_y), obstacles# Finally return a tuple
def is_intersect(rect_left, rect_top, rect_right, rect_bottom, circle_x, circle_y, circle_radius):# Define is_intersect to check whether a circle intersects a rectangle; rect_* define rectangle bounds and circle_* define the circle
if circle_x < rect_left:
closest_x = rect_left
elif circle_x > rect_right:
closest_x = rect_right
else:
closest_x = circle_x
if circle_y < rect_top:
closest_y = rect_top
elif circle_y > rect_bottom:
closest_y = rect_bottom
else:
closest_y = circle_y
distance = math.sqrt((closest_x - circle_x) ** 2 + (closest_y - circle_y) ** 2)# Compute the distance between the circle center and the closest point using the Euclidean distance formula
return distance <= circle_radius
def make_occupancy_grid_from_map(map_data, cell_size=5):
'''
This function takes a map and a cell size (the physical size of one "cell" in the grid)
and returns a 2D numpy array,
with each cell containing a '1' if it is occupied and '0' if it is empty
Inputs: map (tuple) - see read_map_from_file for description.
Outputs: occupancy_grid - 2D numpy array
'''
#Your code goes heregrid_size, start_pos, goal_pos, obstacles = map_data
# Unpack map_data: grid_size, start_pos, goal_pos, obstacles
if type(grid_size) == type(0):
# Check whether grid_size is an integer. If so, the grid is square.
columns = math.ceil(grid_size / cell_size)
rows = math.ceil(grid_size / cell_size)
# Both row and column counts equal grid size divided by cell size
else:
columns = math.ceil(grid_size[0] / cell_size)
# Number of columns = width / cell size
rows = math.ceil(grid_size[1] / cell_size)
# Number of rows = height / cell size
zero_array = np.zeros((rows, columns))
# Initialize a 2D array of size rows x columns
for i in range(rows):
for j in range(columns):
# Iterate over every cell (i, j) in the 2D grid
rect_left = j * cell_size
rect_right = (j + 1) * cell_size
rect_top = i * cell_size
rect_bottom = (i + 1) * cell_size
# Compute the cell's left/right/top/bottom boundaries
collide = False
# Initialize collide as False
for circle_x, circle_y, circle_radius in obstacles:
if is_intersect(rect_left, rect_top, rect_right, rect_bottom, circle_x, circle_y, circle_radius):
collide = True
# Use is_intersect to check whether the cell intersects an obstacle; if so, set collide to True
if collide:
# collide is a boolean value and can be used directly in conditionals
zero_array[i, j] = 1
# If collide is True, mark cell (i, j) as occupied (1)
return zero_array
# Return the generated occupancy grid (zero_array)
def make_adjacency_matrix_from_occupancy_grid(occupancy_grid):
# Build the adjacency matrix
'''
This function converts an occupancy grid into an adjacency matrix. We assume that cells are connected to their neighbours unless the
neighbour is occupied.
We also assume that the cost of moving from a cell to a neighbour is always '1' and allow only horizontal and vertical connections (i.e.
no diagonals allowed).
Inputs: occupancy_grid - a 2D (NxN) numpy array. An element with value '1' is occupied, while those with value '0' are empty.
Outputs: A 2D (MxM where M=NxN) array. Element (i,j) contains the cost of travelling from node i to node j in the occupancy grid.
'''
#Your code goes here
dirr = [1, 0, -1, 0]
dirc = [0, 1, 0, -1]
# Define direction arrays for row/column moves: down/right +1, up/left -1
rows, cols = occupancy_grid.shape
# Get the number of rows and columns from occupancy_grid
M = rows * cols
# Compute the total number of cells M in the grid
zero_array = np.zeros((M, M))
# Initialize a zero array of size (M, M)
for i in range(M):
row = i // cols
# Integer-divide i by cols to get the cell's row index
col = i % cols
# Take i modulo cols to get the cell's column index
for j in range(4):
# Direction indices: 0 down, 1 right, 2 up, 3 left
rr = row + dirr[j]
# rr is the new row; dirr[j] is the row offset
cc = col + dirc[j]
# cc is the new column; dirc[j] is the column offset
# Derive the four neighbor coordinates (rr, cc) from (row, col)
if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
# Boundary check to ensure (rr, cc) is within the grid
index = rr * cols + cc
# Convert 2D coordinates to a 1D linear index
if occupancy_grid[rr][cc] == 0 and occupancy_grid[row][col] == 0:
# Check that both the current cell and the neighbor cell are free (0)
zero_array[i, index] = 1
zero_array[index, i] = 1
# Set both directions in the adjacency matrix to 1 (bidirectional connection)
# If both cells are free, connect them in the adjacency matrix
return zero_array
def get_path_from_predecessors(predecessors, map_data, cell_size=5):
# Extract the path from start to goal using the Dijkstra predecessors matrix, and convert grid coordinates to physical coordinates
'''This function takes a predecessors matrix, map_data and cell_size as input and
returns the path from start to goal position.
We take the mid-point of each cell as the (x, y) coordinate for the path.
Inputs: predecessors - a 1D numpy array (size = M = NxN, where N is the length of an occupancy grid) produced by scipy's implementation of
Dijkstra's algorithm.
Each element i tells us the index of the node we should travel to if we are in node j.
map_data - (tuple) see read_map_from_file for description.
cell_size - (integer) the physical size corresponding to a single cell in the grid.
Outputs: path - A list of tuples (x, y), where (x, y) are the coordinates of a position we can travel to in the map.
'''
#Your code goes here
grid_size, start_pos, goal_pos, obstacles = map_data
columns = math.ceil(grid_size[0] / cell_size)
rows = math.ceil(grid_size[1] / cell_size)
start_row = start_pos[0] // cell_size
start_col = start_pos[1] // cell_size
start_index = start_row * columns + start_col
goal_row = goal_pos[0] // cell_size
goal_col = goal_pos[1] // cell_size
goal_index = goal_row * columns + goal_col
# Same logic as around line 156
path = []
# Create an empty list path to store the backtracked indices from goal to start
while True:
path.append(goal_index)
# Append the current goal_index to the path list
if goal_index < 0:
return []
# If the index is negative, the path does not exist or backtracking failed; return an empty list
if goal_index == start_index:
break
# When the goal index equals the start index, backtracking is complete; break out of the loop
goal_index = int(predecessors[start_index][goal_index])
# Look up the predecessor of the current node in predecessors and update goal_index (step back along the path)
path.reverse()
# The path is currently in reverse (goal-to-start); reverse it to get start-to-goal order
res = []
# Convert each path index to physical map coordinates and store in res
for index in path:
row = index // columns
col = index % columns
# Compute the row and column indices
res.append((col * cell_size + cell_size / 2, row * cell_size + cell_size / 2))
# Convert the cell center to physical coordinates (x, y): col offset + half cell width, row offset + half cell height
return res
# Return the converted physical-coordinate path res
def plot_map(ax, map_data):
'''
This function plots a map given a description of the map
Inputs:
ax (matplotlib axis) - the axis the map should be drawn on
map_data - a tuple describing the map. See definition in read_map_from_file function for details.
'''
if map_data:
start_pos = map_data[1]
goal_pos = map_data[2]
obstacles = map_data[3]
ax.plot(goal_pos[0], goal_pos[1], 'r*')
ax.plot(start_pos[0], start_pos[1], 'b*')
for obstacle in obstacles:#Obstacle[0] is x position, [1] is y position and [2] is radius
c_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
ax.add_patch(c_patch)
else:
print("No map data provided- have you implemented read_map_from_file?")
def plot_path(ax, path):
'''
This function plots the path found by your occupancy grid solution.
Inputs: ax (matplotlib axis) - the axis object where the path will be drawn
path (list of tuples) - a list of points (x, y) representing the spatial co-ordinates of a path.
'''
if len(path) == 0:
return
x_coords, y_coords = zip(*path)
ax.plot(x_coords, y_coords, marker='o', linestyle='-')
def plot_path_with_color(ax, path, color):
if len(path) == 0:return
x_coords, y_coords = zip(*path)
ax.plot(x_coords, y_coords, linestyle='-', color=color)
def test_make_occupancy_grid():
map0 = (10, (1, 1), (9, 9), [])
assert np.array_equal(make_occupancy_grid_from_map(map0, cell_size=1), np.zeros((10, 10))), "Test 1 - checking map 0 with cell size 10"
map1 = (10, (1, 1), (9, 9), [(5, 5, 2)])
assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=10), np.array([[1]])), "Test 1 - checking map 1 with cell size 10"
assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=5), np.array([[1, 1], [1, 1]])), "Test 2 - checking map 1 with cell
size 5"
map1_cell_size_2_answer = np.array([[0, 0, 0, 0, 0],
[0, 1, 1, 1, 0],
[0, 1, 1, 1, 0],
[0, 1, 1, 1, 0],
[0, 0, 0, 0, 0]])
assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=2), map1_cell_size_2_answer), "Test 3 - checking map 1 with cell size 2"
map2 = (100, (1, 1), (9, 9), [(10, 10, 5), (90, 90, 5)])
map2_answer = np.array([[1, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 1]])
occupancy_grid1 = np.array([[0, 0, 0],
[0, 0, 0],
[0, 0, 0]])
adjacency_matrix1 = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
[1., 0., 1., 0., 1., 0., 0., 0., 0.],
[0., 1., 0., 0., 0., 1., 0., 0., 0.],
[1., 0., 0., 0., 1., 0., 1., 0., 0.],
[0., 1., 0., 1., 0., 1., 0., 1., 0.],
[0., 0., 1., 0., 1., 0., 0., 0., 1.],
[0., 0., 0., 1., 0., 0., 0., 1., 0.],
[0., 0., 0., 0., 1., 0., 1., 0., 1.],
[0., 0., 0., 0., 0., 1., 0., 1., 0.]])
assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid1), adjacency_matrix1)
occupancy_grid2 = np.array([[1, 1, 1],
[1, 1, 1],
[1, 1, 1]])
adjacency_matrix2 = np.zeros((occupancy_grid2.size, occupancy_grid2.size))
assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid2), adjacency_matrix2)
occupancy_grid3 = np.array([[0, 1, 0],
[0, 1, 0],
[0, 1, 0]])
adjacency_matrix3 = np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 1., 0., 0., 0.],
[1., 0., 0., 0., 0., 0., 1., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 1., 0., 0., 0., 0., 0., 1.],
[0., 0., 0., 1., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 1., 0., 0., 0.]])
assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid3), adjacency_matrix3)
def test_get_path():
map_data = read_map_from_file('map2.csv')
grid = make_occupancy_grid_from_map(map_data, 5)
distance_matrix = make_adjacency_matrix_from_occupancy_grid(grid)
def test_occupancy_grid():
test_make_occupancy_grid()
def occupancy_grid(file, cell_size=5):
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
map_data = read_map_from_file(file)
plot_map(ax, map_data)
grid = make_occupancy_grid_from_map(map_data, cell_size)
distance_matrix = make_adjacency_matrix_from_occupancy_grid(grid)#You'll need to edit the line below to use Scipy's shortest graph function to find the path for us
predecessors = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_array(distance_matrix), directed=False, return_predecessors=True)[1]
path = get_path_from_predecessors(predecessors, map_data, cell_size)
plot_path(ax, path)
plt.show()
'''
==============================
The code below here is for your RRT solution
==============================
'''
# Check whether a circle intersects a line segment (is_intersect_circle_line). If they intersect, return the intersection point(s).
def is_intersect_circle_line(circle_center, radius, line_start, line_end):
a, b = circle_center
r = radius
# Set circle parameters
x1, y1 = line_start
x2, y2 = line_end
# Set the start and end points of the line segment
dx = x2 - x1
dy = y2 - y1
# Compute the direction vector (dx, dy) from the start to the end of the segment
# Set up the equation: circle (x-a)^2+(y-b)^2=r^2 and segment parametric form x=x1+t*dx, y=y1+t*dy; substituting yields a quadratic At^2 + Bt + C = 0
A = dx ** 2 + dy ** 2
# A is the quadratic coefficient, related to the squared length of the direction vector
B = 2 * (dx * (x1 - a) + dy * (y1 - b))
# B is the linear coefficient
C = (x1 - a) ** 2 + (y1 - b) ** 2 - r ** 2
# C is the constant term
D = B ** 2 - 4 * A * C
# D is the discriminant, used to determine the number of solutions
if D < 0:
return []
# If the discriminant is negative, there is no intersection; return an empty list
elif D == 0:
# Discriminant zero: the circle is tangent to the line
if A == 0:
return []
# If A==0, the segment degenerates to a point; return an empty list
t = -B / (2 * A)
# Otherwise compute the single parameter t
if 0 <= t <= 1:
# Check whether t is within [0, 1]
return [(x1 + t * dx, y1 + t * dy)]
# If so, compute the intersection point and return it
else:
return []
# Otherwise return an empty list
else:
sqrt_d = np.sqrt(D)
t1 = (-B + sqrt_d) / (2 * A)
t2 = (-B - sqrt_d) / (2 * A)
# Compute the two possible t values from the quadratic formula
intersections = []
# Store valid intersections in the intersections list
if 0 <= t1 <= 1:
intersections.append((x1 + t1 * dx, y1 + t1 * dy))
if 0 <= t2 <= 1:
intersections.append((x1 + t2 * dx, y1 + t2 * dy))
# Only intersections within the segment (t in [0,1]) are valid; add them to the list
return intersections
# Determine whether a given segment (line_start,line_end) collides with any obstacle
def check_line_collide_with_obstacles(line_start, line_end, obstacles):
for obstacle in obstacles:
# Iterate over each obstacle in the obstacles list
if is_intersect_circle_line((obstacle[0], obstacle[1]), obstacle[2], line_start, line_end):
# Convert obstacle data to (cx, cy, r) and pass to is_intersect_circle_line
return True
# If the segment intersects the current obstacle, return True; otherwise continue
return False
# Compute the distance between p1 and p2 (segment length)
def distance(p1, p2):
x1, y1 = p1
x2, y2 = p2
return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# Find the point in points that is closest to aim_point
def shortest_points(points, aim_point):
res = points[0]
# Initialize the nearest point as the first point in the list
for point in points:# Iterate through each point in the set
if distance(point, aim_point) < distance(res, aim_point):
# If the current point is closer than the current nearest point
res = point
# Update res to the current point
return res
# Compute a new point by moving from point toward trail_point by step_size
def new_point_towards(point, trail_point, step_size):
dx = trail_point[0] - point[0]
dy = trail_point[1] - point[1]
# Differences in x and y between the target and current point
distance = math.sqrt(dx ** 2 + dy ** 2)
# Distance from the current point to the target point
if distance == 0:
return point
# If the points coincide, return the current point (no movement needed)
dx /= distance
dy /= distance
# Normalize to a unit vector
dx *= step_size
dy *= step_size
# Scale the unit vector by the given step size
new_point = (point[0] + dx, point[1] + dy)
# New point = original point plus the scaled vector offset
return new_point
def rrt_algorithm(map_data, step_size, num_points, ax):
parents = {}
# Create an empty dict parents to store the parent of each node in the tree
grid_size, start_pos, goal_pos, obstacles = map_data
# Unpack the map parameters
for obstacle in obstacles:
if distance(start_pos, (obstacle[0], obstacle[1])) < obstacle[2]:
return []
# Initial check: if the start position lies inside any obstacle, return an empty list (using distance to obstacle center)
points = [start_pos]
# Initialize the node list points with start_pos (the tree's nodes)
for _num in range(num_points):
while True:
trail_point = (random.randint(0, grid_size[0]), random.randint(0, grid_size[1]))
# Generate a random point trail_point uniformly within the map bounds
shortest_point = shortest_points(points, trail_point)
# Find the nearest existing node as the extension start
new_point = new_point_towards(shortest_point, trail_point, step_size)
# Use new_point_towards to step toward the random point
if not check_line_collide_with_obstacles(shortest_point, new_point, obstacles):
# Ensure the segment from shortest_point to new_point does not intersect any obstacle
points.append(new_point)
# Add new_point to the points list
plot_path_with_color(ax, [shortest_point, new_point], 'green')
# Plot the new tree edge in green
parents[new_point] = shortest_point
# Update parents: set new_point's parent to shortest_point
ax.plot(trail_point[0], trail_point[1], 'yx')
# Plot the sampled random point (trail_point) as a yellow 'x'
break
shortest_point = shortest_points(points, goal_pos)
# Find the tree node closest to the goal using shortest_points
if check_line_collide_with_obstacles(shortest_point, goal_pos, obstacles):
return []
# Check whether the segment from shortest_point to goal collides with obstacles; if so, return an empty list
plot_path_with_color(ax, [shortest_point, goal_pos], 'green')
parents[goal_pos] = shortest_point
path = []
path.append(goal_pos)
# Initialize path with the goal position
while path[0] != start_pos:
# Continue until the first node in path is the start position
pre = parents[path[0]]
# Get the parent of the first node in path
path.insert(0, pre)
# Insert the parent at the beginning so the path grows backward from goal to start
return path
def rrt(map_file, step_size=10, num_points=100):
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
# Create a Matplotlib figure and an axis without ticks for plotting the map and planning result
map_data = read_map_from_file(map_file)
plot_map(ax, map_data)
# Read the map file and plot the map on ax
path = rrt_algorithm(map_data, step_size, num_points, ax)# Run the RRT algorithm (above) to get a path from start to goal (a list of points)
plot_path(ax, path)
# Plot the resulting path on ax
plt.show()
# Display the plot
return path
def test_rrt():
#Your test functions / statements go here.
path = rrt('map1.csv')
assert len(path) == 0
# Assert that the planned path for map1 is empty
path = rrt('map2.csv')
assert len(path) > 0
# The planned path for map2 should not be empty
'''
==============================
The code below here is used to read arguments from the terminal, allowing us to run different parts of your code.
You should not need to modify this
==============================
'''
def main():
parser = argparse.ArgumentParser(description=" Path planning Assignment for CPA 2024/25")
parser.add_argument('--rrt', action='store_true')
parser.add_argument('-test_rrt', action='store_true')
parser.add_argument('--occupancy', action='store_true')
parser.add_argument('-test_occupancy', action='store_true')
parser.add_argument('-file')
parser.add_argument('-cell_size', type=int)
args = parser.parse_args()
if args.occupancy:
if args.file is None:
print("Error - Occupancy grid requires a map file to be provided as input with -file <filename>")
exit()
else:
if args.cell_size:
occupancy_grid(args.file, args.cell_size)
else:
occupancy_grid(args.file)
if args.test_occupancy:
print("Testing occupancy_grid")
test_occupancy_grid()
if args.test_rrt:
print("Testing RRT")
test_rrt()
if args.rrt:
if args.file is None:
print("Error - RRT requires a map file to be provided as input with -file <filename>")
exit()
else:
rrt(args.file)
if __name__ == "__main__":
main()
