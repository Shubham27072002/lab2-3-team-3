#!/usr/bin/env python3

import rospy
import queue as Queue
import math
import tf

from nav_msgs.msg import OccupancyGrid, GridCells, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point


class AStarPlanner:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("astar_planner")

        # Store map, start cell, goal cell, and inflated obstacle cells
        self.map_data = None
        self.start = None
        self.goal = None
        self.inflated_obstacles = set()

        # Publishers for RViz visualization of A* search states
        self.expanded_pub = rospy.Publisher("/expanded_cells", GridCells, queue_size=1, latch=True)
        self.frontier_pub = rospy.Publisher("/frontier_cells", GridCells, queue_size=1, latch=True)
        self.unexplored_pub = rospy.Publisher("/unexplored_cells", GridCells, queue_size=1, latch=True)

        # Publisher for obstacle padding around real obstacles
        self.inflated_pub = rospy.Publisher("/inflated_obstacles", GridCells, queue_size=1, latch=True)

        # Publishers for final A* path
        self.path_cells_pub = rospy.Publisher("/path_cells", GridCells, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1, latch=True)

        # TF listener is used to get the current robot pose from map to base_footprint
        self.listener = tf.TransformListener()

        # Subscribe to the map topic
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # Old testing method:
        # This subscriber was used when testing A* without moving the real robot.
        # In that version, the start pose came from the RViz 2D Pose Estimate button.
        # rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.start_callback)

        # Final method:
        # The goal comes from RViz, but the start comes from the robot's current TF position.
        rospy.Subscriber("/astar_goal", PoseStamped, self.goal_callback)

        rospy.loginfo("A* planner node started.")
        rospy.spin()

    def map_callback(self, msg):
        # Store the occupancy grid map
        self.map_data = msg

        # Inflate obstacles by adding padding around them.
        # This prevents the robot from planning too close to walls/obstacles.
        self.inflated_obstacles = self.inflate_obstacles(padding_cells=4)

        # Publish inflated obstacle cells so they can be visualized in RViz
        self.publish_gridcells(self.inflated_obstacles, self.inflated_pub)

        rospy.loginfo_once("Map received and inflated obstacles published.")

    def get_robot_start_cell(self):
        # Get the robot's current position from TF.
        # This uses the localized robot pose after AMCL/localization is complete.
        try:
            self.listener.waitForTransform(
                "map",
                "base_footprint",
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            trans, rot = self.listener.lookupTransform(
                "map",
                "base_footprint",
                rospy.Time(0)
            )

            robot_x = trans[0]
            robot_y = trans[1]

            # Convert robot position from world coordinates to grid cell coordinates
            return self.world_to_grid(robot_x, robot_y)

        except Exception as e:
            rospy.logwarn("Could not get robot pose from TF: %s", str(e))
            return None

    # ------------------------------------------------------------------
    # OLD START CALLBACK USED FOR TESTING A* WITHOUT THE REAL ROBOT
    #
    # This section is intentionally commented out.
    # It was used when testing A* in RViz without making the physical
    # TurtleBot3 move. In that version, the user clicked 2D Pose Estimate
    # in RViz and that pose was used as the A* start point.
    #
    # In the final version, the start point is taken from the actual
    # localized robot pose using TF, so this callback is no longer needed.
    # ------------------------------------------------------------------

    # def start_callback(self, msg):
    #     if self.map_data is None:
    #         rospy.logwarn("Map not received yet.")
    #         return

    #     self.start = self.world_to_grid(
    #         msg.pose.pose.position.x,
    #         msg.pose.pose.position.y
    #     )

    #     rospy.loginfo("Start set: %s", str(self.start))
    #     self.try_plan()

    def goal_callback(self, msg):
        # Do not plan unless a map has been received
        if self.map_data is None:
            rospy.logwarn("Map not received yet.")
            return

        # Get current robot position from TF and use it as the A* start cell
        self.start = self.get_robot_start_cell()

        if self.start is None:
            rospy.logwarn("Cannot plan because robot start pose is not available.")
            return

        # Convert RViz goal position from world coordinates to grid coordinates
        self.goal = self.world_to_grid(
            msg.pose.position.x,
            msg.pose.position.y
        )

        rospy.loginfo("Start from robot TF: %s", str(self.start))
        rospy.loginfo("Goal set: %s", str(self.goal))

        # Start planning once both start and goal are available
        self.try_plan()

    def try_plan(self):
        # Make sure map, start, and goal exist before running A*
        if self.map_data is None or self.start is None or self.goal is None:
            return

        # Run A* search
        path, expanded, frontier = self.astar(self.start, self.goal)

        rospy.loginfo("A* complete. Path length: %d", len(path))

    def astar(self, start, goal):
        # Priority queue stores nodes based on lowest cost + heuristic
        open_set = Queue.PriorityQueue()
        open_set.put((0, start))

        # Dictionary to reconstruct final path
        came_from = {start: None}

        # Cost from start node to each visited node
        cost_so_far = {start: 0}

        # Sets used for RViz visualization
        expanded = set()
        frontier = set()
        unexplored = set()

        # Controls how fast RViz updates during search
        rate = rospy.Rate(20)

        # Main A* loop
        while not open_set.empty() and not rospy.is_shutdown():
            # Get the cell with lowest priority
            current_priority, current = open_set.get()

            # Skip if already expanded
            if current in expanded:
                continue

            # Mark current cell as expanded
            expanded.add(current)

            # Remove current cell from frontier once it is expanded
            if current in frontier:
                frontier.remove(current)

            # Stop search if goal is reached
            if current == goal:
                break

            x, y = current

            # 8-connected neighbors:
            # horizontal, vertical, and diagonal movement
            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y - 1)
            ]

            for next_node in neighbors:
                # Ignore occupied, unknown, out-of-bounds, or inflated obstacle cells
                if not self.is_cell_free(next_node):
                    continue

                # Each move currently has uniform cost
                new_cost = cost_so_far[current] + 1

                # Update this node if it is new or if a cheaper path is found
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost

                    # A* priority = path cost so far + heuristic distance to goal
                    priority = new_cost + self.heuristic(next_node, goal)

                    # Add neighbor to priority queue
                    open_set.put((priority, next_node))

                    # Store parent for final path reconstruction
                    came_from[next_node] = current

                    # Mark node as frontier if not already expanded
                    if next_node not in expanded:
                        frontier.add(next_node)

            # Update unexplored cells for RViz visualization
            unexplored.clear()
            width = self.map_data.info.width
            height = self.map_data.info.height

            for y in range(height):
                for x in range(width):
                    cell = (x, y)
                    if self.is_cell_free(cell) and cell not in expanded and cell not in frontier:
                        unexplored.add(cell)

            # Publish real-time A* search state to RViz
            self.publish_gridcells(expanded, self.expanded_pub)
            self.publish_gridcells(frontier, self.frontier_pub)
            self.publish_gridcells(unexplored, self.unexplored_pub)

            rate.sleep()

        # Reconstruct final path from came_from dictionary
        path = self.reconstruct_path(came_from, start, goal)

        # Publish final search result and final path
        self.publish_gridcells(expanded, self.expanded_pub)
        self.publish_gridcells(frontier, self.frontier_pub)
        self.publish_gridcells(unexplored, self.unexplored_pub)
        self.publish_path(path)

        return path, expanded, frontier

    def heuristic(self, node, goal):
        # Euclidean distance heuristic
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    def is_cell_free(self, cell):
        # Check whether a grid cell is safe for A* to use
        x, y = cell

        width = self.map_data.info.width
        height = self.map_data.info.height

        # Reject cells outside the map
        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        index = y * width + x
        value = self.map_data.data[index]

        # Reject inflated obstacle cells
        if cell in self.inflated_obstacles:
            return False

        # OccupancyGrid values:
        # -1 = unknown
        #  0 = free
        # 100 = occupied
        return value == 0

    def reconstruct_path(self, came_from, start, goal):
        # If goal was never reached, no path exists
        if goal not in came_from:
            rospy.logwarn("No path found.")
            return []

        current = goal
        path = []

        # Backtrack from goal to start using parent links
        while current is not None:
            path.append(current)
            current = came_from[current]

        # Reverse so path goes from start to goal
        path.reverse()
        return path

    def world_to_grid(self, x, y):
        # Convert real-world map coordinates in meters to grid cell indices
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution

        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        return grid_x, grid_y

    def grid_to_world(self, cell):
        # Convert grid cell indices back to real-world map coordinates in meters
        x, y = cell

        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution

        # Add 0.5 so the point is placed at the center of the grid cell
        world_x = origin_x + (x + 0.5) * resolution
        world_y = origin_y + (y + 0.5) * resolution

        return world_x, world_y

    def inflate_obstacles(self, padding_cells=3):
        # Create padding around obstacles to account for robot width
        inflated = set()
        original_obstacles = set()

        width = self.map_data.info.width
        height = self.map_data.info.height

        # First find all original occupied cells in the map
        for y in range(height):
            for x in range(width):
                index = y * width + x
                value = self.map_data.data[index]

                if value > 50:
                    original_obstacles.add((x, y))

        # Add padding cells around every original obstacle
        for obs in original_obstacles:
            ox, oy = obs

            for dy in range(-padding_cells, padding_cells + 1):
                for dx in range(-padding_cells, padding_cells + 1):
                    nx = ox + dx
                    ny = oy + dy

                    if 0 <= nx < width and 0 <= ny < height:
                        inflated.add((nx, ny))

        # Remove original obstacle cells from this display layer.
        # The original obstacles are already visible on the map itself.
        inflated_padding_only = inflated - original_obstacles

        return inflated_padding_only

    def publish_gridcells(self, cells, publisher):
        # Publish a set of grid cells as nav_msgs/GridCells for RViz
        msg = GridCells()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        msg.cell_width = self.map_data.info.resolution
        msg.cell_height = self.map_data.info.resolution

        # Convert each grid cell to a real-world point
        for cell in cells:
            x, y = self.grid_to_world(cell)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            msg.cells.append(point)

        publisher.publish(msg)

    def publish_path(self, path):
        # Publish final A* path as nav_msgs/Path
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        for cell in path:
            x, y = self.grid_to_world(cell)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        # Also publish the path as GridCells so it is easier to see in RViz
        self.publish_gridcells(path, self.path_cells_pub)

        # Publish actual Path message for waypoint maker
        self.path_pub.publish(msg)


if __name__ == "__main__":
    try:
        AStarPlanner()
    except rospy.ROSInterruptException:
        pass