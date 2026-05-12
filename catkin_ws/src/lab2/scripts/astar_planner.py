#!/usr/bin/env python3

import rospy
import queue as Queue
import math
import tf

from nav_msgs.msg import OccupancyGrid, GridCells, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point


class AStarPlanner:
    def __init__(self):
        rospy.init_node("astar_planner")

        self.map_data = None
        self.start = None
        self.goal = None
        self.inflated_obstacles = set()

        self.expanded_pub = rospy.Publisher("/expanded_cells", GridCells, queue_size=1, latch=True)
        self.frontier_pub = rospy.Publisher("/frontier_cells", GridCells, queue_size=1, latch=True)
        self.unexplored_pub = rospy.Publisher("/unexplored_cells", GridCells, queue_size=1, latch=True)
        self.inflated_pub = rospy.Publisher("/inflated_obstacles", GridCells, queue_size=1, latch=True)
        self.path_cells_pub = rospy.Publisher("/path_cells", GridCells, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1, latch=True)

        self.listener = tf.TransformListener()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        # rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.start_callback)
        rospy.Subscriber("/astar_goal", PoseStamped, self.goal_callback)

        rospy.loginfo("A* planner node started.")
        rospy.spin()

    def map_callback(self, msg):
        self.map_data = msg
        self.inflated_obstacles = self.inflate_obstacles(padding_cells=4)
        self.publish_gridcells(self.inflated_obstacles, self.inflated_pub)
        rospy.loginfo_once("Map received and inflated obstacles published.")

    def get_robot_start_cell(self):
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

            return self.world_to_grid(robot_x, robot_y)

        except Exception as e:
            rospy.logwarn("Could not get robot pose from TF: %s", str(e))
            return None

    # def start_callback(self, msg):
        # if self.map_data is None:
        #     rospy.logwarn("Map not received yet.")
        #     return

        # self.start = self.world_to_grid(
        #     msg.pose.pose.position.x,
        #     msg.pose.pose.position.y
        # )

        # rospy.loginfo("Start set: %s", str(self.start))
        # self.try_plan()

    def goal_callback(self, msg):
        if self.map_data is None:
            rospy.logwarn("Map not received yet.")
            return

        self.start = self.get_robot_start_cell()

        if self.start is None:
            rospy.logwarn("Cannot plan because robot start pose is not available.")
            return

        self.goal = self.world_to_grid(
            msg.pose.position.x,
            msg.pose.position.y
        )

        rospy.loginfo("Start from robot TF: %s", str(self.start))
        rospy.loginfo("Goal set: %s", str(self.goal))

        self.try_plan()

    def try_plan(self):
        if self.map_data is None or self.start is None or self.goal is None:
            return

        path, expanded, frontier = self.astar(self.start, self.goal)

        # self.publish_gridcells(expanded, self.expanded_pub)
        # self.publish_gridcells(frontier, self.frontier_pub)
        # self.publish_path(path)

        rospy.loginfo("A* complete. Path length: %d", len(path))

    def astar(self, start, goal):
        open_set = Queue.PriorityQueue()
        open_set.put((0, start))

        came_from = {start: None}
        cost_so_far = {start: 0}

        expanded = set()
        frontier = set()
        unexplored = set()

        rate = rospy.Rate(20)   # controls RViz update speed

        while not open_set.empty() and not rospy.is_shutdown():
            current_priority, current = open_set.get()

            if current in expanded:
                continue

            expanded.add(current)

            if current in frontier:
                frontier.remove(current)

            if current == goal:
                break

            x, y = current
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
                if not self.is_cell_free(next_node):
                    continue

                new_cost = cost_so_far[current] + 1

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)

                    open_set.put((priority, next_node))
                    came_from[next_node] = current

                    if next_node not in expanded:
                        frontier.add(next_node)

            # unexplored = free cells not expanded and not frontier
            unexplored.clear()
            width = self.map_data.info.width
            height = self.map_data.info.height

            for y in range(height):
                for x in range(width):
                    cell = (x, y)
                    if self.is_cell_free(cell) and cell not in expanded and cell not in frontier:
                        unexplored.add(cell)

            # publish real-time search state
            self.publish_gridcells(expanded, self.expanded_pub)
            self.publish_gridcells(frontier, self.frontier_pub)
            self.publish_gridcells(unexplored, self.unexplored_pub)

            rate.sleep()

        path = self.reconstruct_path(came_from, start, goal)

        # publish final result
        self.publish_gridcells(expanded, self.expanded_pub)
        self.publish_gridcells(frontier, self.frontier_pub)
        self.publish_gridcells(unexplored, self.unexplored_pub)
        self.publish_path(path)

        return path, expanded, frontier

    def heuristic(self, node, goal):
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    def is_cell_free(self, cell):
        x, y = cell

        width = self.map_data.info.width
        height = self.map_data.info.height

        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        index = y * width + x
        value = self.map_data.data[index]

        # -1 = unknown, 0 = free, 100 = occupied
        if cell in self.inflated_obstacles:
            return False

        return value == 0

    def reconstruct_path(self, came_from, start, goal):
        if goal not in came_from:
            rospy.logwarn("No path found.")
            return []

        current = goal
        path = []

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path

    def world_to_grid(self, x, y):
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution

        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        return grid_x, grid_y

    def grid_to_world(self, cell):
        x, y = cell

        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution

        world_x = origin_x + (x + 0.5) * resolution
        world_y = origin_y + (y + 0.5) * resolution

        return world_x, world_y
    
    def inflate_obstacles(self, padding_cells=3):
        inflated = set()
        original_obstacles = set()

        width = self.map_data.info.width
        height = self.map_data.info.height

        for y in range(height):
            for x in range(width):
                index = y * width + x
                value = self.map_data.data[index]

                if value > 50:
                    original_obstacles.add((x, y))

        for obs in original_obstacles:
            ox, oy = obs

            for dy in range(-padding_cells, padding_cells + 1):
                for dx in range(-padding_cells, padding_cells + 1):
                    nx = ox + dx
                    ny = oy + dy

                    if 0 <= nx < width and 0 <= ny < height:
                        inflated.add((nx, ny))

        # remove actual obstacle cells from inflated display
        inflated_padding_only = inflated - original_obstacles

        return inflated_padding_only

    def publish_gridcells(self, cells, publisher):
        msg = GridCells()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        msg.cell_width = self.map_data.info.resolution
        msg.cell_height = self.map_data.info.resolution

        for cell in cells:
            x, y = self.grid_to_world(cell)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            msg.cells.append(point)

        publisher.publish(msg)

    def publish_path(self, path):
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
        self.publish_gridcells(path, self.path_cells_pub)
        self.path_pub.publish(msg)


if __name__ == "__main__":
    try:
        AStarPlanner()
    except rospy.ROSInterruptException:
        pass