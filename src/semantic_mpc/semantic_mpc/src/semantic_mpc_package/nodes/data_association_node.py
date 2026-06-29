import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel
from scipy.spatial import cKDTree
from semantic_mpc.srv import GetTreesPoses
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
import tf.transformations as tf_trans
import message_filters


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def weight_value(n_elements, mean_score, midpoint=5., steepness=10.):
    val = (mean_score - 0.5) * (0.5 + 0.5*np.tanh(steepness*(n_elements - midpoint)))
    return np.round(val, 2)


def load_params():
    return {
        "detections_topic": rospy.get_param("~detections_topic", "/yolov7/detect"),
        "depth_image_topic": rospy.get_param("~depth_image_topic", "camera/depth/image/compressed"),
        "camera_info_topic": rospy.get_param("~camera_info_topic", "camera/depth/camera_info"),
        "tree_scores_topic": rospy.get_param("~tree_scores_topic", "tree_scores"),
        "scores_markers_topic": rospy.get_param("~scores_markers_topic", "scores_markers"),
        "fruits_markers_topic": rospy.get_param("~fruits_markers_topic", "fruits_markers"),
        "tree_service": rospy.get_param("~tree_service", "/obj_pose_srv"),
        "map_frame": rospy.get_param("~map_frame", "map"),
        "base_frame": rospy.get_param("~base_frame", "drone_base_link"),
        "depth_camera_frame": rospy.get_param("~depth_camera_frame", "depth_camera_frame"),
        "sync_queue_size": int(rospy.get_param("~sync_queue_size", 10)),
        "sync_slop": float(rospy.get_param("~sync_slop", 0.2)),
        "scores_queue_size": int(rospy.get_param("~scores_queue_size", 1)),
        "markers_queue_size": int(rospy.get_param("~markers_queue_size", 1)),
        "publish_visualization": _as_bool(rospy.get_param("~publish_visualization", True)),
        "publish_score_markers": _as_bool(rospy.get_param("~publish_score_markers", False)),
        "publish_fruit_markers": _as_bool(rospy.get_param("~publish_fruit_markers", True)),
        "publish_path": _as_bool(rospy.get_param("~publish_path", False)),
        "association_distance": float(rospy.get_param("~association_distance", 2.5)),
        "observation_distance": float(rospy.get_param("~observation_distance", 8.0)),
        "default_tree_score": float(rospy.get_param("~default_tree_score", 0.5)),
        "score_midpoint": float(rospy.get_param("~score_midpoint", 5.0)),
        "score_steepness": float(rospy.get_param("~score_steepness", 10.0)),
        "ripe_class_id": int(rospy.get_param("~ripe_class_id", 2)),
        "raw_class_label": rospy.get_param("~raw_class_label", "raw"),
        "ripe_class_label": rospy.get_param("~ripe_class_label", "ripe"),
        "depth_min_distance": float(rospy.get_param("~depth_min_distance", 0.05)),
        "depth_max_distance": float(rospy.get_param("~depth_max_distance", 20.0)),
        "depth_valid_threshold": int(rospy.get_param("~depth_valid_threshold", 11)),
        "fruit_marker_lifetime": float(rospy.get_param("~fruit_marker_lifetime", 0.2)),
        "fruit_marker_scale": float(rospy.get_param("~fruit_marker_scale", 0.1)),
        "score_marker_lifetime": float(rospy.get_param("~score_marker_lifetime", 0.2)),
    }


class DataAssociationNode:
    def __init__(self):
        rospy.init_node('bounding_box_3d_pose', anonymous=True)
        self.params = load_params()

        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.depth_image = None
        self.tree_poses = None
        self.publish_visualization = self.params["publish_visualization"]
        self.publish_score_markers = self.publish_visualization and self.params["publish_score_markers"]
        self.publish_fruit_markers = self.publish_visualization and self.params["publish_fruit_markers"]

        detection_sub = message_filters.Subscriber(self.params["detections_topic"], Detection2DArray)
        depth_image_sub = message_filters.Subscriber(self.params["depth_image_topic"], CompressedImage)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [detection_sub, depth_image_sub],
            queue_size=self.params["sync_queue_size"],
            slop=self.params["sync_slop"],
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.camera_info_sub = rospy.Subscriber(
            self.params["camera_info_topic"],
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )
        self.scores_pub = rospy.Publisher(
            self.params["tree_scores_topic"],
            Float32MultiArray,
            queue_size=self.params["scores_queue_size"],
        )
        
        self.marker_scores_pub = None
        self.marker_fruits_pub = None
        if self.publish_score_markers:
            self.marker_scores_pub = rospy.Publisher(
                self.params["scores_markers_topic"],
                MarkerArray,
                queue_size=self.params["markers_queue_size"],
            )
        if self.publish_fruit_markers:
            self.marker_fruits_pub = rospy.Publisher(
                self.params["fruits_markers_topic"],
                MarkerArray,
                queue_size=self.params["markers_queue_size"],
            )
        
        self.cam_model = PinholeCameraModel()
        self.tf_listener = tf.TransformListener()
        
        rospy.wait_for_service(self.params["tree_service"])
        self.get_trees_poses = rospy.ServiceProxy(self.params["tree_service"], GetTreesPoses)
        self.update_tree_poses()
        
        rospy.spin()

    def update_tree_poses(self):
        try:
            response = self.get_trees_poses()
            self.tree_poses = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_matrix = np.array(self.camera_info.K).reshape(3, 3)
        self.cam_model.fromCameraInfo(msg)
    
    def uint8_to_distance(self, value, min_dist, max_dist):
        value = max(0, min(value, 255))
        fraction = value / 255.0
        distance = max_dist - fraction * (max_dist - min_dist)
        return distance
    
    def associate_fruits_to_trees(self, fruit_positions, fruit_classes, fruit_scores):
        if self.tree_poses is None or len(fruit_positions) == 0:
            return {}
        
        tree_kdtree = cKDTree(self.tree_poses)
        distances, tree_indices = tree_kdtree.query(fruit_positions[:, :2])

        tree_fruit_dict = {i: {"ripe": [], "raw": []} for i in range(len(self.tree_poses))}
        
        for fruit_index, (tree_index, distance) in enumerate(zip(tree_indices, distances)):
            if distance <= self.params["association_distance"]:
                fruit_class = self.params["ripe_class_label"] if fruit_classes[fruit_index] == self.params["ripe_class_label"] else self.params["raw_class_label"]
                tree_fruit_dict[tree_index][fruit_class].append(fruit_scores[fruit_index])
        
        return tree_fruit_dict

    def transform_fruit_positions(self, fruit_positions, header):
        map_fruits_positions = []
        for fruit_pos in fruit_positions:
            point_camera = PointStamped()
            point_camera.point = Point(*fruit_pos)
            point_camera.header.frame_id = self.params["depth_camera_frame"]
            try:
                point_map = self.tf_listener.transformPoint(self.params["map_frame"], point_camera)
                map_fruits_positions.append([point_map.point.x, point_map.point.y, point_map.point.z])
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)
                continue
        return np.array(map_fruits_positions)

    def synchronized_callback(self, detection_msg, depth_image_msg):
        if self.camera_matrix is None or self.tree_poses is None:
            return
        
        try:
            # Extract the depth channel
            self.depth_image = self.bridge.compressed_imgmsg_to_cv2(
                depth_image_msg,
                desired_encoding="passthrough"
            )[:, :, 0]
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        fruit_positions, fruit_scores, fruit_classes = [], [], []

        # Project detections into 3D
        for detection in detection_msg.detections:
            bbox = detection.bbox
            xmin = int(bbox.center.x - bbox.size_x / 2)
            xmax = int(bbox.center.x + bbox.size_x / 2)
            ymin = int(bbox.center.y - bbox.size_y / 2)
            ymax = int(bbox.center.y + bbox.size_y / 2)

            xmin, xmax = max(0, xmin), min(self.depth_image.shape[1], xmax)
            ymin, ymax = max(0, ymin), min(self.depth_image.shape[0], ymax)

            # Sample the center pixel for depth
            cx = int(np.clip(bbox.center.x, 0, self.depth_image.shape[1] - 1))
            cy = int(np.clip(bbox.center.y, 0, self.depth_image.shape[0] - 1))
            depth_value = self.depth_image[cy, cx]
            if np.isscalar(depth_value):
                if depth_value <= self.params["depth_valid_threshold"]:
                    continue
                median_depth = float(depth_value)
            else:
                valid_depths = depth_value[depth_value > self.params["depth_valid_threshold"]]
                if len(valid_depths) == 0:
                    continue
                median_depth = float(np.median(valid_depths))

            if median_depth <= self.params["depth_valid_threshold"]:
                continue
            
            ray = np.array(self.cam_model.projectPixelTo3dRay((cx, cy)))
            distance = self.uint8_to_distance(
                median_depth,
                self.params["depth_min_distance"],
                self.params["depth_max_distance"],
            )
            XYZ = ray * distance

            fruit_positions.append(XYZ)
            fruit_scores.append(detection.results[0].score)
            fruit_classes.append(
                self.params["ripe_class_label"] if detection.results[0].id == self.params["ripe_class_id"] else self.params["raw_class_label"]
            )

        fruit_positions = np.array(fruit_positions)

        # Transform to map frame and associate fruits to trees
        map_fruits_positions = self.transform_fruit_positions(
            fruit_positions, detection_msg.header
        )
        associated_fruits = self.associate_fruits_to_trees(
            map_fruits_positions, fruit_classes, fruit_scores
        )

        # Get drone position for distance-based scoring
        try:
            (drone_trans, drone_rot) = self.tf_listener.lookupTransform(
                self.params["map_frame"], self.params["base_frame"], rospy.Time()
            )
            drone_x, drone_y = drone_trans[0], drone_trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Could not get drone position, skipping distance check.")
            drone_x, drone_y = None, None

        # Initialize base scores for each tree
        tree_scores = np.ones(len(self.tree_poses)) * self.params["default_tree_score"]

        # Update tree scores
        for i, fruits in associated_fruits.items():
            ripe_scores = fruits.get("ripe", [])
            raw_scores  = fruits.get("raw", [])
            tree_x, tree_y = self.tree_poses[i]

            if drone_x is not None:
                dist = np.hypot(drone_x - tree_x, drone_y - tree_y)
                if dist < self.params["observation_distance"]:
                    ripe_value = weight_value(
                        len(ripe_scores),
                        np.mean(ripe_scores) if ripe_scores else 0,
                        midpoint=self.params["score_midpoint"],
                        steepness=self.params["score_steepness"],
                    )
                    raw_value = weight_value(
                        len(raw_scores),
                        np.mean(raw_scores) if raw_scores else 0,
                        midpoint=self.params["score_midpoint"],
                        steepness=self.params["score_steepness"],
                    )
                    tree_scores[i] = ripe_value - raw_value + self.params["default_tree_score"]

        # Build a 2-column array: [score, -score] per tree
        scores_with_neg = np.stack([tree_scores, 1-tree_scores], axis=1)  # shape (N,2)

        # Publish as a 2D multiarray
        msg = Float32MultiArray()
        dim0 = MultiArrayDimension(
            label="tree",
            size=scores_with_neg.shape[0],
            stride=scores_with_neg.shape[0] * scores_with_neg.shape[1]
        )
        dim1 = MultiArrayDimension(
            label="type",
            size=2,
            stride=scores_with_neg.shape[1]
        )
        msg.layout.dim = [dim0, dim1]
        msg.data = scores_with_neg.flatten().tolist()

        self.scores_pub.publish(msg)

        if self.publish_score_markers:
            markers = MarkerArray()
            for i, (tree_pos, score) in enumerate(zip(self.tree_poses, tree_scores)):
                score = float(np.clip(score, 0.0, 1.0))
                marker = Marker()
                marker.header = detection_msg.header
                marker.header.frame_id = self.params["map_frame"]
                marker.ns = "tree_score_markers"
                marker.id = i
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.MODIFY
                marker.pose.position.x = tree_pos[0]
                marker.pose.position.y = tree_pos[1]
                marker.pose.position.z = 3.2
                marker.pose.orientation.w = 1.0
                marker.lifetime = rospy.Duration(self.params["score_marker_lifetime"])
                marker.scale.z = 0.35
                marker.color.a = 1.0
                marker.color.r = 1.0 - score
                marker.color.g = score
                marker.color.b = 0.0
                marker.text = "{:.2f}".format(score)
                markers.markers.append(marker)
            self.marker_scores_pub.publish(markers)

        if self.publish_fruit_markers:
            markers = MarkerArray()
            for i, (fruit_pos, score) in enumerate(zip(map_fruits_positions, fruit_scores)):
                fruit_marker = Marker()
                fruit_marker.header = detection_msg.header
                fruit_marker.header.frame_id = self.params["map_frame"]
                fruit_marker.ns = "fruit_markers"
                fruit_marker.id =  len(self.tree_poses) * 2 + i * 2
                fruit_marker.type = Marker.SPHERE
                fruit_marker.action = Marker.MODIFY
                fruit_marker.pose.position.x = fruit_pos[0]
                fruit_marker.pose.position.y = fruit_pos[1]
                fruit_marker.pose.position.z = fruit_pos[2]
                fruit_marker.pose.orientation.w = 1.0
                fruit_marker.lifetime = rospy.Duration(self.params["fruit_marker_lifetime"])
                fruit_marker.scale.x = fruit_marker.scale.y = fruit_marker.scale.z = self.params["fruit_marker_scale"]
                fruit_marker.color.a = 1.0
                fruit_marker.color.r = 1.0 if fruit_classes[i] == self.params["ripe_class_label"] else 0.0
                fruit_marker.color.g = 1.0 if fruit_classes[i] == self.params["raw_class_label"] else 0.0
                fruit_marker.color.b = 0.0
                markers.markers.append(fruit_marker)
            self.marker_fruits_pub.publish(markers)

def main():
    DataAssociationNode()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
