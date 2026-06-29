#!/usr/bin/env python3
"""
ROS Bridge Node: Connects external RL agents with Unity simulator via gymnasium environment.

Architecture:
  RL Agent (external) --[action]--> ROS Bridge --[env.step()]--> Gymnasium Env
                                         |
                                         +--[publish reward/done/info]--> RL Agent
  
  Unity Simulator --[publishes state]--> ROS Bridge (optional observation bridge)

Simple flow:
  1. RL agent publishes action to /rl/action
  2. Bridge receives action, steps environment
  3. Bridge publishes reward, done, info to /rl/reward, /rl/done, /rl/info
"""
import os
import numpy as np
import rospy
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray
from active_rl_classification.env import TreeClassificationEnv


def make_float32_multiarray(data):
    """Convert numpy array or list to Float32MultiArray message."""
    msg = Float32MultiArray()
    if isinstance(data, np.ndarray):
        msg.data = data.flatten().astype(np.float32).tolist()
    else:
        msg.data = data
    return msg


def make_int32_multiarray(data):
    """Convert numpy array or list to Int32MultiArray message."""
    msg = Int32MultiArray()
    if isinstance(data, np.ndarray):
        msg.data = data.flatten().astype(np.int32).tolist()
    else:
        msg.data = data
    return msg


class RosRLBridge:
    """
    Simple ROS bridge for gymnasium environment.
    Receives actions from RL agent, publishes reward/done/info.
    """
    
    def __init__(self):
        rospy.init_node('active_rl_classification_node')
        
        # Load environment configuration from ROS parameters
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_raw = os.path.join(pkg_root, 'data', 'RawData.csv')
        default_ripe = os.path.join(pkg_root, 'data', 'RipeData.csv')
        
        raw_csv = rospy.get_param('~raw_csv', default_raw)
        ripe_csv = rospy.get_param('~ripe_csv', default_ripe)
        use_oracle = rospy.get_param('~use_oracle', True)
        obs_range = rospy.get_param('~obs_range', 5.0)
        k_obs = rospy.get_param('~k_obs', 5)
        layout = rospy.get_param('~layout', 'random')
        side = rospy.get_param('~side', 25.0)
        horizon = rospy.get_param('~horizon', 100)
        
        # Topic names (configurable via ROS params)
        action_topic = rospy.get_param('~action_topic', 'rl/action')
        reward_topic = rospy.get_param('~reward_topic', 'rl/reward')
        done_topic = rospy.get_param('~done_topic', 'rl/done')
        info_topic = rospy.get_param('~info_topic', 'rl/info')
        obs_location_topic = rospy.get_param('~obs_location_topic', 'rl/obs/location')
        obs_belief_topic = rospy.get_param('~obs_belief_topic', 'rl/obs/belief')
        obs_measurement_topic = rospy.get_param('~obs_measurement_topic', 'rl/obs/measurement')
        obs_tracked_topic = rospy.get_param('~obs_tracked_topic', 'rl/obs/tracked')
        obs_mask_topic = rospy.get_param('~obs_mask_topic', 'rl/obs/mask')
        
        # Create environment
        self.env = TreeClassificationEnv({
            'perception_csvs': [raw_csv, ripe_csv],
            'use_oracle': use_oracle,
            'obs_range': obs_range,
            'k_obs': k_obs,
            'layout': layout,
            'side': side,
            'horizon': horizon,
        })
        
        # Publishers: send results back to RL agent
        self.pub_reward = rospy.Publisher(reward_topic, Float32, queue_size=1)
        self.pub_done = rospy.Publisher(done_topic, Bool, queue_size=1)
        self.pub_info = rospy.Publisher(info_topic, Float32MultiArray, queue_size=1)
        
        # Publishers: send observations for RL agent
        self.pub_obs_location = rospy.Publisher(obs_location_topic, Float32MultiArray, queue_size=1)
        self.pub_obs_belief = rospy.Publisher(obs_belief_topic, Float32MultiArray, queue_size=1)
        self.pub_obs_measurement = rospy.Publisher(obs_measurement_topic, Float32MultiArray, queue_size=1)
        self.pub_obs_tracked = rospy.Publisher(obs_tracked_topic, Float32MultiArray, queue_size=1)
        self.pub_obs_mask = rospy.Publisher(obs_mask_topic, Float32MultiArray, queue_size=1)
        
        # Subscriber: receive action from RL agent
        self.sub_action = rospy.Subscriber(action_topic, Float32MultiArray, self._on_action)
        
        # Reset environment and publish initial observation
        obs, _ = self.env.reset()
        self.pub_obs_location.publish(make_float32_multiarray(obs['location']))
        self.pub_obs_belief.publish(make_float32_multiarray(obs['belief']))
        self.pub_obs_measurement.publish(make_float32_multiarray(obs['measurement']))
        self.pub_obs_tracked.publish(make_float32_multiarray(obs['tracked']))
        self.pub_obs_mask.publish(make_float32_multiarray(obs['mask']))
        
        rospy.loginfo('ROS-RL Bridge initialized. Listening on %s', action_topic)
        rospy.loginfo('Publishing observations to: %s/*, %s, %s, %s', 
                     obs_location_topic.rsplit('/', 1)[0], reward_topic, done_topic, info_topic)
    
    def _on_action(self, msg):
        """Callback when RL agent publishes action."""
        try:
            # Parse action from message
            action = np.array(msg.data[:3], dtype=np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Publish observations for RL agent
            self.pub_obs_location.publish(make_float32_multiarray(obs['location']))
            self.pub_obs_belief.publish(make_float32_multiarray(obs['belief']))
            self.pub_obs_measurement.publish(make_float32_multiarray(obs['measurement']))
            self.pub_obs_tracked.publish(make_float32_multiarray(obs['tracked']))
            self.pub_obs_mask.publish(make_float32_multiarray(obs['mask']))
            
            # Publish results
            self.pub_reward.publish(Float32(reward))
            self.pub_done.publish(Bool(done))
            
            # Flatten info dict: [num_tracked, episode_length, success]
            info_array = np.array([
                info.get('num_tracked', 0),
                info.get('episode_length', 0),
                float(info.get('success', False))
            ], dtype=np.float32)
            self.pub_info.publish(make_float32_multiarray(info_array))
            
            # Reset if episode ended
            if done:
                obs, _ = self.env.reset()
                # Publish initial observation for next episode
                self.pub_obs_location.publish(make_float32_multiarray(obs['location']))
                self.pub_obs_belief.publish(make_float32_multiarray(obs['belief']))
                self.pub_obs_measurement.publish(make_float32_multiarray(obs['measurement']))
                self.pub_obs_tracked.publish(make_float32_multiarray(obs['tracked']))
                self.pub_obs_mask.publish(make_float32_multiarray(obs['mask']))
                rospy.loginfo('Episode done | Reward: %.3f | Tracked: %d | Success: %s',
                            reward, info.get('num_tracked', 0), info.get('success', False))
        
        except Exception as e:
            rospy.logerr('Error stepping environment: %s', str(e))
    
    def spin(self):
        """Keep node running."""
        rospy.spin()


if __name__ == '__main__':
    try:
        bridge = RosRLBridge()
        bridge.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('ROS RL Bridge shutting down')

