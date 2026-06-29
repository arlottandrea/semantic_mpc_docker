import os
import io
import zipfile
import numpy as np
import rospy
import torch as th
from gymnasium import spaces
from std_msgs.msg import Bool, Float32, Float32MultiArray
from stable_baselines3 import PPO, A2C, DQN

# Explicitly import your custom network structures from your package
from active_rl_classification.model import TreeClassFeatureExtractor


def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RLAgentNode:
    """RL Agent that deploys a trained policy via ROS topics."""
    
    def __init__(self, policy_path, policy_type='PPO'):
        rospy.init_node('rl_agent_node')
        
        self.policy_path = policy_path
        self.policy_type = policy_type
        
        # Map policy type
        self.policy_classes = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN,
        }
        
        if self.policy_type not in self.policy_classes:
            rospy.logerr('Unknown policy type: %s. Use one of: %s', 
                        self.policy_type, list(self.policy_classes.keys()))
            raise ValueError(f'Unknown policy type: {self.policy_type}')
        
        # Policy instance will be built lazily on the first observation step
        self.model = None
        
        # Topic names (configurable via ROS params)
        obs_location_topic = rospy.get_param('~obs_location_topic', 'rl/obs/location')
        obs_belief_topic = rospy.get_param('~obs_belief_topic', 'rl/obs/belief')
        obs_measurement_topic = rospy.get_param('~obs_measurement_topic', 'rl/obs/measurement')
        obs_tracked_topic = rospy.get_param('~obs_tracked_topic', 'rl/obs/tracked')
        obs_mask_topic = rospy.get_param('~obs_mask_topic', 'rl/obs/mask')
        action_topic = rospy.get_param('~action_topic', 'rl/action')
        reward_topic = rospy.get_param('~reward_topic', 'rl/reward')
        done_topic = rospy.get_param('~done_topic', 'rl/done')
        info_topic = rospy.get_param('~info_topic', 'rl/info')
        
        # Control parameters
        self.step_frequency = rospy.get_param('~step_frequency', 10.0)  # Hz
        self.deterministic = rospy.get_param('~deterministic', True)    # Deterministic or stochastic
        
        # Publishers: send actions to bridge
        self.pub_action = rospy.Publisher(action_topic, Float32MultiArray, queue_size=1)
        
        # Subscribers: receive observations and feedback from bridge
        self.sub_obs_location = rospy.Subscriber(obs_location_topic, Float32MultiArray, self.on_obs_location)
        self.sub_obs_belief = rospy.Subscriber(obs_belief_topic, Float32MultiArray, self.on_obs_belief)
        self.sub_obs_measurement = rospy.Subscriber(obs_measurement_topic, Float32MultiArray, self.on_obs_measurement)
        self.sub_obs_tracked = rospy.Subscriber(obs_tracked_topic, Float32MultiArray, self.on_obs_tracked)
        self.sub_obs_mask = rospy.Subscriber(obs_mask_topic, Float32MultiArray, self.on_obs_mask)
        self.sub_reward = rospy.Subscriber(reward_topic, Float32, self.on_reward)
        self.sub_done = rospy.Subscriber(done_topic, Bool, self.on_done)
        self.sub_info = rospy.Subscriber(info_topic, Float32MultiArray, self.on_info)
        
        # State: store latest observations
        self.obs = {
            'location': None,
            'belief': None,
            'measurement': None,
            'tracked': None,
            'mask': None,
        }
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_episodes = 0
        self.episode_success = False
        
        # Control flags
        self.ready_to_step = False
        
        # Timer: step policy at fixed frequency
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.step_frequency), self.policy_step)
        
        rospy.loginfo('RL Agent Node initialized (Awaiting observations to map network dimensions...)')
        rospy.loginfo('  Policy type: %s', self.policy_type)
        rospy.loginfo('  Step frequency: %.1f Hz', self.step_frequency)
        rospy.loginfo('  Deterministic: %s', self.deterministic)

    def _init_model_lazily(self):
        """Dynamically builds a mock gym environment matching the live dimensions and loads weights."""
        rospy.loginfo('First complete observation received. Dynamically assembling policy network...')
        try:
            # 1. Build observation space dictionary matching received array dimensions
            obs_space_dict = {}
            for k, v in self.obs.items():
                obs_space_dict[k] = spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=np.float32)
            
            obs_space = spaces.Dict(obs_space_dict)
            
            # 2. Build continuous action space with dimension 3 (as required by your state_dict)
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            
            # 3. Create an inline mock environment object matching SB3 signature expectations
            class DummyEnv(gym.Env):
                def __init__(self):
                    self.observation_space = obs_space
                    self.action_space = action_space
                def reset(self): return obs_space.sample()
                def step(self, action): return obs_space.sample(), 0.0, False, {}

            dummy_env = DummyEnv()

            # 4. Define the policy architecture setup matching your train.py parameters
            policy_kwargs = dict(
                features_extractor_class=TreeClassFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                share_features_extractor=False,
                net_arch=dict(pi=[], vf=[]),
            )
            
            # 5. Instantiate the blank model structure
            self.model = self.policy_classes[self.policy_type](
                "MultiInputPolicy", 
                dummy_env, 
                policy_kwargs=policy_kwargs, 
                verbose=0
            )
            
            # 6. Unzip and inject the raw PyTorch weights
            with zipfile.ZipFile(self.policy_path, "r") as archive:
                policy_bytes = archive.read("policy.pth")
                buffer = io.BytesIO(policy_bytes)
                state_dict = th.load(buffer, map_location="cpu")
            
            self.model.policy.load_state_dict(state_dict)
            rospy.loginfo('Successfully bypassed NumPy 2.0 error. Policy layers mapped completely!')

        except Exception as e:
            rospy.logerr('Lazy initialization failed: %s', str(e))
            raise

    # Observation callbacks
    def on_obs_location(self, msg):
        self.obs['location'] = np.array(msg.data, dtype=np.float32)
        self.ready_to_step = all(v is not None for v in self.obs.values())
    
    def on_obs_belief(self, msg):
        self.obs['belief'] = np.array(msg.data, dtype=np.float32)
        self.ready_to_step = all(v is not None for v in self.obs.values())
    
    def on_obs_measurement(self, msg):
        self.obs['measurement'] = np.array(msg.data, dtype=np.float32)
        self.ready_to_step = all(v is not None for v in self.obs.values())
    
    def on_obs_tracked(self, msg):
        self.obs['tracked'] = np.array(msg.data, dtype=np.float32)
        self.ready_to_step = all(v is not None for v in self.obs.values())
    
    def on_obs_mask(self, msg):
        self.obs['mask'] = np.array(msg.data, dtype=np.float32)
        self.ready_to_step = all(v is not None for v in self.obs.values())
    
    # Feedback callbacks
    def on_reward(self, msg):
        self.episode_reward += msg.data
    
    def on_done(self, msg):
        if msg.data:
            self.total_episodes += 1
            rospy.loginfo('Episode %d done | Total Reward: %.3f | Steps: %d | Success: %s',
                         self.total_episodes, self.episode_reward, 
                         self.episode_steps, self.episode_success)
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.episode_success = False
    
    def on_info(self, msg):
        if len(msg.data) >= 3:
            self.episode_success = msg.data[2] > 0.5
    
    def policy_step(self, timer_event):
        """Step the policy at fixed frequency."""
        if not self.ready_to_step:
            return  # Wait until all observations are available
        
        # Construct observation dict as expected by policy
        obs_dict = {
            'location': self.obs['location'],
            'belief': self.obs['belief'],
            'measurement': self.obs['measurement'],
            'tracked': self.obs['tracked'],
            'mask': self.obs['mask'],
        }

        # Lazy compile network layers on step 1
        if self.model is None:
            self._init_model_lazily()
        
        try:
            # Get action from policy via the local model instance
            action, _states = self.model.predict(obs_dict, deterministic=self.deterministic)
            
            # Publish action
            msg = Float32MultiArray()
            msg.data = action.astype(np.float32)
            self.pub_action.publish(msg)
            
            self.episode_steps += 1
        
        except Exception as e:
            rospy.logerr('Error in policy inference: %s', str(e))
    
    def spin(self):
        """Keep node running."""
        rospy.spin()


if __name__ == '__main__':
    try:
        default_policy = os.path.join(package_root(), 'artifacts', 'gym', 'models', 'final_model.zip')
        policy_path = rospy.get_param('~policy_path', default_policy)
        if policy_path is None:
            rospy.logerr('No policy path provided. Use: _policy_path:=/path/to/policy.zip')
            exit(1)
        
        policy_type = rospy.get_param('~policy_type', 'PPO')
        
        agent = RLAgentNode(policy_path, policy_type)
        agent.spin()
    
    except rospy.ROSInterruptException:
        rospy.loginfo('RL Agent shutting down')
    except Exception as e:
        rospy.logerr('Fatal error: %s', str(e))
        exit(1)
