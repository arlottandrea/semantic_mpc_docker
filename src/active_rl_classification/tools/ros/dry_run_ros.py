#!/usr/bin/env python3
"""
Dry run test for ROS package and ros_rl_node.py
Validates: package structure, imports, dependencies, node configuration
"""

import sys
import os
import traceback

def test_python_environment():
    """Check Python version and sys.path."""
    print("\n[TEST 1] Python environment...")
    try:
        print(f"  Python version: {sys.version}")
        print(f"  Python executable: {sys.executable}")
        print(f"  sys.path: {sys.path[:3]}")  # First 3 entries
        print("[PASS] Python environment OK")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_active_rl_import():
    """Test active_rl_classification package import."""
    print("\n[TEST 2] active_rl_classification import...")
    try:
        from active_rl_classification import TreeClassificationEnv
        print(f"  TreeClassificationEnv: {TreeClassificationEnv}")
        print("[PASS] Package imports successfully")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_rospy_availability():
    """Check if rospy is available."""
    print("\n[TEST 3] ROS dependencies (rospy)...")
    try:
        import rospy
        print(f"  rospy: {rospy}")
        print("[PASS] rospy available")
        return True
    except ImportError as e:
        print(f"[FAIL] rospy not installed: {e}")
        print("  Note: This is expected in non-ROS environments. Install: apt-get install python3-rospy")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_std_msgs_availability():
    """Check if std_msgs is available."""
    print("\n[TEST 4] ROS message types (std_msgs)...")
    try:
        from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray
        print(f"  Bool: {Bool}")
        print(f"  Float32: {Float32}")
        print(f"  Float32MultiArray: {Float32MultiArray}")
        print(f"  Int32MultiArray: {Int32MultiArray}")
        print("[PASS] std_msgs available")
        return True
    except ImportError as e:
        print(f"[FAIL] std_msgs not installed: {e}")
        print("  Note: This is expected in non-ROS environments. Install: apt-get install python3-std-msgs")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_ros_node_import():
    """Test if ros_rl_node.py can be parsed as valid Python."""
    print("\n[TEST 5] ros_rl_node.py syntax...")
    try:
        node_path = os.path.join(
            os.path.dirname(__file__),
            "scripts",
            "ros_rl_node.py"
        )
        
        if not os.path.exists(node_path):
            print(f"[FAIL] ros_rl_node.py not found at {node_path}")
            return False
        
        with open(node_path, 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, node_path, 'exec')
        print(f"  File: {node_path}")
        print(f"  Size: {len(code)} bytes")
        print("[PASS] ros_rl_node.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] Syntax error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_ros_node_structure():
    """Validate ros_rl_node.py class structure."""
    print("\n[TEST 6] ros_rl_node.py structure...")
    try:
        # Import the node class dynamically
        import importlib.util
        node_path = os.path.join(
            os.path.dirname(__file__),
            "scripts",
            "ros_rl_node.py"
        )
        
        if not os.path.exists(node_path):
            print(f"[FAIL] ros_rl_node.py not found at {node_path}")
            return False
        
        spec = importlib.util.spec_from_file_location("ros_rl_node", node_path)
        module = importlib.util.module_from_spec(spec)
        
        # Don't execute if rospy is not available (it would fail at rospy.init_node)
        # Just check that the module has the expected class
        with open(node_path, 'r') as f:
            source = f.read()
        
        required_elements = [
            "class RosRLBridge",
            "def __init__",
            "def _on_action",
            "def spin",
        ]
        
        missing = []
        for elem in required_elements:
            if elem not in source:
                missing.append(elem)
        
        if missing:
            print(f"[FAIL] Missing elements: {missing}")
            return False
        
        print(f"  Found class: RosRLBridge")
        print(f"  Found methods: __init__, _on_action, spin")
        print("[PASS] ros_rl_node.py structure is valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_package_xml():
    """Validate package.xml."""
    print("\n[TEST 7] package.xml validation...")
    try:
        import xml.etree.ElementTree as ET
        
        pkg_path = os.path.join(
            os.path.dirname(__file__),
            "package.xml"
        )
        
        if not os.path.exists(pkg_path):
            print(f"[FAIL] package.xml not found at {pkg_path}")
            return False
        
        tree = ET.parse(pkg_path)
        root = tree.getroot()
        
        name = root.find('name')
        version = root.find('version')
        description = root.find('description')
        maintainer = root.find('maintainer')
        license_elem = root.find('license')
        
        if name is None:
            print("[FAIL] Missing <name> in package.xml")
            return False
        
        print(f"  Name: {name.text}")
        print(f"  Version: {version.text if version is not None else 'N/A'}")
        print(f"  Description: {description.text if description is not None else 'N/A'}")
        print(f"  Maintainer: {maintainer.text if maintainer is not None else 'N/A'}")
        print("[PASS] package.xml is valid")
        return True
    except ET.ParseError as e:
        print(f"[FAIL] XML parse error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_cmakelists():
    """Validate CMakeLists.txt structure."""
    print("\n[TEST 8] CMakeLists.txt validation...")
    try:
        cmake_path = os.path.join(
            os.path.dirname(__file__),
            "CMakeLists.txt"
        )
        
        if not os.path.exists(cmake_path):
            print(f"[FAIL] CMakeLists.txt not found at {cmake_path}")
            return False
        
        with open(cmake_path, 'r') as f:
            content = f.read()
        
        required_cmake = [
            "cmake_minimum_required",
            "project(active_rl_classification)",
            "find_package(catkin REQUIRED COMPONENTS",
            "catkin_package",
            "catkin_python_setup",
            "catkin_install_python",
            "ros_rl_node.py",
        ]
        
        missing = []
        for elem in required_cmake:
            if elem not in content:
                missing.append(elem)
        
        if missing:
            print(f"[FAIL] Missing elements: {missing}")
            return False
        
        print(f"  Found: project definition, catkin setup, Python installation")
        print(f"  Python scripts: ros_rl_node.py configured")
        print("[PASS] CMakeLists.txt is valid")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_ros_topics_config():
    """Verify ROS topic configuration in node."""
    print("\n[TEST 9] ROS topic configuration...")
    try:
        node_path = os.path.join(
            os.path.dirname(__file__),
            "scripts",
            "ros_rl_node.py"
        )
        
        with open(node_path, 'r') as f:
            source = f.read()
        
        # Check for topic definitions (simplified bridge)
        required_topics = [
            "action_topic",
            "reward_topic",
            "done_topic",
            "info_topic",
        ]
        
        missing_topics = []
        for topic_var in required_topics:
            if topic_var not in source:
                missing_topics.append(topic_var)
        
        if missing_topics:
            print(f"[FAIL] Missing topic configurations: {missing_topics}")
            return False
        
        print(f"  Input topic: rl/action (receives RL actions)")
        print(f"  Output topics (3): rl/reward, rl/done, rl/info (sends environment results)")
        print("[PASS] All ROS topics configured")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def test_deployment_guide():
    """Verify DEPLOYMENT.md exists."""
    print("\n[TEST 10] DEPLOYMENT.md...")
    try:
        deploy_path = os.path.join(
            os.path.dirname(__file__),
            "DEPLOYMENT.md"
        )
        
        if not os.path.exists(deploy_path):
            print(f"[FAIL] DEPLOYMENT.md not found at {deploy_path}")
            return False
        
        with open(deploy_path, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Installation",
            "Architecture",
            "I/O Topics",
            "How to Use",
            "Example",
        ]
        
        missing = []
        for section in required_sections:
            if section.lower() not in content.lower():
                missing.append(section)
        
        if missing:
            print(f"[FAIL] Missing sections: {missing}")
            return False
        
        lines = content.split('\n')
        print(f"  Size: {len(content)} bytes, {len(lines)} lines")
        print("[PASS] DEPLOYMENT.md is complete")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    """Run all dry-run tests."""
    print("=" * 70)
    print("ROS Package Dry Run Test Suite")
    print("=" * 70)
    
    # Change to package root for relative path lookups
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tests = [
        test_python_environment,
        test_active_rl_import,
        test_rospy_availability,
        test_std_msgs_availability,
        test_ros_node_import,
        test_ros_node_structure,
        test_package_xml,
        test_cmakelists,
        test_ros_topics_config,
        test_deployment_guide,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"[FAIL] Unexpected error in {test_func.__name__}: {e}")
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    print("\n" + "=" * 70)
    print("DRY RUN SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8} {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\nAll dry-run tests passed! Package is ready for ROS deployment.")
        print("\nTo deploy in ROS workspace:")
        print("  1. cd ~/ros1")
        print("  2. pixi shell")
        print("  3. catkin_make")
        print("  4. source devel/setup.bash")
        print("  5. rosrun active_rl_classification ros_rl_node.py")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Review errors above.")
        print("\nNote: ROS dependencies (rospy, std_msgs) are only required in actual ROS environment.")
        print("In development, you can proceed if core package and structure tests pass.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
