from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            namespace= "ascamera",
            package='ascamera',
            executable='ascamera_node',
            respawn=True,
            output='both',
            parameters=[
                {"confiPath": "/home/ubuntu22/Desktop/linux_ros/ros2/ascamera/configurationfiles"},
                {"rgb_gain": 1.1}
            ]
        ),
    ])
