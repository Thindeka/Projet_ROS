import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    projet2025_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('projet2025'),
                'launch',
                'projet.launch.py'
            )
        )
    )

    manager = Node(
        package='projet_ros2',
        executable='manager_node',
        name='challenge_manager',
        output='screen',
    )


    return LaunchDescription([
        projet2025_launch,
        manager,
    ])