import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Argument configurable en ligne de commande
    roundabout_arg = DeclareLaunchArgument(
        'roundabout_direction',
        default_value='left',
        description='Direction du rond-point : left ou right'
    )

    # Launch du prof
    projet2025_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('projet2025'),
                'launch',
                'projet.launch.py'
            )
        )
    )

    follow = Node(
        package='projet_ros2',
        executable='follow_line',
        name='line_follower',
        output='screen',
        parameters=[{
            'roundabout_direction': LaunchConfiguration('roundabout_direction')
        }]
    )

    return LaunchDescription([
        roundabout_arg,
        projet2025_launch,
        follow,
    ])