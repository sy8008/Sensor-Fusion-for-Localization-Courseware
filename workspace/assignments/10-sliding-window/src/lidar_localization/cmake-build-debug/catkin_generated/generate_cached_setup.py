# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/melodic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/melodic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/sy/ShenLan_MultiSensorFusionCourse/Sensor-Fusion-for-Localization-Courseware/workspace/assignments/10-sliding-window/devel;/home/sy/ros_work_space/catkin_overlay_ws/devel;/opt/ros/melodic;/home/sy/ros_work_space/idc_ws/idc_ws_dep/devel;/home/sy/ros_work_space/idc_ws/idc_gazebo_ws/devel'.split(';'):
        python_path = os.path.join(workspace, 'lib/python2.7/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/sy/ShenLan_MultiSensorFusionCourse/Sensor-Fusion-for-Localization-Courseware/workspace/assignments/10-sliding-window/devel/env.sh')

output_filename = '/home/sy/ShenLan_MultiSensorFusionCourse/Sensor-Fusion-for-Localization-Courseware/workspace/assignments/10-sliding-window/src/lidar_localization/cmake-build-debug/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
