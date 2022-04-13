#!/bin/bash
cd ./slam_data/trajectory
evo_ape kitti ground_truth.txt laser_odom.txt -va --plot 
