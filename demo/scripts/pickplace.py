#! /usr/bin/env python
# -*- coding: utf-8 -*-

from moveit.python_tools import roscpp_init
from moveit.task_constructor import core, stages
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, TwistStamped
import time

roscpp_init("pickplace")

# Specify robot parameters
arm = "panda_arm"
eef = "hand"

# Specify object parameters
object_name = "grasp_object"
object_radius = 0.02

# Start with a clear planning scene
psi = PlanningSceneInterface(synchronous=True)
psi.remove_world_object()

# Grasp object properties
objectPose = PoseStamped()
objectPose.header.frame_id = "world"
objectPose.pose.orientation.x = 1.0
objectPose.pose.position.x = 0.30702
objectPose.pose.position.y = 0.0
objectPose.pose.position.z = 0.285

# Add the grasp object to the planning scene
psi.add_box(object_name, objectPose, size=[0.1, 0.05, 0.03])

# Create a task
task = core.Task("PandaPickPipelineExample")
task.enableIntrospection()

# Start with the current state
task.add(stages.CurrentState("current"))

# Create a planner instance that is used to connect
# the current state to the grasp approach pose
pipeline = core.PipelinePlanner()
pipeline.planner = "RRTConnectkConfigDefault"
planners = [(arm, pipeline)]

# Connect the two stages
task.add(stages.Connect("connect1", planners))

# The grasp generator spawns a set of possible grasp poses around the object
grasp_generator = stages.GenerateGraspPose("Generate Grasp Pose")
grasp_generator.angle_delta = 0.2
grasp_generator.pregrasp = "open"
grasp_generator.grasp = "close"
grasp_generator.setMonitoredStage(task["current"])  # Generate solutions for all initial states

# SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
# Set frame for IK calculation in the center between the fingers
ik_frame = PoseStamped()
ik_frame.header.frame_id = "panda_hand"
ik_frame.pose.position.z = 0.1034
simpleGrasp.setIKFrame(ik_frame)

# Pick container comprises approaching, grasping (using SimpleGrasp stage), and lifting of object
pick = stages.Pick(simpleGrasp, "Pick")
pick.eef = eef
pick.object = object_name

# Twist to approach the object
approach = TwistStamped()
approach.header.frame_id = "world"
approach.twist.linear.z = -1.0
pick.setApproachMotion(approach, 0.03, 0.1)

# Twist to lift the object
lift = TwistStamped()
lift.header.frame_id = "panda_hand"
lift.twist.linear.z = -1.0
pick.setLiftMotion(lift, 0.03, 0.1)

# Add the pick stage to the task's stage hierarchy
task.add(pick)

# Connect the Pick stage with the following Place stage
task.add(stages.Connect("connect2", planners))

# Define the pose that the object should have after placing
placePose = objectPose
placePose.pose.position.y += 0.2  # shift object by 20cm along y axis

# Generate Cartesian place poses for the object
place_generator = stages.GeneratePlacePose("Generate Place Pose")
place_generator.setMonitoredStage(task["Pick"])
place_generator.object = object_name
place_generator.pose = placePose

# The SimpleUnGrasp container encapsulates releasing the object at the given Cartesian pose
simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")
simpleUnGrasp.pregrasp = "open"

# Place container comprises placing, ungrasping, and retracting
place = stages.Place(simpleUnGrasp, "Place")
place.eef = eef
place.object = object_name
place.eef_frame = "panda_link8"

# Twist to retract from the object
retract = TwistStamped()
retract.header.frame_id = "world"
retract.twist.linear.z = 1.0
place.setRetractMotion(retract, 0.03, 0.1)

# Twist to place the object
placeMotion = TwistStamped()
placeMotion.header.frame_id = "panda_hand"
placeMotion.twist.linear.z = 1.0
place.setPlaceMotion(placeMotion, 0.03, 0.1)

# Add the place pipeline to the task's hierarchy
task.add(place)

if task.plan():
    task.publish(task.solutions[0])

# avoid ClassLoader warning
del pipeline
del planners

# Prevent the program from exiting, giving you the opportunity to inspect solutions in rviz
time.sleep(3600)
