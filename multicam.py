"""
Simple python script to run two ZED cameras at the same time with the same settings.

Author: Jacob Bringham, Matthew Dim, Amber Oliver, Jacob McClaskey
"""

import cv2
import sys
import pyzed.sl as sl

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import sys

import numpy as np

# TODO: Put source website
res_settings = {
    2000: sl.RESOLUTION.HD2K,
    1080: sl.RESOLUTION.HD1080,
    720: sl.RESOLUTION.HD720,
    672: sl.RESOLUTION.VGA
}

fps_settings = {
        sl.RESOLUTION.HD2K: [15],
        sl.RESOLUTION.HD1080: [15, 30],
        sl.RESOLUTION.HD720: [15, 30, 60],
        sl.RESOLUTION.VGA: [15, 30, 60, 100]
}

def print_camera_information(cam):
    # print( dir(cam) )
    print( cam.get_camera_information().serial_number )

    # print("Resolution: {0} {1}".format( round(cam.get_resolution().width, 2), cam.get_resolution().height) )
    # print("FPS: {0}".format( cam.get_camera_fps() ))
    # print("Serial Number: {0}".format(cam.get_camera_information().serial_number) )


def usage():
    print(f"{sys.argv[0]} <resolution> <fps>\n")
    print(f"Resolutions: ")
    for key in res_settings.keys():
        print(f"    {key} FPS: {fps_settings[res_settings[key]]}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage()
        exit(-1)

    res = int(sys.argv[1])
    fps = int(sys.argv[2])

    if res not in res_settings:
        usage()
        exit(-1)
        
        if fps not in fps_settings[res]:
            usage()
            exit(-1)

    print("about to find devices", flush=True)
    try:
        devices = sl.Camera.get_device_list()
    except:
        print("Error getting devices", flush=True)
        exit(-1)
    
    print("devices found", flush=True)
    device_number = len(devices)

    print(f"I see {device_number} cameras!", flush=True)

    if device_number == 0:
        print("No Cameras connected!")
        exit (-1)

    if device_number >= 1:
        # Set camera settings (Cam 1)
        init = sl.InitParameters()
        init.camera_resolution = res_settings[res]
        init.camera_fps = fps
        init.set_from_serial_number(devices[0].serial_number)

        init.coordinate_units = sl.UNIT.METER # Set coordinate units
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        cam1 = sl.Camera()
        status = cam1.open(init)

        # ZED Pose Tracking Cam 1
        print_camera_information(cam1)
        positional_tracking_parameters1 = sl.PositionalTrackingParameters()
        cam1.enable_positional_tracking(positional_tracking_parameters1)
    if device_number == 2:
        # Set camera settings (Cam 2)
        init = sl.InitParameters()
        init.camera_resolution = res_settings[res]
        init.camera_fps = fps
        init.set_from_serial_number(devices[1].serial_number)

        init.coordinate_units = sl.UNIT.METER # Set coordinate units
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        cam2 = sl.Camera()
        status = cam2.open(init)
        
        print_camera_information(cam2)

        # ZED Pose Tracking Camera 2
        positional_tracking_parameters2 = sl.PositionalTrackingParameters()
        cam2.enable_positional_tracking(positional_tracking_parameters2)


    if device_number > 2:
        print("Additional devices detected, Please be aware of additional implementaiton required.")



    # Show video feed
    # Get ZED camera information
    camera_info = cam1.get_camera_information()
    
    print("------------------------")
    print(str(camera_info.camera_configuration.camera_resolution.width))
    print(str(camera_info.camera_configuration.camera_resolution.height))
    print("------------------------")

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]


    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST 
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use
    if device_number >= 1:
        cam1.enable_object_detection(obj_param)
        obj_runtime_param1 = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param1.detection_confidence_threshold = 40
    if device_number == 2:    
        cam2.enable_object_detection(obj_param)
        obj_runtime_param2 = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param2.detection_confidence_threshold = 40

    #bodies = sl.Objects() # Objects 




    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)
    # viewer.init(camera_info.calibration_parameters.left_cam, False, None)

    # Create ZED objects filled in the main loop
    bodies1 = sl.Objects()
    bodies2 = sl.Objects()
    image1 = sl.Mat()
    image2 = sl.Mat()
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)


    while True:
        # Grab an image
        if device_number >= 1 and cam1.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            cam1.retrieve_image(image1, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

            # viewer.set_image(image_left_ocv)

            # Pose Tracking - Object Detection
            cam1.retrieve_objects(bodies1, obj_runtime_param1)
            print(len(bodies2.object_list), " objects detected Camera 1.\n")
            
            for body in bodies1.object_list:
                print(body.position)
            
            # viewer.update_view(image1, bodies1)

            # Update OCV view
            image_left_ocv = image1.get_data()


            # 2D Rendering
            cv_viewer.render_2D(image_left_ocv,image_scale,bodies1.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)

            print("==============================================================")
        if device_number == 2 and cam2.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            cam2.retrieve_image(image2, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

            cam2.retrieve_objects(bodies2, obj_runtime_param2)
            print(len(bodies2.object_list), " objects detected Camera 2.\n")

            for body in bodies2.object_list:
                print(body.position)

            # Update OCV view
            image_left_ocv = image2.get_data()
            # viewer.set_image(image_left_ocv)

            cv_viewer.render_2D(image_left_ocv,image_scale,bodies2.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View | Second Camera", image_left_ocv)
            cv2.waitKey(10)



    # viewer.exit()
    # Disable modules and close camera
    cam1.disable_object_detection()
    cam1.disable_positional_tracking()
    cam1.close()
    if device_number == 2:
        cam2.disable_object_detection()
        cam2.disable_positional_tracking()
        cam2.close()



