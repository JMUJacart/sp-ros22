"""
Shows all connected ZED cameras and outputs their serial numbers in the order the operator selects.

Authors: Jacob Bringham, Matthew Dim, Amber Oliver, Jacob McClaskey
Date: 2/18/2022
"""

import cv2
import sys
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer



class CameraSelector():

    def __init__(self, camera_info=[], text=""):
        """
        """
        self.orig_info = camera_info

        self.selected = []

        self.window = tk.Tk()
        self.window.winfo_toplevel().title("Camera Identifier")

        self.images = []
        self.buttons = []
        self.selected = []

        self.selected_txt = tk.StringVar()
        self.selected_txt.set("[]")

        self.instruction_lbl = tk.Label(self.window, text=text).grid(row=0, column=0, columnspan=len(camera_info))
        self.selected_lbl    = tk.Label(self.window, textvariable=self.selected_txt).grid(row=1, column=0, columnspan=len(camera_info))
        self.finish_btn = tk.Button(self.window, text="Finish", command=self.finish).grid(row=2, column=0, columnspan=len(camera_info))
        
        # Create image and label for each camera
        for i, info in enumerate(camera_info):
            serial = info[0]
            pic = info[1]
            
            # Need to resize image and convert from BGR for tkinter
            pic = cv2.resize(pic, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

            blue, green, red, alpha = cv2.split(pic)
            pic = cv2.merge( (red, green, blue) )

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(pic[:,:,0:3]))
            self.images.append(imgtk)

            button = tk.Button(self.window, image=imgtk, command=lambda i=i: self.buttonClick(i))
            button.grid(row=3, column=i)
            self.buttons.append(button)

    def buttonClick(self, i):
        if i in self.selected:
            self.selected.remove(i)
        else:
            self.selected.append(i)
        self.selected_txt.set(str(self.selected))

        
    def finish(self):
        for i in self.selected:
            print(f"{self.orig_info[i][0]}")
        exit()
        


def get_cameras():
    """
    Gets a picture and serial number from each camera for identification.

    @return Array of (serial number, pic) from each connected camera
    """
    print("get_cameras")
    devices = sl.Camera.get_device_list()
    print("get_cameras")

    sys.stderr.write(f"# Cameras: {len(devices)}\n")
    
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 15
    init.coordinate_units = sl.UNIT.METER # Set coordinate units
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    cameras = []

    for i, device in enumerate(devices):

        init.set_from_serial_number(devices[i].serial_number)

        cam = sl.Camera()
        
        sys.stderr.write(f"Opening camera {i}\n")
        status = cam.open(init)

        # Attempt to open camera
        if status == sl.ERROR_CODE.SUCCESS and cam.grab() == sl.ERROR_CODE.SUCCESS:
            
            image = sl.Mat()

            # Get ZED camera information
            camera_info = cam.get_camera_information()

            # 2D viewer utilities
            display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), \
                                               min(camera_info.camera_resolution.height, 720))
 

            cam.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            
            cam.close()


            pic = image.get_data()


            cv2.putText(pic, f"Camera {i}", (100, 100), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 5, 5)

            cameras.append( (camera_info.serial_number, pic) )

    return cameras


def main():
    """
    Starts the selection GUI.
    """

    cams = get_cameras()

    gui = CameraSelector(cams, "Select the front camera then the back camera")

    gui.window.mainloop()




if __name__ == "__main__":
    
    main()

    # for key, cam in cameras.items():
        # Show image to operator
        # TODO: Change to show all of them combined with labels
        # cv2.imshow("ZED | 2D View", cam[2])
        # cv2.waitKey()
        #
    

    
            


