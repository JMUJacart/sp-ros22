"""
    controller_teleop.py - Used to send controller inputs to cart_teleop.ino for manual control of the cart

    Author:  Jacob Bringham
    Version: 4/29/22
"""

from inputs import get_gamepad
import math
import time
import threading
import multiprocessing
from playsound import playsound

import curses
import sys
from sys import platform
import serial #this package is pySerial not serial!!!
import bitstruct
import numpy as np

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        x1 = self.LeftJoystickX
        y1 = self.LeftJoystickY
        x2 = self.RightJoystickX
        y2 = self.RightJoystickY
        a = self.A
        b = self.B # b=1, x=2
        x = self.X
        y = self.Y # b=1, x=2
        rb = self.RightBumper
        lb = self.LeftBumper
        return x1, y1, x2, y2, a, b, x, y, rb, lb


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state



cart_port = '/dev/ttyUSB9'  #hardcoded depending on computer

#cart_port = '/dev/cu.usbserial-1463340' #Mac OS value

# assume the first port we find that has 'usbserial' in the name is the one from the mac
def detect_com_port():
    ports_list = comports()

    for port_candidate in ports_list:
        port_name = port_candidate.device

        if 'usbserial' in port_name:
            return port_name

    return -1


class teleop(object):

    # define max speed and min / max steering angle for cart
    MAX_SPEED = 255
    MIN_ANGLE = 0
    MAX_ANGLE = 100

    def __init__(self):

        # initialize current velocity and steering angle variables
        self.cur_vel = 0  # (0 - 255)
        self.cur_angle = 50  # 50 is middle, (0 - 100)
        self.sound = False
        self.p = multiprocessing.Process(target=playsound, args=("/home/jacart/sp-ros22/kart/coco.mp3",))
        self.tp = multiprocessing.Process(target=playsound, args=("/home/jacart/sp-ros22/kart/invincible.mp3",))
        self.turbo_count = 0
        self.turbo = False
        self.turbo_playing = False

        self.prev_key = 1
        self.controller = XboxController()

        # try to set up serial port for sending commands to arduino
        try:
            self.cart_ser = serial.Serial(cart_port, 57600, write_timeout=0)
        except Exception as e:
            print("ERROR. . .could not connect to arduino: " + str(e))
            exit(0)

        # start curses wrapper to get input
        curses.wrapper(self.get_input)

    """ main wrapper for curses """
    def get_input(self, stdscr):
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i, i, -1)

        stdscr.nodelay(True)
        stdscr.addstr(0, 0, 'Move with WASD, Z for brake, X for hard stop and Y for centering the wheel.')
        stdscr.addstr(1, 0, 'CTRL-C to exit')
        stdscr.addstr(7, 0, 'Throttle val:')
        stdscr.addstr(8, 0, 'Brake val:')
        stdscr.addstr(9, 0, 'Steering val:')

        # runs indefinitely, getting user input
        while True:

            keyval = stdscr.getch()
            # if keyval == ord('w'):
                # self.cur_vel = min(self.MAX_SPEED, self.cur_vel + 15)
            # elif keyval == ord('a'):
                # self.cur_angle = max(self.MIN_ANGLE, self.cur_angle - 5)
            # elif keyval == ord('s'):
                # self.cur_vel = max(0, self.cur_vel - 15)
            # elif keyval == ord('d'):
                # self.cur_angle = min(self.MAX_ANGLE, self.cur_angle + 5)
            # elif keyval == ord('y'):
                # self.cur_angle = 50
            # elif keyval == ord('x'):
                # self.cur_vel = 0
            # elif keyval == ord('z'):
                # self.brake(self.cur_vel / 255.0 * 3, stdscr)
                # self.cur_vel = 0

            x1, y1, x2, y2, a, b, x, y, rb, lb = self.controller.read()

            self.cur_vel   = min(max(0,              (-y2 / 2) * self.MAX_SPEED), self.MAX_SPEED * .8)
            self.cur_angle = min(max(self.MIN_ANGLE,  ( (x1 / 2) * self.MAX_ANGLE * 0.75) + (self.MAX_ANGLE / 2)), self.MAX_ANGLE)

            if a == 1 and not self.sound:
                self.p.start()
                self.sound = True

            if b == 1 and self.sound:
                self.p.terminate()
                self.p = multiprocessing.Process(target=playsound, args=("/home/jacart/sp-ros22/kart/coco.mp3",))
                self.sound = False
             
            # Turbo :)
            if lb == 1 or rb == 1:
                self.turbo_count += 1
            else:
                self.turbo_count = 0
                
            
            if self.turbo_count >= 10:
                self.turbo = True
                if not self.turbo_playing:
                    self.tp.start()
                self.turbo_playing = True
                self.send_cmd(160, 0, self.cur_angle, stdscr)

            else:
                if self.turbo_playing:
                    self.tp.terminate()
                    self.tp = multiprocessing.Process(target=playsound, args=("/home/jacart/sp-ros22/kart/invincible.mp3",))
                    self.turbo_playing = False
                self.turbo = False
                self.send_cmd(self.cur_vel, 0, self.cur_angle, stdscr)

            self.prev_key = keyval
            time.sleep(1 / 10.)

    """ constructs brake command """
    def brake(self, delay, stdscr):
        rate = 10.
        steps = int(delay * rate)
        for brake in np.linspace(0, 255, steps, endpoint=True):
            self.send_cmd(0, int(brake), self.cur_angle, stdscr)
            time.sleep(1. / rate)
            stdscr.getch()

    """ sends a set of throttle, brake, and steering commands to the arduino """
    def send_cmd(self, throttle, brake, steering, stdscr):
        data = bytearray(b'\x00' * 5)
        bitstruct.pack_into('u8u8u8u8u8', data, 0, 42, 21,
                             throttle, brake, steering)
        self.cart_ser.write(data) # TODO: Uncomment

        stdscr.addstr(7, 0, 'Throttle val: ' + str(throttle) + '            ')
        stdscr.addstr(8, 0, 'Brake val:    ' + str(brake)    + '            ')
        stdscr.addstr(9, 0, 'Steering val: ' + str(steering) + '            ')

if __name__ == "__main__":

    if platform == "linux" or platform == "linux2":
        cart_port = '/dev/ttyUSB9'
    elif platform == "darwin":
        cart_port=detect_com_port()
    #elif platform == "win32":

    print("Using serial port: "+str(cart_port))

    teleop()

