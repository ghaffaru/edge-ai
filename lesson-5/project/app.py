import argparse
import cv2
import sys
import numpy as np
import socket
import json
import paho.mqtt.client as mqtt
from random import randint
from inference import Network

