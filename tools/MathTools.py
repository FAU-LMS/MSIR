import math

def get_baseline(x0, y0, x1, y1):
    baseline = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return baseline

def get_angle(x0, y0, x1, y1):
    angle = math.atan2(y1 - y0, x1 - x0)
    return angle
