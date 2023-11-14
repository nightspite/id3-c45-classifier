""" Helper functions. """
import math

def angle(vector1, vector2):
    """Calculates the angle between two vectors."""
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return 180 - (math.degrees(math.acos(inner_product/(len1*len2))))

def build_vector(xa, ya, xb, yb):
    """Builds a vector from two points."""
    ux = xb-xa
    uy = yb-ya
    return (ux, uy)

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, (int, float))
