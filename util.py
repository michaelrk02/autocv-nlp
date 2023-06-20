import random

def readfile(path):
    try:
        f = open(path, "r")
        return f.read()
    except:
        return None

def nrandom(precision = 6):
    min_value = 0
    max_value = 10 ** precision
    return random.randint(min_value, max_value) / max_value
