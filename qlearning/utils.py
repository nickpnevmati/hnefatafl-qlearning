from operator import add

def tuple_add(a: tuple, b: tuple) -> tuple:
    return tuple(map(add, a, b))

def tuple_abs(a: tuple) -> tuple:
    return tuple(map(abs, a))