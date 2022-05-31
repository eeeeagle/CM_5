import sympy as sm
import numpy as np


def interval_halving(f, a, b, eps):
    c = 0

    iters = 0
    while abs(b - a) >= eps:
        c = a + ((b - a) / 2)

        if f.subs('x', a) * f.subs('x', c) > 0:
            a = c
        else:
            b = c

        iters += 1

    return iters, c


def secant(f, a, b, eps):
    c = 0

    iters = 0
    while abs(c - b) >= eps:
        tmp = c
        c = b - f.subs('x', b) * (a - b) / (f.subs('x', a) - f.subs('x', b))
        a = b
        b = tmp
        iters += 1

    return iters, c


def newton(f, d, a, eps):
    x_prev = a
    x_cur = x_prev - f.subs('x', x_prev) / d.subs('x', x_prev)

    iters = 1
    while abs(x_cur - x_prev) >= eps:
        x_prev = x_cur
        x_cur = x_prev - f.subs('x', x_prev) / d.subs('x', x_prev)

        iters += 1

    return iters, x_cur


def task1():
    print("===================================\nTASK 1\n")
    x = sm.Symbol('x')

    f = x ** 3 - 2.92 * x ** 2 + 1.435 * x + 0.791
    a = -1
    b = 3
    eps = 0.0001

    print("      F(x) =", f)
    print("         a =", a)
    print("         b =", b)
    print("       eps =", eps)

    print("\n[1] INTERVAL HALVING METHOD")
    iters, result = interval_halving(f, a, b, eps)
    print("Iterations =", iters)
    print("    Result =", result)

    print("\n[2] SECANT METHOD")
    iters, result = secant(f, a, b, eps)
    print("Iterations =", iters)
    print("    Result =", result)

    print("\n[3] NEWTON'S METHOD")
    iters, result = newton(f, sm.diff(f), a, eps)
    print("Iterations =", iters)
    print("    Result =", result, "\n")


def newton_system(f1, f2, eps, start):
    x, y = sm.symbols('x y')

    j = np.array([[sm.diff(f1, x), sm.diff(f1, y)],
                  [sm.diff(f2, x), sm.diff(f2, y)]])

    iters = 0
    while iters < 1000:
        x0, y0 = start

        j_ = np.array([[j[0, 0].subs({x: x0, y: y0}), j[0, 1].subs({x: x0, y: y0})],
                       [j[1, 0].subs({x: x0, y: y0}), j[1, 1].subs({x: x0, y: y0})]], float)

        f = np.array([[f1.subs({x: x0, y: y0})],
                      [f2.subs({x: x0, y: y0})]], float)

        temp = np.linalg.solve(j_, f)
        x1 = x0 - temp[0, 0]
        y1 = y0 - temp[1, 0]

        start = (x1, y1)
        iters += 1

        if abs(x1 - x0) < eps:
            return iters, (x1, y1)

    return iters, "Unable to count"


def task2():
    print("===================================\nTASK 2\n")

    x, y = sm.symbols('x y')

    f1 = sm.cos(x + 0.5) - y - 2
    f2 = sm.sin(y) - 2 * x - 1
    eps = 0.0001
    start = (-5, 3)

    print("    F_1(x) =", f1)
    print("    F_2(x) =", f2)
    print("       eps =", eps)
    print("     start =", start, "\n")

    iters, result = newton_system(f1, f2, eps, start)
    print("Iterations =", iters)
    print("    Result =", result, "\n")


task1()
task2()
