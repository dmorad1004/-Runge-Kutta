import numpy as np
from typing import Callable


class RugenKutta:
    """
    Clase para implementar el método de Runge-Kutta.

    Args:
        func (Callable): Función que representa la ecuación diferencial a resolver.
        interval (list): Lista que define el rango [xi, xf] en el cual se resolverá la ecuación diferencial.
        Y0 (float): Valor inicial de la variable dependiente y en el punto inicial del intervalo.
        step (float): Tamaño del paso para la discretización del intervalo.
    """

    def __init__(self, func: Callable, interval: list, Y0: float, step: float,
                 ):
        self.func = func
        self.interval = interval
        self.Y0 = Y0
        self.step = step

    def __euler(self, x, y):
        y[0] = self.Y0

        for i in range(0, len(x)-1):
            y[i+1] = y[i]+self.func(x[i], y[i])*self.step
        return [x, y]

    def __RK2(self, x, y):
        y[0] = self.Y0
        for i in range(0, len(x)-1):
            xi = x[i]
            yi = y[i]

            k1 = self.func(xi, yi)
            k2 = self.func(xi+self.step*1/2, yi+self.step*k1*1/2)

            y[i+1] = yi + k2*self.step

        return [x, y]

    def __RK4(self, x, y):
        y[0] = self.Y0
        for i in range(0, len(x)-1):
            xi = x[i]
            yi = y[i]

            k1 = self.func(xi, yi)
            k2 = self.func(xi+self.step*1/2, yi+self.step*k1*1/2)
            k3 = self.func(xi+self.step*1/2, yi+self.step*k2*1/2)
            k4 = self.func(xi+self.step, yi+self.step*k3)

            y[i+1] = yi + 1/6*(k1+2*k2+2*k3+k4)*self.step

        return [x, y]

    def solve_ode(self, method="RK2"):
        x = np.arange(
            self.interval[0], self.interval[1]+self.step, self.step)
        y = np.zeros_like(x)
        if method == "euler":
            return self.__euler(x, y)
        if method == "RK2":
            return self.__RK2(x, y)
        if method == "RK4":
            return self.__RK4(x, y)
