import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Función de costo (parábola)
def cost_function(x):
    return x ** 2

# Algoritmo de gradiente descendente
def gradient_descent(x, learning_rate, epochs):
    parameter_history = [x]
    cost_history = [cost_function(x)]

    for epoch in range(epochs):
        gradient = 2 * x
        x -= learning_rate * gradient
        parameter_history.append(x)
        cost_history.append(cost_function(x))

    return parameter_history, cost_history

# Parámetros iniciales
initial_x = 4
learning_rate = 0.01
epochs = 200

# Ejecución del gradiente descendente
parameter_history, cost_history = gradient_descent(initial_x, learning_rate, epochs)

# Crear la figura y los ejes
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 30)
line, = ax.plot([], [], 'r', lw=2)
point, = ax.plot([], [], 'bo')

# Función de inicialización de la animación
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# Función de actualización de la animación
def update(frame):
    x_vals = np.linspace(-5, 5, 100)
    cost_vals = cost_function(x_vals)

    line.set_data(x_vals, cost_vals)
    point.set_data(parameter_history[frame], cost_history[frame])

    return line, point

# Crear la animación
animation = FuncAnimation(fig, update, frames=len(parameter_history),
                          init_func=init, blit=True)

# Mostrar la animación
plt.xlabel('Parámetro')
plt.ylabel('Costo')
plt.title('Evolución del Gradiente Descendente')
plt.grid(True)
plt.show()
