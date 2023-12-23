import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
m = 1.0e-26         # Mass of a molecule (kg)
epsilon = 1.0e-21   # Lennard-Jones potential parameter (J)
sigma = 1.0e-10     # Lennard-Jones potential parameter (m)

# Simulation parameters
num_molecules = int(input("Enter the number of molecules: "))
cube_size = 10.0e-9  # Size of the simulation cube (m)
time_step = 1.0e-14  # Increased time step for slower motion
num_steps = 500

# Initialize molecule positions and velocities randomly
positions = np.random.rand(num_molecules, 3) * cube_size
velocities = np.random.randn(num_molecules, 3) * 50.0  # Initial velocities

# Metadynamics parameters
metadynamics_interval = 100
bias_factor = 1.0e-20
bias_potential = np.zeros((num_steps, num_molecules))

# Initialize array to store positions at each time step
trajectory = np.zeros((num_steps, num_molecules, 3))

# Variable to control animation pause
pause = False
button_label = "Pause"

# Variable to control animation speed
animation_speed = 1.0

def toggle_pause(event):
    global pause, button_label
    pause = not pause
    if pause:
        animation.event_source.stop()
        button_label = "Resume"
    else:
        animation.event_source.start()
        button_label = "Pause"
    button_pause.label.set_text(button_label)

# Update the animation speed
def update_speed(val):
    global animation_speed
    animation_speed = speed_slider.val

# Simulation loop
for step in range(num_steps):
    # Update positions using Verlet integration
    positions += velocities * (time_step / animation_speed) + 0.5 * (time_step**2) * bias_potential[step][:, np.newaxis] / m

    # Calculate forces using Lennard-Jones potential
    distances = np.linalg.norm(positions[:, np.newaxis, :] - positions, axis=2)
    distances[distances == 0] = 1.0  # Avoid division by zero    
    positions = np.mod(positions, cube_size)

    forces = 48 * epsilon * ((sigma / distances)**13 - 0.5 * (sigma / distances)**7)[:, :, np.newaxis] \
             * (positions[:, np.newaxis, :] - positions) / distances[:, :, np.newaxis]**2

    # Exclude self-forces
    np.fill_diagonal(forces[:,:,0], 0)
    np.fill_diagonal(forces[:,:,1], 0)
    np.fill_diagonal(forces[:,:,2], 0)

    # Calculate total forces and update velocities
    total_forces = np.sum(forces, axis = 1)
    velocities += total_forces * (time_step / animation_speed) / m

    # Apply metadynamics bias potential
    if step % metadynamics_interval == 0:
        bias_potential[step + 1] = bias_potential[step] + bias_factor * np.sum(distances**(-12) - 0.5 * distances**(-6), axis=1)

    # Save positions at each time step
    trajectory[step] = positions.copy()

# Update the plot at each animation frame
def update(frame):
    ax.clear()
    ax.scatter(trajectory[frame, :, 0], trajectory[frame, :, 1], trajectory[frame, :, 2], marker='o', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Molecular Dynamics Simulation - Frame {frame}')
    ax.set_xlim([0, cube_size])
    ax.set_ylim([0, cube_size])
    ax.set_zlim([0, cube_size])
    ax.set_box_aspect([1, 1, 1])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
animation = FuncAnimation(fig, update, frames=num_steps, interval=100/animation_speed, blit=False)

# Pause button
axpause = plt.axes([0.81, 0.05, 0.1, 0.075])
button_pause = Button(axpause, button_label)
button_pause.on_clicked(toggle_pause)

# Slider for controlling animation speed
axspeed = plt.axes([0.1, 0.01, 0.65, 0.03])
speed_slider = Slider(axspeed, 'Speed', 0.1, 2.0, valinit=animation_speed)
speed_slider.on_changed(update_speed)

# Display the plot
plt.show()
