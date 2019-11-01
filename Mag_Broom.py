import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


class Electron:
    
    def __init__(self, x, y, z, vx, vy, vz):
        # This function will be called whenever we create a new Electron Object
        # Inputs:
        # x, y, z - The positions of the Electrons, should be numpy arrays with
        #           astropy units attached (usually [mm])
        # vx, vy, vz - The velocites of the Electrons, should be numpy arrays
        #           with astropy units attached (usually [m/s])
        
        # Save the positions as parameters. If you have an Electron object e,
        # you can access its x position with the syntax: "e.x"
        self.x = x
        self.y = y
        self.z = z
        
        # Save the velocities of the Electrons
        self.vx = vx
        self.vy = vy
        self.vz = vz
        
        # Save the mass and charge of the Electrons as constants
        self.mass = 9.10938356e-31 * u.kg
        self.charge = 1.60217662e-19 * u.C
    
    def updateVel(self, ax, ay, az, dt):
        # This function updates the Electron's velocity given accelerations
        
        # Perform Euler forward steps
        self.vx += ax*dt
        self.vy += ay*dt
        self.vz += az*dt
    
    def updatePosn(self, dt):
        # This function updates the Electron's positions using its velocity
        
        # Perform Euler forward steps
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.z += self.vz*dt
    
    def findAccel(self, bx, by, bz):
        # Given a magnetic field vector, this function calculates the 
        # acceleration vector that an electron experiences based on the equation
        # a = (q/m)(vxB)
        
        # Compute the cross product vxB
        ax = self.vy*bz - by*self.vz
        ay = self.vz*bx - self.vx*bz 
        az = self.vx*by - bx*self.vy
        
        # Convert to acceleration
        ax *= self.charge / self.mass
        ay *= self.charge / self.mass
        az *= self.charge / self.mass
        
        # Convert to correct units
        ax = ax.to('m/s^2')
        ay = ay.to('m/s^2')
        az = az.to('m/s^2')
        
        # Return acceleration vector
        return ax,ay,az


def genElec(num, angles):
    # This function is used to generate an Electron Object
    # Inputs:
    #   - num: The number of electrons to be generated
    #   - angles: The angles at which the electrons will enter the field, angles
    #     are measured from the z-axis, towards the y-axis
    
    # x and y positions are randomly placed in a 10x10 mm box
    x = np.random.uniform(-5,5,num) * u.mm
    y = np.random.uniform(-5,5,num) * u.mm
    # z is set to 9.5mm, so it is just about to enter the magnetic field
    z = 9.5*np.ones(num) * u.mm
    
    # Give it an initial velocity using the incident angles
    vx = np.zeros(num) * u.m/u.s
    vy = 88000*np.sin(angles) * u.m/u.s
    vz = 88000*np.cos(angles) * u.m/u.s
    
    # Return the electron object
    return Electron(x,y,z,vx,vy,vz)


def genAngles(num, max):
    # This function generates a list of angles at which electrons will enter
    # the magnetic field
    # Inputs:
    #   - num: The number of angles to be generated
    #   - max: The maximum angle allowed in Radians
    # Note: Angles are generated uniformly from -max to max
    return np.random.uniform(-max,max,num) << u.rad


def getB(e):
    # This function takes in an Electron object and returns the magnetic field
    # at the position of each electron.
    
    # Initialize the magnetic field vectors, start out with a zero vector
    num = len(e.x)
    bx = np.zeros(num)
    by = np.zeros(num)
    bz = np.zeros(num)

    # Determine which electrons are in the field from 10 to 15 mm
    infield = np.logical_and(e.z > 10*u.mm, e.z < 15 * u.mm)
    # Give the electrons in the field a magnetic field
    bx[infield] = 0.5
    
    # Set all units to [gauss]
    bx = bx << u.gauss
    by = by << u.gauss
    bz = bz << u.gauss
    
    # Return the magnetic field
    return bx, by, bz

# Get the number of electrons to simulate
numElec = 100

# Initialize a maximum incident angle
maxang = (10 * u.deg).to('rad').value

# Generate incident angles
angles = genAngles(numElec,maxang)

# Generate those electrons
e = genElec(numElec,angles)

# Initialize our time step
dt = 0.1 * u.ns

# Initialize arrays which will hold the y and z positions of the electron, so
# we can plot its path in the future
ys = []
zs = []

# Iterate through 1000 time steps
from tqdm import tqdm
for i in tqdm(range(1000)):
    
    # Find the magnetic field and acceleration at the electrons current position
    bx, by, bz = getB(e)
    ax, ay, az = e.findAccel(bx,by,bz)
    
    # Save the electron's position and velocity for later
    v0 = (e.vx.copy(),e.vy.copy(),e.vz.copy())
    p0 = (e.x.copy(),e.y.copy(),e.z.copy())

    # Move the electrons half a step
    e.updateVel(ax, ay, az, dt/2)
    e.updatePosn(dt/2)
    
    # Find the magnetic field and acceleration at this half step position
    bx, by, bz = getB(e)
    ax, ay, az = e.findAccel(bx,by,bz)
    
    # Move back to the original position
    e.vx = v0[0]
    e.vy = v0[1]
    e.vz = v0[2]
    e.x = p0[0]
    e.y = p0[1]
    e.z = p0[2]
    
    # Update velocity and position using the half step values
    e.updateVel(ax, ay, az, dt)
    e.updatePosn(dt)
    
    
    # Record the y and z positions for later plotting
    ys.append(e.y.value[0])
    zs.append(e.z.value[0])


# Find the expected exit angles:
predicted_theta =  np.arcsin(np.sin(angles) - (e.charge * (0.5 * u.gauss) * (5 * u.mm) / e.mass / (88000 * u.m/u.s)))

# Find the actual exit angles for these electrons:
actual_theta = np.arctan(e.vy/e.vz)

# Find the error in angle
theta_err = np.abs(predicted_theta) - np.abs(actual_theta)

# Find how much the electron has accelerated. In a perfect simulation, this
# velocity error should be zero and the speed of the electrons should not change
vel_err = (88000 * u.m/u.s) - np.sqrt(e.vx**2 + e.vy**2 + e.vz**2)


# Plot the y and z positions of the electron
# Red lines give the edges of the field
plt.figure()
plt.plot(zs,ys)
plt.xlabel('Z-Position')
plt.ylabel('Y-Position')
plt.axis('equal')
plt.axvline(x=10,c='r')
plt.axvline(x=15,c='r')
plt.show()


























