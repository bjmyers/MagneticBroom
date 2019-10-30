import numpy as np
import astropy.units as u


class Electron:
    
    def __init__(self, x, y, z, vx, vy, vz):
        
        self.x = x
        self.y = y
        self.z = z
        
        self.vx = vx
        self.vy = vy
        self.vz = vz
        
        self.mass = 9.10938356e-31 * u.kg
        self.charge = 1.60217662e-19 * u.C
    
    def updateVel(self, ax, ay, az, dt):
        
        self.vx += ax*dt
        self.vy += ay*dt
        self.vz += az*dt
    
    def updatePosn(self, dt):
        
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.z += self.vz*dt
    
    def findAccel(self, bx, by, bz):
        
        ax = self.vy*bz - by*self.vz
        ay = self.vx*bz - self.vz*bx
        az = self.vx*by - bx*self.vy
        
        ax *= self.charge / self.mass
        ay *= self.charge / self.mass
        az *= self.charge / self.mass
        
        ax = ax.to('m/s^2')
        ay = ay.to('m/s^2')
        az = az.to('m/s^2')
        
        return ax,ay,az


def genElec(num):
    
    x = np.random.uniform(-5,5,num) * u.mm
    y = np.random.uniform(-5,5,num) * u.mm
    z = 9*np.ones(num) * u.mm
    
    vx = np.zeros(num) * u.m/u.s
    vy = np.zeros(num) * u.m/u.s
    vz = 88000*np.ones(num) * u.m/u.s
    
    return Electron(x,y,z,vx,vy,vz)

def getB(e):
    
    num = len(e.x)
    bx = np.zeros(num)
    by = np.zeros(num)
    bz = np.zeros(num)

    infield = np.logical_and(e.z > 10*u.mm, e.z < 15 * u.mm)
    # bx[infield] = 0.5
    bx[infield] = 10000
    by[infield] = 0
    bz[infield] = 0
    
    bx = bx << u.gauss
    by = by << u.gauss
    bz = bz << u.gauss
    
    return bx, by, bz

numElec = 1

e = genElec(numElec)
dt = 0.0001 * u.ns

e.z = np.ones(numElec)*12*u.mm

bx, by, bz = getB(e)

ax, ay, az = e.findAccel(bx,by,bz)

e.updateVel(ax, ay, az, dt)

from tqdm import tqdm
for i in range(1000):
    
    bx, by, bz = getB(e)
    
    ax, ay, az = e.findAccel(bx,by,bz)
    
    x = np.sqrt(e.vx**2 + e.vy**2 + e.vz**2)
    
    v0 = (e.vx,e.vy,e.vz)
    p0 = (e.x,e.y,e.z)

    e.updateVel(ax, ay, az, dt/2)
    e.updatePosn(dt/2)
    
    bx, by, bz = getB(e)
    
    ax, ay, az = e.findAccel(bx,by,bz)
    
    e.vx = v0[0]
    e.xy = v0[1]
    e.vz = v0[2]
    e.x = p0[0]
    e.y = p0[1]
    e.z = p0[2]
    
    e.updateVel(ax, ay, az, dt)
    e.updatePosn(dt)
    
    print(np.sqrt(e.vx**2 + e.vy**2 + e.vz**2) - x)

# Expected angle: 0.5232 rad deg
theta = np.arcsin((e.charge * (0.5*u.gauss) * 5*u.mm / e.mass / (88000*u.m/u.s)).to(''))

# Actual Angle: 24.0848 deg
actual_theta = np.arctan(e.vy/e.vz)

# print(np.sqrt(e.vx**2 + e.vy**2 + e.vz**2))
































