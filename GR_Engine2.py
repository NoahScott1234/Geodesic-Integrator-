
import sympy as sp
import numpy as np
t = sp.symbols("t")
tau = sp.symbols('tau')
r= sp.symbols("r", positive = True)
theta = sp.symbols("theta") 
phi = sp.symbols("phi")
x = sp.symbols("x")
y = sp.symbols("y")
z = sp.symbols("z")
u = sp.symbols("u")
v = sp.symbols("v")
U = sp.symbols("U")
V = sp.symbols("V") 

coordinates_dict = {
    'Manual' : '[ , , , ]',
    'Cartesian' : [t, x, y, z],
    'Spherical' : [t, r, theta, phi],
    'Cylindrical' : [t, r, phi, z],
    'Null (Cartesian)' : [u, v, y, z],
    'Null (Kruskal)' : [U, V, theta, phi],
}

functions = {
t : sp.Function('t')(tau),
x : sp.Function('x')(tau),
y : sp.Function('y')(tau),
z : sp.Function('z')(tau),
r : sp.Function('r')(tau),
theta : sp.Function('theta')(tau),
phi : sp.Function('phi')(tau),
u : sp.Function('u')(tau),
v : sp.Function('v')(tau),
U : sp.Function('U')(tau),
V : sp.Function('V')(tau)
}

c = sp.symbols('c', positive = True)
G = sp.symbols("G", positive = True)
M = sp.symbols("M", positive = True)
J = sp.symbols("J")
A = sp.symbols("A")
a = sp.symbols('a', positive = True)

mu = sp.symbols("mu")
nu = sp.symbols("nu")
sigma = sp.symbols("sigma")
rho = sp.symbols("rho")
epsilon = sp.symbols("epsilon")
alpha = sp.symbols("alpha")

def calculate_christoffel(metric, coordinates):
    gamma = {}
    metric_inv = sp.simplify(metric.inv())
    for sigma in range(4):
        for mu in range(4):
            for nu in range (4):
                gamma_x_two = 0
                for rho in range (4):
                    gamma_x_two += metric_inv[sigma, rho] * (
                    sp.diff(metric[rho, nu], coordinates[mu]) 
                    + sp.diff(metric[rho, mu], coordinates[nu]) 
                    - sp.diff(metric[mu, nu], coordinates[rho]))
                gamma[sigma, mu, nu] = (gamma_x_two/2)
                gamma[sigma, nu, mu] = (gamma_x_two/2)
    return gamma
    
"""Find Reimann Tesnsor Using Connection Coeffecients, calculated above"""
def calculate_riemann_tensor(gamma, coordinates):
    R = {}
    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                for rho in range(4):
                
                    Reimann = (sp.diff(gamma[rho,mu,sigma],coordinates[nu]) 
                    - sp.diff(gamma[rho,mu,nu],coordinates[sigma]))
                
                    for alpha in range(4):
                        Reimann += (gamma[alpha,mu,sigma] * gamma[rho,nu,alpha]
                        - gamma[alpha,mu,nu] * gamma[rho,sigma,alpha])
                    R[rho,mu,nu,sigma] = sp.trigsimp(sp.simplify(Reimann))
    return R



"""Find Ricci Tensor Using Connection Coeffecients, Same as Reimann tesnor except first and third indicies are contracted"""
def calculate_ricci_tensor(gamma, coordinates):
    ricci = {}
    for mu in range(4):
        for nu in range(4):
            Ric = 0
            for sigma in range(4):
                Ric += (sp.diff(gamma[sigma,mu,nu],coordinates[sigma]) 
                - sp.diff(gamma[sigma,mu,sigma],coordinates[nu]))
                for rho in range(4):  
                    Ric += ((gamma[rho,mu,nu] * gamma[sigma,sigma,rho])
                    -(gamma[rho,mu,sigma] * gamma[sigma,nu,rho]))
            ricci[mu,nu] = sp.trigsimp(sp.simplify(Ric))
    return ricci



"""Calculate Ricci Scalar"""
def calculate_ricci_scalar(metric, ricci):
    metric_inv = sp.simplify(metric.inv())
    ricci_scalar = 0
    for mu in range (4):
        for nu in range(4):
            ricci_scalar += metric_inv[mu,nu] * ricci[mu,nu]
    return sp.trigsimp(sp.simplify(ricci_scalar))



"""Calulate Einstein Tensor"""
def calculate_einstein_tensor(ricci, ricci_scalar, metric):
    E={}
    for mu in range(4):
        for nu in range(4):
            E[mu,nu] = ricci[mu,nu] - sp.Rational(1,2) * ricci_scalar * metric[mu,nu]
    return E


"""Calculate Stess Energy Tensor"""
def calculate_stress_energy_tensor(E):
    T = {}
    for mu in range(4):
        for nu in range(4):
            T[mu,nu] = c**4/(8*sp.pi*G) * E[mu,nu]
    return T


"""Calculate Geodesics calls on dictionary coordinate func, which maps sympy symbols to sympy functions of the same name for use in differentiaon with repsect to proper time"""
def calculate_gamma_func(metric, coordinates):
    gamma = calculate_christoffel(metric, coordinates)
    sub = {coord: functions[coord] for coord in coordinates}
    gamma_func = {}
    for indices, value in gamma.items():
        gamma_func[indices] = sp.simplify(value.subs(sub))
    return gamma_func


def calculate_geodesics(gamma_func, coordinates, tau):
    coordinates_func = [functions[nu] for nu in coordinates]
    d2xdtau2 = {}
    for mu in range(4):
        geodesic = 0
        for sigma in range(4):
            for rho in range(4):
                geodesic += (gamma_func[mu,sigma,rho] * sp.diff(coordinates_func[sigma], tau) * sp.diff(coordinates_func[rho], tau))
        d2xdtau2[mu] = -(sp.factor(sp.cancel(sp.together(sp.trigsimp(sp.simplify(geodesic))))))
    return d2xdtau2

def create_gamma_numeric(gamma_func, coordinates, grav_const, mass, lightspeed, spin_parameter):
    gamma_numeric = {}
    coordinates_func = [functions[nu] for nu in coordinates]
    for indices, expression in gamma_func.items():
        expression_numeric = expression.subs({G : grav_const, M : mass, c : lightspeed, a : spin_parameter})
        gamma_numeric[indices] = sp.lambdify(coordinates_func, expression_numeric, modules ="numpy")
    return gamma_numeric
def metric_numeric(metric, coordinates, grav_const, mass, lightspeed, spin_parameter):
    metric_sub = metric.subs({G : grav_const, M : mass, c : lightspeed, a : spin_parameter})
    metric_callable = sp.lambdify(coordinates, metric_sub, modules = "numpy")
    return metric_callable
def gamma_value(position, gamma_numeric):
    value = {}
    for indices, gamma_callable in gamma_numeric.items():
        value[indices] = gamma_callable(*position)
    return value
def cartesian_to_spherical(starting_position, three_velocity):
    spherical_position = [0,0,0]
    spherical_position[0] = np.sqrt(starting_position[0]**2 + starting_position[1]**2 + starting_position[2]**2)
    spherical_position[1] = np.arccos(starting_position[2]/spherical_position[0])
    spherical_position[2] = np.arctan2(starting_position[1], starting_position[0])
    e_r = np.array([np.sin(spherical_position[1])*np.cos(spherical_position[2]),np.sin(spherical_position[1])*np.sin(spherical_position[2]),np.cos(spherical_position[1])])
    e_theta = np.array([np.cos(spherical_position[1])*np.cos(spherical_position[2]),np.cos(spherical_position[1])*np.sin(spherical_position[2]),-np.sin(spherical_position[1])])
    e_phi = np.array([-np.sin(spherical_position[2]),np.cos(spherical_position[2]),0])
    spherical_velocity = [0,0,0]
    spherical_velocity[0] = np.dot(three_velocity,e_r)
    spherical_velocity[1] = np.dot(three_velocity,e_theta)
    spherical_velocity[2] = np.dot(three_velocity,e_phi)
    return *spherical_position, * spherical_velocity
