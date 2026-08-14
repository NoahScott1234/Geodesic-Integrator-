import sympy as sp

"""Define algebraic symbols"""
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

"""Define Physical Qauntities """
c = sp.symbols('c' ,positive = True)
G = sp.symbols("G", positive = True)
M = sp.symbols("M", positive = True)
J = sp.symbols("J")
a = sp.symbols('a', positive = True)
k = sp.symbols("k")
Lambda = sp.symbols('Lambda')

Sigma = sp.symbols('Sigma', positive = True)
Delta = sp.symbols('Delta', posotive = True)

coordinates_dict = {
    'Manual' : '[]',
    'Cartesian' : [t, x, y, z],
    'Spherical' : [t, r, theta, phi],
    'Cylindrical' : [t, r, phi, z],
    'Null (Cartesian)' : [u, v, y, z],
    'Null (Kruskal)' : [U, V, theta, phi],
}
def minkowski_spherical():
    metric = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)
    coordinates = coordinates_dict['Spherical']
    return metric, coordinates
def minkowksi_cartesian():
    metric = sp.diag(-1, 1, 1, 1)
    coordinates = coordinates_dict['Cartesian']
    return metric, coordinates
def schwarzschild_metric():
    f = 1 - ((2 * G * M) / (c**2  * r))
    metric = sp.diag(-f * c**2, 1/f, r**2, r**2 * sp.sin(theta)**2)
    coordinates = coordinates_dict['Spherical']
    return metric, coordinates
def kerr_metric():
    twoMR = 2*M*r
    r2 = r**2
    sin2 = sp.sin(theta)**2
    a2 = a**2
    Sigma = r2 + a2 * sp.cos(theta)**2
    Delta = r2 - twoMR + a2
    gtt = -(1-(twoMR)/Sigma)
    gphit = -(twoMR * a * sin2)/Sigma
    grr = Sigma/Delta
    gthetatheta = Sigma
    gphiphi = sin2 * ((r2 + a2) + (twoMR * a2 * sin2)/Sigma)
    metric = sp.Matrix([
        [gtt,0,0,gphit],
        [0,grr,0,0],
        [0,0,gthetatheta,0],
        [gphit,0,0,gphiphi]
    ])
    display_metric = sp.Matrix([
        [-(1-(2*M*r)/Sigma),0,0,-(2*M*(a)*r*sp.sin(theta)**2)/Sigma],
        [0,Sigma/Delta,0,0],
        [0,0,Sigma,0],
        [-(2*M*(a)*r*sp.sin(theta)**2)/Sigma,0,0,sp.sin(theta)**2*((r**2+(a)**2)+(2*M*(a)**2*r*sp.sin(theta)**2)/Sigma)]]
    )
    coordinates = coordinates_dict['Spherical']
    return metric, coordinates
def flrw_metric():
    metric = sp.diag(-1, a**2, a**2, a**2)
    coordinates = coordinates_dict['Cartesian']
    return metric, coordinates
def flrw_spherical_metric():
    curvature = 1/(1-k*r**2)
    metric = sp.diag(-1, a**2*(curvature), a**2*r**2, a**2*r**2*sp.sin(theta)**2)
    coordinates = coordinates_dict['Spherical']
    return metric, coordinates
def manual_entry():
    metric = sp.zeros(4)
    coordinates = coordinates_dict['Manual']
    return metric, coordinates

METRICS = {
    'Minkowksi Cartesian' : minkowksi_cartesian,
    'Minkowksi Spherical' : minkowski_spherical,
    'Schwarzchild': schwarzschild_metric,
    'FLRW (Cartesian)': flrw_metric,
    'FLRW (Spherical)': flrw_spherical_metric,
    'Kerr': kerr_metric,
    'Manual' : manual_entry
}
  
GEODESICS = {
    'Schwarzschild Metric' : schwarzschild_metric,
    'Kerr Metric' : kerr_metric
}