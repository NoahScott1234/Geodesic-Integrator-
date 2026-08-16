import streamlit as st
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

mu = sp.symbols("mu")
nu = sp.symbols("nu")
sigma = sp.symbols("sigma")
rho = sp.symbols("rho")
epsilon = sp.symbols("epsilon")
alpha = sp.symbols("alpha")

coordinates_dict = {
    'Manual' : '[]',
    'Cartesian' : [t, x, y, z],
    'Spherical' : [t, r, theta, phi],
    'Cylindrical' : [t, r, phi, z],
    'Null (Cartesian)' : [u, v, y, z],
    'Null (Kruskal)' : [U, V, theta, phi],
}

c = sp.symbols('c', positive = True)
G = sp.symbols("G", positive = True)
M = sp.symbols("M", positive = True)
a = sp.symbols('a', positive = True)
J = sp.symbols("J")
A = sp.symbols("A")
k = sp.symbols('k')
pi = sp.symbols('pi')
local_symbols = {
    't' : t,
    'r': r,
    'theta' : theta,
    'phi': phi,
    'x' : x,
    'y' : y,
    'z' : z,
    'tau':'tau',
    'sin' : sp.sin,
    'cos' : sp.cos,
    'G' : G,
    'M' : M,
    'exp' : sp.exp
}

Lambda = sp.symbols('Lambda')
Sigma = sp.symbols('Sigma')
Delta = sp.symbols('Delta')
playplaceholder = st.empty()
left, right = st.columns([4,2])

if "trajectory" not in st.session_state:
    st.session_state.trajectory = None

st.set_page_config(page_title="Geodesic Integrator", layout="wide")

from Metrics import GEODESICS
from GR_Engine2 import (calculate_gamma_func, create_gamma_numeric, metric_numeric, gamma_value, cartesian_to_spherical)

three_velocity = [0,0,0]
starting_position = [0,0,0]

if st.sidebar.button('Reset Trajectory'):
    st.session_state.trajectory = None
with right:
    metric_name = st.selectbox("Select a metric:", list(GEODESICS.keys()))
metric, coordinates = GEODESICS[metric_name]()

particle_type = st.sidebar.segmented_control("Particle Type",["Massive Particle", "Photon"],default="Massive Particle")
if particle_type == 'Massive Particle':
    massive = True
else:
    massive = False

plot_settings = st.sidebar.segmented_control('Plot Settings', ['3D Plot', '2D+1D Plot', '2D Plot'], default='3D Plot')
if plot_settings == '3D Plot':
    ThreeD = True
    twoplus1 = False
    twoD = False
if plot_settings == '2D+1D Plot':
    twoplus1 = True
    ThreeD = False
    twoD = False
if plot_settings == '2D Plot':
    twoD = True
    ThreeD = False
    twoplus1 = False

with st.sidebar.expander('Input Options', expanded = True):
    spherical = st.toggle('Spherical Coordinates', value = True)
lightspeed = 1
grav_const = 1

if spherical:
    input_coordinates = coordinates_dict['Spherical']
else:
    input_coordinates = coordinates_dict['Cartesian']



loadplaceholder = st.empty()
prev_metric = st.session_state.get('metric')
if prev_metric is None or prev_metric != metric_name:
    st.session_state['trajectory'] = None
    st.session_state['metric'] = metric_name
    st.session_state['christoffel'] = None
gamma_func = st.session_state.get('christoffel')
with loadplaceholder.spinner('Calculating Connection Coeffecients...'):
    if gamma_func is None:
        gamma_func = calculate_gamma_func(metric, coordinates)
        st.session_state['christoffel'] = gamma_func
with st.sidebar.expander('Settings'):
    if metric_name == 'Kerr Metric':
        spin_parameter = st.number_input('Spin Parameter', min_value=0.0, max_value=1.0, value = 0.5)
    else:
        spin_parameter = 0
    mass = st.number_input('Object Mass', min_value=0.0, value = 1)
    sim_run_time = st.number_input('Simulation Run Time (Proper Time)', min_value = 0.0, value = 100)

columns = st.sidebar.columns(3)
for mu in range (3):
    entry = columns[mu].number_input(f'd{input_coordinates[mu+1]}/dtau', key=f'd{mu}/dtau')
    three_velocity[mu] = entry
columns1 = st.sidebar.columns(3)
for mu in range(3):
    entry = columns1[mu].number_input(f'{(input_coordinates[mu+1])}', key = f'{mu} pos ')
    entry = float(entry)
    starting_position[mu] = entry
if spherical == False:
    spherical_state = cartesian_to_spherical(starting_position, three_velocity)
    spherical_position = spherical_state[:3]
    spherical_velocity = spherical_state[3:]


with st.sidebar.expander('Select Quantities to Plot'):
    with st.expander('Velocities (Proper Time)'):
        show_dt_dtau = st.checkbox(f'Show Time Velocity Plot', value = True)
        show_dx1_dtau = st.checkbox(f'Show {coordinates[1]} Velocity Plot', value = True)
        show_dx2_dtau = st.checkbox(f'Show {coordinates[2]} Velocity Plot', value = True)
        show_dx3_dtau = st.checkbox(f'Show {coordinates[3]} Velocity Plot', value = True)
    with st.expander('Velocities (Coordinate Time)'):
        show_dx1_dt =  st.checkbox(f'Show {coordinates[1]} Coordinate Velocity Plot', value = True)
        show_dx2_dt =  st.checkbox(f'Show {coordinates[2]} Coordinate Velocity Plot', value = True)
        show_dx3_dt =  st.checkbox(f'Show {coordinates[3]} Coordinate Velocity Plot', value = True)
    with st.expander('Positions'):
        show_t = st.checkbox(f'Show Time Plot', value = True)
        show_dx1 = st.checkbox(f'Show {coordinates[1]} Plot', value = True)
        show_dx2 = st.checkbox(f'Show {coordinates[2]} Plot', value = True)
        show_dx3 = st.checkbox(f'Show {coordinates[3]} Plot', value = True)
    with st.expander('Momenta'):
        show_pt = st.checkbox(f'Show Time Momentum', value = True)
        show_pr = st.checkbox(f'Show {coordinates[1]} Momentum', value = True)
        show_ptheta = st.checkbox(f'Show {coordinates[2]} Momentum', value = True)
        show_phi = st.checkbox(f'Show {coordinates[3]} Momentum', value = True)
    with st.expander('Conserved Qunatities'):
        show_energy = st.checkbox(f'Show Energy Plot', value = True)
        show_angular_momentum = st.checkbox(f'Show Momentum', value = True)
        show_carter_constant = st.checkbox(f'Show Carter Constant')
    show_velocity = st.checkbox(f'Show velocity', value = True)
    show_gtt = st.checkbox(f'Show Metric Time-Time Component', value = True)
    show_grr = st.checkbox(f'Show Metric {coordinates[1]}-{coordinates[1]} Component', value = True)

if st.session_state.trajectory is None:
    with right:
        st.markdown("Metric Tensor")
        st.latex(r"g_{\mu\nu} = " + sp.latex(metric))
        st.markdown("Coordinates")
        st.latex(r"x^\mu = \left(" + ", ".join(sp.latex(i) for i in input_coordinates) + r"\right)")
        st.markdown("Initial 3 Velocity")
        st.latex(f"[{three_velocity[0]}], [{three_velocity[1]}], [{three_velocity[2]}]")
        st.markdown("Initial Position")
        st.latex(f"[{starting_position[0]}], [{starting_position[1]}], [{starting_position[2]}]")

def position_plot(starting_position, three_velocity):
    x_position_plot = starting_position[0] * np.sin(starting_position[1]) * np.cos(starting_position[2])
    y_position_plot = starting_position[0] * np.sin(starting_position[1]) * np.sin(starting_position[2])
    z_position_plot = starting_position[0] * np.cos(starting_position[1])
    e_r = np.array([np.sin(starting_position[1])*np.cos(starting_position[2]),np.sin(starting_position[1])*np.sin(starting_position[2]),np.cos(starting_position[1])])
    e_theta = np.array([np.cos(starting_position[1])*np.cos(starting_position[2]),np.cos(starting_position[1])*np.sin(starting_position[2]),-np.sin(starting_position[1])])
    e_phi = np.array([-np.sin(starting_position[2]),np.cos(starting_position[2]),0])
    position_plot = [x_position_plot, y_position_plot, z_position_plot]
    velocity_plot = (three_velocity[0] * e_r + three_velocity[1] * e_theta + three_velocity[2] * e_phi)
    norm_vector = np.linalg.norm(velocity_plot)
    norm_velocity = velocity_plot/norm_vector
    return [*position_plot, *norm_velocity]

with left:
    loadplaceholder = st.empty()
    sliderplaceholder = st.empty()
    placeholder = st.empty()
    placeholder3d = st.empty()
    if st.session_state.trajectory is None:
        fig_preview, a1x = plt.subplots()
        if ThreeD:
            figure1 = go.Figure()
            if metric_name == 'Schwarzschild Metric':
                r_h = 2 * grav_const * mass / lightspeed**2
                r_p = 3 * grav_const * mass / lightspeed**2
                u = np.linspace(0, 2*np.pi, 60)
                v = np.linspace(0, np.pi, 30)
                U, V = np.meshgrid(u, v)
                X = r_h * np.cos(U) * np.sin(V)
                Y = r_h * np.sin(U) * np.sin(V)
                Z = r_h * np.cos(V)
                figure1.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=1,name="Event Horizon"))
                X = r_p * np.cos(U) * np.sin(V)
                Y = r_p * np.sin(U) * np.sin(V)
                Z = r_p * np.cos(V)
                figure1.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "red"], [1, "red"]],showscale=False,opacity=0.01,name="Photon Sphere"))
            if metric_name == 'Kerr Metric':
                r_o = grav_const * mass + np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                r_i = grav_const * mass - np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                u = np.linspace(0, 2*np.pi, 60)
                v = np.linspace(0, np.pi, 30)
                U, V = np.meshgrid(u, v)
                X = r_i * np.cos(U) * np.sin(V)
                Y = r_i * np.sin(U) * np.sin(V)
                Z = r_i * np.cos(V)
                figure1.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=1,name="Inner Event Horizon"))
                X = r_o * np.cos(U) * np.sin(V)
                Y = r_o * np.sin(U) * np.sin(V)
                Z = r_o * np.cos(V)
                figure1.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=0.8,name="Outer Event Horizon"))
                r_e = np.sqrt((grav_const * mass)**2 -(spin_parameter * np.cos(V))**2) + grav_const * mass
                X = r_e * np.cos(U) * np.sin(V)
                Y = r_e * np.sin(U) * np.sin(V)
                Z = r_e * np.cos(V)
                figure1.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "yellow"], [1, "yellow"]],showscale=False,opacity=0.1,name="Ergosphere"))
            figure1.update_layout(scene=dict(aspectmode="data"))
            if spherical:
                values = position_plot(starting_position, three_velocity)
                [x0,y0,z0] = values[:3]
                [vx,vy,vz] = values[3:]
            else:
                [x0,y0,z0] = starting_position
                [vx,vy,vz] = three_velocity
            
            figure1.add_trace(go.Cone(x=[x0],y=[y0],z=[z0],u=[vx],v=[vy],w=[vz],sizemode="absolute",
            sizeref=0.2,showscale=False,colorscale=[[0, "red"], [1, "red"]],anchor="tip"))
            placeholder3d.plotly_chart(figure1, use_container_width=True)
        else:
            if metric_name =='Kerr Metric':
                r_o = grav_const * mass + np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                r_i = grav_const * mass - np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                outer_horizon = Circle((0,0), r_o, color = 'black', fill = False)
                inner_horizon = Circle((0,0), r_i, color = 'black', fill = False)
                a1x.add_patch(outer_horizon)
                a1x.add_patch(inner_horizon)
            if metric_name == 'Schwarzschild Metric':
                r_h = 2 * grav_const * mass / lightspeed**2
                r_p = 3 * grav_const * mass / lightspeed**2
                photon_sphere = Circle((0,0), r_p, color = 'yellow', fill = False, linewidth = 2, linestyle = '--')
                a1x.add_patch(photon_sphere)
                horizon = Circle((0,0), r_h, color = 'black', fill = True)
                a1x.add_patch(horizon)
            if spherical:
                pos = position_plot(starting_position, three_velocity)
                values = pos[0],pos[1],pos[3],pos[4]
            else:
                values = [starting_position[0],starting_position[1],three_velocity[0],three_velocity[1]]
            a1x.quiver(*values)
            a1x.axis("equal")
            placeholder.pyplot(fig_preview)

if spherical == False:
    starting_position = spherical_position
    three_velocity = spherical_velocity
starting_position = [0,*starting_position]

        
with left:
    
    if st.button('Plot Geodesic'):
        with loadplaceholder.spinner('Plotting Geodesic'):
            gamma_numeric = create_gamma_numeric(gamma_func, coordinates, grav_const, mass, lightspeed, spin_parameter)
            def acceleration_value(gamma_val, velocities):
                acceleration = [0,0,0,0]
                for mu in range (4):
                    total = 0
                    for rho in range(4):
                        for sigma in range(4):
                            total -= gamma_val[mu, rho, sigma] * velocities[rho] * velocities[sigma]
                    acceleration[mu] = total
                return acceleration
            def geodesic_rhs(tau, state, gamma_numeric):
                positions = state[:4]
                velocities = state[4:]
                gamma_val = gamma_value(positions, gamma_numeric)
                accelerations = acceleration_value(gamma_val, velocities)
                return list(velocities) + list(accelerations)
            def intial_velocity_normalization(position, three_velocity):
                velocity_mag = np.sqrt(three_velocity[0]**2 + three_velocity[1]**2 + three_velocity[2]**2)
                if velocity_mag > lightspeed:
                    raise ValueError("3 velocity magnitude exceeds c!")
                lorentz_factor = 1/(np.sqrt(1 - (velocity_mag/lightspeed)**2))
                velocity_4 = ([lorentz_factor,lorentz_factor*three_velocity[0],lorentz_factor*three_velocity[1],lorentz_factor*three_velocity[2]])
                metric_callable = metric_numeric(metric, coordinates, grav_const, mass, lightspeed, spin_parameter)
                metric_val = np.array(metric_callable(*position),dtype=float)
                if metric_name == 'Schwarzschild Metric':
                    tetrad = np.zeros((4,4))
                    for mu in range(4):
                        tetrad[mu,mu] = np.sqrt(np.abs(1/metric_val[mu,mu]))
                if metric_name == 'Kerr Metric':
                    g = metric_val 
                    omega = -g[0,3]/g[3,3]
                    A = 1/np.sqrt(g[0,3]**2/g[3,3] - g[0,0])
                    B = omega * A
                    tetrad = np.array([
                            [A,0,0,B],
                            [0,1/np.sqrt(g[1,1]),0,0],
                            [0,0,1/np.sqrt(g[2,2]),0],
                            [0,0,0,1/np.sqrt(g[3,3])]
                        ])
                normalized_4_velocity = {}
                for mu in range (4):
                    metric_4_velocity = 0 
                    for nu in range(4):
                        metric_4_velocity += tetrad[mu,nu] * velocity_4[nu]
                    normalized_4_velocity[mu] = metric_4_velocity
                return normalized_4_velocity
            def photon_normalization(position, three_velocity):
                direction = np.array([*three_velocity])
                norm = np.linalg.norm(direction)
                norm_vector = direction/norm
                photon_vector = np.concatenate((np.array([1]), lightspeed * (norm_vector)))
                metric_callable = metric_numeric(metric, coordinates, grav_const, mass, lightspeed, spin_parameter)                
                metric_val = np.array(metric_callable(*position))
                if metric_name == 'Schwarzschild Metric':
                    tetrad = np.zeros((4,4))
                    for mu in range(4):
                        tetrad[mu,mu] = np.sqrt(np.abs(1/(metric_val[mu,mu])))
                if metric_name == 'Kerr Metric':
                    g = metric_val 
                    omega = -g[0,3]/g[3,3]
                    A = 1/np.sqrt(g[0,3]**2/g[3,3] - g[0,0])
                    B = omega * A
                    tetrad = np.array([
                            [A,0,0,B],
                            [0,1/np.sqrt(g[1,1]),0,0],
                            [0,0,1/np.sqrt(g[2,2]),0],
                            [0,0,0,1/np.sqrt(g[3,3])]
                        ])
                generally_orthonormal_photon_vector = {}
                for mu in range(4):
                    entry = 0
                    for nu in range(4):
                        entry += tetrad[mu,nu] * photon_vector[nu]
                    generally_orthonormal_photon_vector[mu] = entry  
        
                return generally_orthonormal_photon_vector
            if massive:
                starting_velocity = intial_velocity_normalization(starting_position, three_velocity)
            else:
                starting_velocity = photon_normalization(starting_position, three_velocity)
            initial_state = starting_position + list(starting_velocity.values())
            solution = solve_ivp(fun=lambda tau, state: geodesic_rhs(tau,state,gamma_numeric),t_span=(0.0, sim_run_time),y0=initial_state,
            method="RK45",max_step=0.05,rtol=1e-9,atol=1e-11)

            tau_values = solution.t
            t_values = solution.y[0]
            x1_values = solution.y[1]
            x2_values = solution.y[2]
            x3_values = solution.y[3]
            dt_dtau_values = solution.y[4]
            dx1_dtau_values = solution.y[5]
            dx2_dtau_values = solution.y[6]
            dx3_dtau_values = solution.y[7]
            
            if coordinates == [t,r,theta,phi]:
                x_values = x1_values * np.cos(x3_values) * np.sin(x2_values)
                y_values = x1_values * np.sin(x3_values) * np.sin(x2_values)
                z_values = x1_values * np.cos(x2_values)

            if coordinates == [t,x,y,z]:
                x_values = x1_values
                y_values = x2_values
                z_values = x3_values

            st.session_state.trajectory = {
                'tau' : tau_values,
                'x' : x_values,
                'y' : y_values,
                'z' : z_values,
                't' : t_values,
                'r' : x1_values,
                'theta' : x2_values,
                'phi' : x3_values,
                'dt/dtau' : dt_dtau_values,
                'dx1/dtau' : dx1_dtau_values,
                'dx2/dtau' : dx2_dtau_values,
                'dx3/dtau' : dx3_dtau_values
            }
            
            
if st.session_state.trajectory is not None:
    trajectory = st.session_state.trajectory
    tau = trajectory['tau']
    t = trajectory['t']
    x = trajectory['x']
    y = trajectory['y']
    z = trajectory['z']
    r = trajectory['r']
    theta = trajectory['theta']
    phi = trajectory['phi']
    dt_dtau = trajectory['dt/dtau']
    if coordinates == coordinates_dict['Cartesian']:
        positions = np.column_stack((t, x, y, z))
        dx_dtau = trajectory['dx1/dtau']
        dy_dtau = trajectory['dx2/dtau']
        dz_dtau = trajectory['dx3/dtau']
        velocity_list = np.column_stack((dt_dtau, dx_dtau, dy_dtau, dz_dtau))
    if coordinates == coordinates_dict['Spherical']:
        positions = np.column_stack((t,r,theta,phi))
        dr_dtau = trajectory['dx1/dtau']
        dtheta_dtau = trajectory['dx2/dtau']
        dphi_dtau = trajectory['dx3/dtau']
        velocity_list = np.column_stack((dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau))

    r_h = 2 * grav_const * mass / lightspeed**2
    r_p = 3 * grav_const * mass / lightspeed**2

    with left:
        frame = sliderplaceholder.slider('Proper Time', min_value=0, max_value=len(tau)-1,value = 0, step = 1)
        if twoplus1:
            figuret = go.Figure()
            figuret.add_trace(go.Scatter3d(x=x[:frame+1],y=y[:frame+1],z=t[:frame+1], mode="lines",name="Geodesic"))
            u = np.linspace(0, 2*np.pi, 60)
            j = np.linspace(t[0],t[frame],100)
            U, Z = np.meshgrid(u, j)
            X = r_h * np.cos(U)
            Y = r_h * np.sin(U)
            figuret.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=1,name="Event Horizon"))
            figuret.update_layout(scene=dict(aspectmode="data"))
            placeholder3d.plotly_chart(figuret, use_container_width=True)
        elif ThreeD:
            figure = go.Figure()
            figure.add_trace(go.Scatter3d(x=x[:frame+1],y=y[:frame+1],z=z[:frame+1], mode="lines",name="Geodesic"))
            if metric_name == 'Schwarzschild Metric':
                u = np.linspace(0, 2*np.pi, 60)
                v = np.linspace(0, np.pi, 30)
                U, V = np.meshgrid(u, v)
                X = r_h * np.cos(U) * np.sin(V)
                Y = r_h * np.sin(U) * np.sin(V)
                Z = r_h * np.cos(V)
                figure.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=1,name="Event Horizon"))
                X = r_p * np.cos(U) * np.sin(V)
                Y = r_p * np.sin(U) * np.sin(V)
                Z = r_p * np.cos(V)
                figure.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "red"], [1, "red"]],showscale=False,opacity=0.01,name="Photon Sphere"))
            if metric_name == 'Kerr Metric':
                r_o = grav_const * mass + np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                r_i = grav_const * mass - np.sqrt(grav_const**2 * mass**2 - spin_parameter**2)
                u = np.linspace(0, 2*np.pi, 60)
                v = np.linspace(0, np.pi, 30)
                U, V = np.meshgrid(u, v)
                X = r_i * np.cos(U) * np.sin(V)
                Y = r_i * np.sin(U) * np.sin(V)
                Z = r_i * np.cos(V)
                figure.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=1,name="Inner Event Horizon"))
                X = r_o * np.cos(U) * np.sin(V)
                Y = r_o * np.sin(U) * np.sin(V)
                Z = r_o * np.cos(V)
                figure.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "black"], [1, "black"]],showscale=False,opacity=0.8,name="Outer Event Horizon"))
                r_e = np.sqrt((grav_const * mass)**2 -(spin_parameter * np.cos(V))**2) + grav_const * mass
                X = r_e * np.cos(U) * np.sin(V)
                Y = r_e * np.sin(U) * np.sin(V)
                Z = r_e * np.cos(V)
                figure.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0, "yellow"], [1, "yellow"]],showscale=False,opacity=0.1,name="Ergosphere"))
            figure.update_layout(scene=dict(aspectmode="data"))
            placeholder3d.plotly_chart(figure, use_container_width=True)
        else:
            fig1, ax = plt.subplots()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axis("equal")
            if massive:
                ax.set_title(f'Proper Time: {tau[frame]:.3f}')
            else:
                ax.set_title(f'Affine Paramter: {tau[frame]:.3f}')
            horizon = Circle((0,0), r_h, color = 'black', fill = True)
            photon_sphere = Circle((0,0), r_p, color = 'yellow', fill = False, linewidth = 2, linestyle = '--')
            ax.add_patch(horizon)
            ax.add_patch(photon_sphere)
            ax.plot(x[:frame+1], y[:frame+1])
            placeholder.pyplot(fig1)
        
        if spherical:
            coordinate_positions = [t[frame], r[frame], theta[frame], phi[frame]]
            coordinate_velocites = [dt_dtau[frame], dr_dtau[frame], dtheta_dtau[frame], dphi_dtau[frame]]
        else:
            e_r = np.array([np.sin(theta[frame])*np.cos(phi[frame]),np.sin(theta[frame])*np.sin(phi[frame]),np.cos(theta[frame])])
            e_theta = np.array([np.cos(theta[frame])*np.cos(phi[frame]),np.cos(theta[frame])*np.sin(phi[frame]),-np.sin(theta[frame])])
            e_phi = np.array([-np.sin(phi[frame]),np.cos(phi[frame]),0])
            velocities = (dr_dtau[frame] * e_r + dtheta_dtau[frame] * e_theta + dphi_dtau[frame] * e_phi)
            coordinate_positions = [t[frame], x[frame], y[frame], z[frame]]
            coordinate_velocites = [dt_dtau[frame],*velocities]
        with right:
            st.markdown(f'Proper Time: {tau[frame]}')
            col1, col2 = st.columns(2)
            decimal_accuracy = st.number_input('Decimal Accuracy', min_value=0, max_value=12, value = 2, step = 1)
            st.latex(r"x^\mu = \left(" + ", ".join(sp.latex(i) for i in input_coordinates) + r"\right)")
            st.latex(r"x^\mu = \left(" + ", ".join(f'{float(i):.{int(decimal_accuracy)}f}' for i in coordinate_positions) + r"\right)")
            st.latex(r"\frac{dx^\mu}{d\tau} = \left(" + ", ".join(rf"\frac{{d\,{sp.latex(i)}}}{{d\tau}}" for i in input_coordinates) + r"\right)")
            st.latex(r"\frac{dx^\mu}{d\tau} = \left(" + ", ".join(f'{float(i):.{int(decimal_accuracy)}f}' for i in coordinate_velocites) + r"\right)")
    p = np.zeros((len(tau), 4))
    Killing_Vectors= []
    K = [0,0,0,0]
    energy = np.zeros((len(tau)))
    angular_momentum = np.zeros((len(tau)))
    Q = np.zeros((len(tau)))
    Q2 = np.zeros((len(tau)))
    mew = 1 if massive else 0
    for mu in range(4):
        if sp.diff(metric, coordinates[mu]) == sp.zeros(4):
            K = np.zeros(4)
            K[mu] = 1
            Killing_Vectors.append(K)
    metric_callable = metric_numeric(metric, coordinates, grav_const, mass, lightspeed, spin_parameter)
    metric_values = np.array([metric_callable(*point) for point in positions])
    for i in range(len(tau)):
        for mu in range(4):
            momentum = 0
            for nu in range(4):
                momentum += metric_values[i, mu, nu] * velocity_list[i, nu]
            p[i,mu] = momentum
        L = 0
        E = 0
        for mu in range (4):
            E += p[i,mu] * Killing_Vectors[0][mu]
        energy[i] = -E
        for mu in range(4):
            L += p[i,mu]*Killing_Vectors[1][mu]
        angular_momentum[i] = L
        Q[i] = (p[i,2])**2 + np.cos(positions[i,2])**2 * (spin_parameter*spin_parameter*(mew**2 - energy[i]**2) + (angular_momentum[i]**2/(np.sin(positions[i,2])**2)))
        Q2[i] = p[i,2]**2 + np.cos(positions[i,2])**2 * spin_parameter**2 * (mew**2 - energy[0]**2) + angular_momentum[0]**2 /(np.tan(positions[i,2])**2)
    left1, middle, right1 = st.columns([3,3,3])
    with left1:
        with st.expander('Velocities (Proper Time)'):
            if show_dt_dtau:
                fig2, bx = plt.subplots()
                bx.plot(tau, dt_dtau)
                bx.set_ylabel("Time Velocity")
                if massive:
                    bx.set_xlabel('Proper Time')
                else: 
                    bx.set_xlabel('Affine Parameter')
                st.pyplot(fig2)
            if show_dx1_dtau:
                fig3, cx = plt.subplots()
                cx.plot(tau, dr_dtau)
                cx.set_ylabel(f'{coordinates[1]} Velocity')
                if massive:
                    cx.set_xlabel('Proper Time')
                else: 
                    cx.set_xlabel('Affine Parameter')
                st.pyplot(fig3)
            if show_dx2_dtau:
                fig4, dx = plt.subplots()
                dx.plot(tau, dtheta_dtau)
                dx.set_ylabel(f'{coordinates[2]} Velocity')
                if massive:
                    dx.set_xlabel('Proper Time')
                else: 
                    dx.set_xlabel('Affine Parameter')
                st.pyplot(fig4)
            if show_dx3_dtau:
                fig5, ex = plt.subplots()
                ex.plot(tau, dphi_dtau)
                ex.set_ylabel(f'{coordinates[3]} Velocity')
                if massive:
                    ex.set_xlabel('Proper Time')
                else: 
                    ex.set_xlabel('Affine Parameter')
                st.pyplot(fig5)
            if show_velocity:
                velocity = np.sqrt(dr_dtau**2 + dtheta_dtau**2 + dphi_dtau**2)
                fig12, lx = plt.subplots()
                lx.plot(tau, velocity)
                lx.set_ylabel('Velocity')
                if massive:
                    lx.set_xlabel('Proper Time')
                else: 
                    lx.set_xlabel('Affine Parameter')
                st.pyplot(fig12)
    with middle:
        with st.expander('Positions'):
            if show_t:
                fig6, fx = plt.subplots()
                fx.plot(tau, t)
                fx.set_ylabel("Time")
                if massive:
                    fx.set_xlabel('Proper Time')
                else: 
                    fx.set_xlabel('Affine Parameter')
                st.pyplot(fig6)
            if show_dx1:
                fig7, gx = plt.subplots()
                gx.plot(tau, r)
                gx.set_ylabel(f'{coordinates[1]}')
                if massive:
                    gx.set_xlabel('Proper Time')
                else: 
                    gx.set_xlabel('Affine Parameter')
                st.pyplot(fig7)
            if show_dx2:
                fig8, hx = plt.subplots()
                hx.plot(tau, theta)
                hx.set_ylabel(f'{coordinates[2]}')
                if massive:
                    hx.set_xlabel('Proper Time')
                else: 
                    hx.set_xlabel('Affine Parameter')
                st.pyplot(fig8)
            if show_dx3:
                fig9, ix = plt.subplots()
                ix.plot(tau, phi)
                ix.set_ylabel(f'{coordinates[3]}')
                if massive:
                    ix.set_xlabel('Proper Time')
                else: 
                    ix.set_xlabel('Affine Parameter')
                st.pyplot(fig9)
        with st.expander('Momenta'):
            if show_pt:
                fig20, tx = plt.subplots()
                tx.plot(tau, p[:,0])
                tx.set_ylabel(f'{coordinates[0]} Momentum')
                if massive:
                    tx.set_xlabel('Proper Time')
                else: 
                    tx.set_xlabel('Affine Parameter')
                st.pyplot(fig20)
            if show_pr:
                fig21, ux = plt.subplots()
                ux.plot(tau,  p[:,1])
                ux.set_ylabel(f'{coordinates[1]} Momentum')
                if massive:
                    ux.set_xlabel('Proper Time')
                else: 
                    ux.set_xlabel('Affine Parameter')
                st.pyplot(fig21)
            if show_ptheta:
                fig23, wx = plt.subplots()
                p_theta = velocity_list[:,2]*metric_values[:,2,2]
                wx.plot(tau, p_theta)
                wx.set_ylabel('Theta Momentum')
                if massive:
                    wx.set_xlabel('Proper Time')
                else: 
                    wx.set_xlabel('Affine Parameter')
                st.pyplot(fig23)
                fig19, xx = plt.subplots()
                xx.plot(tau, p[:,2])
                xx.set_ylabel(f'{coordinates[2]} Momentum')
                if massive:
                    xx.set_xlabel('Proper Time')
                else: 
                    xx.set_xlabel('Affine Parameter')
                st.pyplot(fig19)
            if show_ptheta:
                fig22, vx = plt.subplots()
                vx.plot(tau, p[:,3])
                vx.set_ylabel(f'{coordinates[3]} Momentum')
                if massive:
                    vx.set_xlabel('Proper Time')
                else: 
                    vx.set_xlabel('Affine Parameter')
                st.pyplot(fig22)
    with right1:
        with st.expander('Conserved Qunatities'):
            if show_energy:
                fig10, jx = plt.subplots()
                jx.plot(tau, energy)
                jx.set_ylabel('Energy')
                if massive:
                    jx.set_xlabel('Proper Time')
                else: 
                    jx.set_xlabel('Affine Parameter')
                st.pyplot(fig10)
            if show_angular_momentum:
                fig11, kx = plt.subplots()
                kx.plot(tau, angular_momentum)
                kx.set_ylabel('Angular Momentum')
                if massive:
                    kx.set_xlabel('Proper Time')
                else: 
                    kx.set_xlabel('Affine Parameter')
                st.pyplot(fig11)
            if show_carter_constant:
                fig18, rx = plt.subplots()
                rx.plot(tau, Q2)
                rx.set_ylabel('Carter Constant')
                if massive:
                    rx.set_xlabel('Proper Time')
                else: 
                    rx.set_xlabel('Affine Parameter')
                st.pyplot(fig18)
                fig24, zx = plt.subplots()
                Q_error =(Q2[:len(tau)] - Q2[0])/Q2[0]
                zx.plot(tau, Q_error)
                zx.set_ylabel('Carter Constant Error')
                if massive:
                    zx.set_xlabel('Proper Time')
                else: 
                    zx.set_xlabel('Affine Parameter')
                st.pyplot(fig24)
    with left1:
        with st.expander('Coordinate Velocities'):
            if show_dx1_dt:
                dx1_dt = dr_dtau/ dt_dtau
                fig13, mx = plt.subplots()
                mx.plot(t, dx1_dt)
                mx.set_ylabel(f'{coordinates[1]} Velocity')
                mx.set_xlabel('Coordinate Time')
                st.pyplot(fig13)
            if show_dx2_dt:
                dx2_dt = dtheta_dtau/ dt_dtau
                fig14, nx = plt.subplots()
                nx.plot(t, dx2_dt)
                nx.set_ylabel(f'{coordinates[2]} Velocity')
                nx.set_xlabel('Coordinate Time')
                st.pyplot(fig14)
            if show_dx3_dt:
                dx3_dt = dphi_dtau/ dt_dtau
                fig15, ox = plt.subplots()
                ox.plot(t, dx3_dt)
                ox.set_ylabel(f'{coordinates[3]} Velocity')
                ox.set_xlabel('Coordinate Time')
                st.pyplot(fig15)
    with right1:
        with st.expander('Metric Components'):
            if show_gtt:
                g_00_val = np.zeros((len(tau)))
                for i in range((len(tau))):
                    g_00_val[i] = metric_values[i,0,0] 
                fig16, px = plt.subplots()
                px.plot(tau, g_00_val)
                px.set_ylabel(f'g_00 Value')
                if massive:
                    px.set_xlabel('Proper Time')
                else: 
                    px.set_xlabel('Affine Parameter')
                st.pyplot(fig16)
            if show_grr:
                g_11_val = np.zeros((len(tau)))
                for i in range((len(tau))):
                    g_11_val[i] = metric_values[i,1,1]
                fig17, qx = plt.subplots()
                qx.plot(tau, g_11_val)
                qx.set_ylabel(f'g_11 Value')
                if massive:
                    qx.set_xlabel('Proper Time')
                else: 
                    qx.set_xlabel('Affine Parameter')
                st.pyplot(fig17)
