"""Shape-parameterized planar fluid loads for NPendulum absolute angles.

No body outline or fluid type is inferred from mass, pivot spacing, or inertia.
Resistance coefficients include viscosity and are independently supplied.
Moving, rotating, and straining prescribed affine flows are supported within
a quasi-steady linear resistance model. This does not solve fluid evolution.
"""

from dataclasses import dataclass
import numpy as np


Z = np.array([[0.0, -1.0], [1.0, 0.0]])


def vector(value, n, name):
    try:
        out = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or length {n}") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def xy(value, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (2,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite 2-vector")
    return out


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


@dataclass(frozen=True)
class FlowSample:
    velocity: np.ndarray
    spin: np.ndarray
    strain: np.ndarray


class AffineFlow:
    """u(x,t) = U(t) + A(t) @ (x-origin(t)), trace(A)=0.

    Each argument can be constant or a callable of time. Includes uniform
    translation, solid-body rotation, simple shear, and planar extension.
    A is the spatial gradient with A_ij=partial u_i/partial x_j.
    """

    def __init__(self, velocity=(0.0, 0.0), gradient=None, origin=(0.0, 0.0)):
        self.velocity = velocity
        self.gradient = np.zeros((2, 2)) if gradient is None else gradient
        self.origin = origin

    def sample(self, positions, time):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2 or not np.all(np.isfinite(positions)):
            raise ValueError("positions must have shape (N,2) and be finite")
        u = xy(self.velocity(time) if callable(self.velocity) else self.velocity, "flow velocity")
        origin = xy(self.origin(time) if callable(self.origin) else self.origin, "flow origin")
        a = np.asarray(self.gradient(time) if callable(self.gradient) else self.gradient,
                       dtype=float)
        if a.shape != (2, 2) or not np.all(np.isfinite(a)):
            raise ValueError("flow gradient must be a finite 2x2 matrix")
        if abs(np.trace(a)) > 1e-12*max(1.0, np.linalg.norm(a)):
            raise ValueError("This affine model assumes incompressible flow: trace(A)=0")
        count = len(positions)
        velocities = u+(positions-origin)@a.T
        spin = np.full(count, (a[1, 0]-a[0, 1])/2)
        strain = np.broadcast_to((a+a.T)/2, (count, 2, 2)).copy()
        return FlowSample(velocities, spin, strain)


def rotating_flow(spin, *, translation=(0.0, 0.0), origin=(0.0, 0.0)):
    """Prescribed solid-body rotating ambient fluid; no strain."""
    gradient = (lambda t: float(spin(t))*Z) if callable(spin) else float(spin)*Z
    return AffineFlow(translation, gradient, origin)


def shear_flow(rate):
    """u=(rate*y,0), so the ambient spin is -rate/2, not rate."""
    if callable(rate):
        gradient = lambda t: np.array([[0.0, float(rate(t))], [0.0, 0.0]])
    else:
        gradient = np.array([[0.0, float(rate)], [0.0, 0.0]])
    return AffineFlow(gradient=gradient)


@dataclass(frozen=True)
class Kinematics:
    joints: np.ndarray
    centers: np.ndarray
    center_velocities: np.ndarray
    along: np.ndarray
    normal: np.ndarray
    com_offsets: np.ndarray
    body_to_world: np.ndarray
    H: np.ndarray


def kinematics(lengths, a_cm, b_cm, theta, omega, *,
               pivot_position=(0.0, 0.0), pivot_velocity=(0.0, 0.0)):
    lengths = np.asarray(lengths, dtype=float)
    if lengths.ndim != 1 or len(lengths) == 0 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("lengths must be a nonempty positive finite vector")
    count = len(lengths)
    a, b = vector(a_cm, count, "a_cm"), vector(b_cm, count, "b_cm")
    theta, omega = vector(theta, count, "theta"), vector(omega, count, "omega")
    rp, vp = xy(pivot_position, "pivot position"), xy(pivot_velocity, "pivot velocity")
    sn, cs = np.sin(theta), np.cos(theta)
    along = np.column_stack((sn, -cs))
    normal = np.column_stack((cs, sn))
    joints = np.vstack((rp, rp+np.cumsum(lengths[:, None]*along, axis=0)))
    offsets = a[:, None]*normal+b[:, None]*along
    centers = joints[:-1]+offsets
    h = np.zeros((count, 3, count))
    for i in range(count):
        h[i, :2, :i] = (lengths[:i, None]*normal[:i]).T
        h[i, :2, i] = b[i]*normal[i]-a[i]*along[i]
        h[i, 2, i] = 1
    velocities = vp+np.einsum("ijk,k->ij", h[:, :2, :], omega)
    rotation = np.zeros((count, 3, 3))
    rotation[:, 0, 0] = cs
    rotation[:, 0, 1] = -sn
    rotation[:, 1, 0] = sn
    rotation[:, 1, 1] = cs
    rotation[:, 2, 2] = 1
    return Kinematics(joints, centers, velocities, along, normal, offsets, rotation, h)


@dataclass(frozen=True)
class FluidLoads:
    forces: np.ndarray
    torques_about_com: np.ndarray
    torques_about_joint: np.ndarray
    generalized_torques: np.ndarray
    damping_matrix: np.ndarray
    flow_forcing: np.ndarray
    relative_body_velocities: np.ndarray
    resistance_potential: float
    resistance_power: float
    strain_relative_power: float
    ambient_power: float
    mechanical_power: float


class GeneralFluidModel:
    """Planar body resistance at each COM, in body Cartesian x-right/y-up axes.

    resistance_body: (N,3,3), or one shared (3,3), maps [Vx,Vy,omega] to
    minus [Fx,Fy,T_COM]. It must be symmetric positive semidefinite.

    strain_response_body: (N,3,2), or shared (3,2), maps [E_xx_body,E_xy_body]
    to added [Fx,Fy,T_COM]. E is symmetric and trace-free. Its entries are
    separate shape-dependent coefficients. None is accepted for zero-strain
    flows only. Explicitly pass zeros to deliberately assume zero response.

    Single-body resistance blocks neglect inter-body hydrodynamic interaction.
    A global coupled hydrodynamic matrix would be needed to include it.
    """

    def __init__(self, lengths, a_cm, b_cm, resistance_body,
                 strain_response_body=None):
        lengths = np.asarray(lengths, dtype=float)
        kinematics(lengths, a_cm, b_cm, np.zeros(len(lengths)), np.zeros(len(lengths)))
        self.lengths = lengths.copy()
        self.n = len(lengths)
        self.a_cm = vector(a_cm, self.n, "a_cm")
        self.b_cm = vector(b_cm, self.n, "b_cm")
        r = np.asarray(resistance_body, dtype=float)
        if r.shape == (3, 3):
            r = np.broadcast_to(r, (self.n, 3, 3)).copy()
        if r.shape != (self.n, 3, 3) or not np.all(np.isfinite(r)):
            raise ValueError("resistance_body must have shape (N,3,3) or (3,3)")
        for item in r:
            if not np.allclose(item, item.T, rtol=1e-12, atol=1e-14):
                raise ValueError("Resistance must be symmetric")
            if np.linalg.eigvalsh(item)[0] < -1e-12*max(np.linalg.norm(item), 1e-300):
                raise ValueError("Resistance must be positive semidefinite")
        self.resistance = (r+r.transpose(0, 2, 1))/2
        self.strain_response = None
        if strain_response_body is not None:
            s = np.asarray(strain_response_body, dtype=float)
            if s.shape == (3, 2):
                s = np.broadcast_to(s, (self.n, 3, 2)).copy()
            if s.shape != (self.n, 3, 2) or not np.all(np.isfinite(s)):
                raise ValueError("strain_response_body must have shape (N,3,2) or (3,2)")
            self.strain_response = s.copy()

    @classmethod
    def isotropic(cls, lengths, a_cm, b_cm, gamma, eta,
                  *, strain_response_body=None):
        """Reduced force=-gamma*slip, torque=-eta*(omega-fluid_spin) model.

        gamma has units kg/s, eta kg*m^2/s. Neither is inferred from m or I.
        Equal x/y translational resistance is an explicit modeling assumption.
        """
        count = len(lengths)
        gamma, eta = vector(gamma, count, "gamma"), vector(eta, count, "eta")
        if np.any(gamma < 0) or np.any(eta < 0):
            raise ValueError("gamma and eta must be nonnegative")
        matrices = np.array([np.diag([g, g, e]) for g, e in zip(gamma, eta)])
        return cls(lengths, a_cm, b_cm, matrices, strain_response_body)

    def loads(self, theta, omega, *, time=0.0, flow=None,
              pivot_position=(0.0, 0.0), pivot_velocity=(0.0, 0.0)):
        theta, omega = vector(theta, self.n, "theta"), vector(omega, self.n, "omega")
        vp = xy(pivot_velocity, "pivot_velocity")
        kin = kinematics(self.lengths, self.a_cm, self.b_cm, theta, omega,
                         pivot_position=pivot_position, pivot_velocity=vp)
        field = AffineFlow() if flow is None else flow
        ambient = field.sample(kin.centers, time)
        if self.strain_response is None and np.max(np.abs(ambient.strain)) > 1e-14:
            raise ValueError("Straining flow needs explicit shape-dependent strain response; "
                             "pass zero matrices only as a deliberate zero-response assumption")
        forces = np.zeros((self.n, 2))
        tc = np.zeros(self.n)
        q = np.zeros(self.n)
        damping = np.zeros((self.n, self.n))
        forcing = np.zeros(self.n)
        relative_body = np.zeros((self.n, 3))
        resistance_power = strain_power = ambient_power = mechanical_power = 0.0
        for i in range(self.n):
            rotation, h = kin.body_to_world[i], kin.H[i]
            velocity = np.r_[kin.center_velocities[i]-ambient.velocity[i],
                              omega[i]-ambient.spin[i]]
            nu = rotation.T@velocity
            relative_body[i] = nu
            strain_body = rotation[:2, :2].T@ambient.strain[i]@rotation[:2, :2]
            epsilon = np.array([strain_body[0, 0], strain_body[0, 1]])
            strain_load = (np.zeros(3) if self.strain_response is None
                           else self.strain_response[i]@epsilon)
            load_world = rotation@(-self.resistance[i]@nu+strain_load)
            forces[i], tc[i] = load_world[:2], load_world[2]
            q += h.T@load_world
            resistance_world = rotation@self.resistance[i]@rotation.T
            damping += h.T@resistance_world@h
            flow_slip = np.r_[ambient.velocity[i]-vp, ambient.spin[i]]
            forcing += h.T@(resistance_world@flow_slip+rotation@strain_load)
            resistance_power += float(nu@self.resistance[i]@nu)
            strain_power += float(nu@strain_load)
            ambient_power += float(ambient.velocity[i]@forces[i]+ambient.spin[i]*tc[i])
            mechanical_power += float(kin.center_velocities[i]@forces[i]+omega[i]*tc[i])
        tp = tc+cross(kin.com_offsets, forces)
        return FluidLoads(forces, tc, tp, q, damping, forcing, relative_body,
                          resistance_power/2, resistance_power, strain_power,
                          ambient_power, mechanical_power)


def two_body_drag_matrix(theta, L1, a1, b1, a2, b2, gamma1, gamma2, eta1, eta2):
    """Closed reduced-model matrix, for any COM offsets; independent of mass."""
    delta = theta[0]-theta[1]
    coupling = gamma2*L1*(b2*np.cos(delta)+a2*np.sin(delta))
    return np.array([[gamma1*(a1*a1+b1*b1)+gamma2*L1*L1+eta1, coupling],
                     [coupling, gamma2*(a2*a2+b2*b2)+eta2]])


def two_body_mechanics(theta, omega, masses, inertias, lengths, a_cm, b_cm,
                        *, g=9.81, pivot_acceleration=(0.0, 0.0)):
    """Return M,h,G,P in M*theta_ddot+h+G+P=applied generalized torques."""
    q, w = vector(theta, 2, "theta"), vector(omega, 2, "omega")
    m, inertia = vector(masses, 2, "masses"), vector(inertias, 2, "inertias")
    l = vector(lengths, 2, "lengths")
    a, b = vector(a_cm, 2, "a_cm"), vector(b_cm, 2, "b_cm")
    if np.any(m <= 0) or np.any(inertia < 0) or np.any(l <= 0):
        raise ValueError("Masses and lengths must be positive; inertias nonnegative")
    ap = xy(pivot_acceleration, "pivot acceleration")
    delta = q[0]-q[1]
    coupling = m[1]*l[0]*(b[1]*np.cos(delta)+a[1]*np.sin(delta))
    mass = np.array([[inertia[0]+m[0]*(a[0]**2+b[0]**2)+m[1]*l[0]**2, coupling],
                     [coupling, inertia[1]+m[1]*(a[1]**2+b[1]**2)]])
    k = m[1]*l[0]*(b[1]*np.sin(delta)-a[1]*np.cos(delta))
    nonlinear = np.array([k*w[1]**2, -k*w[0]**2])
    gravity = g*np.array([m[0]*(a[0]*np.cos(q[0])+b[0]*np.sin(q[0]))
                          +m[1]*l[0]*np.sin(q[0]),
                          m[1]*(a[1]*np.cos(q[1])+b[1]*np.sin(q[1]))])
    kin = kinematics(l, a, b, q, w)
    pivot = sum((m[i]*kin.H[i, :2].T@ap for i in range(2)), start=np.zeros(2))
    return mass, nonlinear, gravity, pivot


def attach_to_repository(module, model, *, flow, pivot_position,
                         pivot_velocity):
    """Add general torques to the reviewed applied_torques(N,t,theta,omega) hook.

    All three external-motion/flow inputs are explicit. Position matters when
    fluid velocity varies in space. Call once before numerical integration.
    The returned prior hook can be restored after a run.
    """
    previous = module.applied_torques

    def combined(count, time, theta, omega):
        if count != model.n:
            raise ValueError("Body count differs between drag and pendulum models")
        rp = pivot_position(time) if callable(pivot_position) else pivot_position
        vp = pivot_velocity(time) if callable(pivot_velocity) else pivot_velocity
        extra = model.loads(theta, omega, time=time, flow=flow,
                            pivot_position=rp, pivot_velocity=vp).generalized_torques
        return vector(previous(count, time, theta, omega), count, "prior torque")+extra

    module.applied_torques = combined
    return previous
