"""Fluid loads expressed in Physics_88___Proposal's l_ni,theta_ni notation.

This is an independent explicit implementation of the reduced linear COM
resistance model. Bodies are not assumed to be rods. R_cm is a COM distance,
not a body radius or Rayleigh dissipation function.
"""

from dataclasses import dataclass
import numpy as np


def _vector(value,n,name):
    try:
        a=np.broadcast_to(np.asarray(value,dtype=float),(n,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or length {n}") from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


@dataclass(frozen=True)
class PaperGeometry:
    l_ni: np.ndarray
    theta_ni: np.ndarray
    jacobians: np.ndarray
    centers: np.ndarray


def geometry(lengths,R_cm,phi,theta,pivot_position=(0.0,0.0)):
    """Proposal Eqs. (35)-(36), using constant COM offsets and atan2 angles."""
    lengths=np.asarray(lengths,dtype=float)
    if lengths.ndim!=1 or len(lengths)==0 or np.any(lengths<=0) or not np.all(np.isfinite(lengths)):
        raise ValueError("lengths must be a nonempty positive finite vector")
    n=len(lengths)
    radius=_vector(R_cm,n,'R_cm'); phi=_vector(phi,n,'phi'); theta=_vector(theta,n,'theta')
    if np.any(radius<0):
        raise ValueError("COM distances must be nonnegative")
    rp=np.asarray(pivot_position,dtype=float)
    if rp.shape!=(2,) or not np.all(np.isfinite(rp)):
        raise ValueError("pivot_position must be a finite 2-vector")
    ell=np.zeros((n,n)); angle=np.zeros((n,n))
    for body in range(n):
        ell[body,:body]=lengths[:body]
        ell[body,body]=radius[body]
        angle[body,:body]=theta[:body]
        angle[body,body]=theta[body]+phi[body]
    jacobian=np.stack((ell*np.cos(angle),ell*np.sin(angle)),axis=1)
    centers=rp+np.column_stack((np.sum(ell*np.sin(angle),axis=1),
                                -np.sum(ell*np.cos(angle),axis=1)))
    return PaperGeometry(ell,angle,jacobian,centers)


@dataclass(frozen=True)
class PaperFluidLoads:
    damping_matrix: np.ndarray
    fluid_forcing: np.ndarray
    generalized_torques: np.ndarray
    forces: np.ndarray
    torques_about_com: np.ndarray
    center_velocities: np.ndarray
    geometry: PaperGeometry


def linear_fluid_loads(lengths,R_cm,phi,theta,omega,c_t,c_r,*,
                        pivot_position=(0.0,0.0),pivot_velocity=(0.0,0.0),
                        fluid_velocity=(0.0,0.0),fluid_spin=0.0):
    """Return tau^f=-D^f*omega+h^f in the proposal's full torque equations.

    c_t: kg/s; c_r: kg*m^2/s. fluid_velocity is a uniform (2,) vector or
    (N,2) values evaluated at COMs. fluid_spin is HALF fluid vorticity.
    This reduced model deliberately has isotropic translation and no
    translation-rotation or strain-response coupling. Use GeneralFluidModel
    for the full shape-parameterized affine-flow law.

    For the earlier row-normalized point-mass equations ONLY, divide returned
    generalized_torques by lengths. The final physical-body equation in the
    proposal already has torque units and takes the result directly.
    """
    geom=geometry(lengths,R_cm,phi,theta,pivot_position)
    n=len(lengths)
    omega=_vector(omega,n,'omega'); ct=_vector(c_t,n,'c_t'); cr=_vector(c_r,n,'c_r')
    if np.any(ct<0) or np.any(cr<0):
        raise ValueError("Resistances must be nonnegative")
    vp=np.asarray(pivot_velocity,dtype=float)
    u=np.asarray(fluid_velocity,dtype=float)
    if vp.shape!=(2,) or not np.all(np.isfinite(vp)):
        raise ValueError("pivot_velocity must be a finite 2-vector")
    if u.shape==(2,): u=np.broadcast_to(u,(n,2))
    if u.shape!=(n,2) or not np.all(np.isfinite(u)):
        raise ValueError("fluid_velocity must have shape (2,) or (N,2)")
    spin=_vector(fluid_spin,n,'fluid_spin')
    J=geom.jacobians
    velocity=vp+np.einsum('naj,j->na',J,omega)
    force=-ct[:,None]*(velocity-u)
    torque_com=-cr*(omega-spin)
    damping=np.einsum('naj,n,nak->jk',J,ct,J)+np.diag(cr)
    forcing=np.einsum('naj,n,na->j',J,ct,u-vp)+cr*spin
    torque=np.einsum('naj,na->j',J,force)+torque_com
    return PaperFluidLoads(damping,forcing,torque,force,torque_com,velocity,geom)


def physical_matrices(lengths,R_cm,phi,theta,masses,I_cm,*,
                       g=9.81,pivot_acceleration=(0.0,0.0)):
    """Return M,B,G in the proposal's final equation M*alpha+B*omega^2+G=tau.

    These matrices are used for alignment verification. c_t,c_r above are
    independent inputs and are not derived from masses or central inertias.
    """
    geom=geometry(lengths,R_cm,phi,theta)
    n=len(lengths)
    masses=_vector(masses,n,'masses'); inertia=_vector(I_cm,n,'I_cm')
    ap=np.asarray(pivot_acceleration,dtype=float)
    if ap.shape!=(2,) or not np.all(np.isfinite(ap)):
        raise ValueError("pivot_acceleration must be a finite 2-vector")
    if np.any(masses<=0) or np.any(inertia<0):
        raise ValueError("Masses must be positive; inertias nonnegative")
    M=np.diag(inertia); B=np.zeros((n,n)); G=np.zeros(n)
    for k in range(n):
        for j in range(n):
            for body in range(max(k,j),n):
                pref=masses[body]*geom.l_ni[body,k]*geom.l_ni[body,j]
                difference=geom.theta_ni[body,k]-geom.theta_ni[body,j]
                M[k,j]+=pref*np.cos(difference)
                B[k,j]+=pref*np.sin(difference)
        for body in range(k,n):
            t=geom.theta_ni[body,k]
            G[k]+=masses[body]*geom.l_ni[body,k]*(
                ap[0]*np.cos(t)+(g+ap[1])*np.sin(t))
    return M,B,G
