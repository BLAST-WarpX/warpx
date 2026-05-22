#!/usr/bin/env python3
"""
Example: Computing and outputting Ohm's law terms for hybrid-PIC simulations

This script demonstrates how to compute individual terms from Ohm's law:
  E = (J - Ji) x B / (ne) - grad(Pe)/(ne) + eta*J - eta_h*nabla^2(J)
    = [Hall term] + [Pressure term] + [Resistivity] + [Hyper-resistivity]

And add them to diagnostic output using the new dynamic field registration.
"""

from pywarpx import picmi, callbacks
import numpy as np

# ============================================================================
# Simulation setup (simplified - adjust for your actual simulation)
# ============================================================================

sim = picmi.Simulation(
    max_steps=100,
    warpx_hybrid_pic_model=True,  # Enable hybrid-PIC
    # ... other parameters ...
)

# Create field diagnostic that will include Ohm's law terms
diag = picmi.FieldDiagnostic(
    name="diag1",
    period=10,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", 
               "Jx", "Jy", "Jz", "rho", "electron_pressure"],
)
sim.add_diagnostic(diag)

# ============================================================================
# Helper function to compute Ohm's law terms
# ============================================================================

def compute_ohm_law_terms():
    """
    Compute and store individual Ohm's law terms.
    
    This function:
    1. Gets the required fields (E, B, J, Ji, rho, Pe)
    2. Computes the Hall term: (J - Ji) x B / (ne)
    3. Computes the pressure gradient term: -grad(Pe) / (ne)
    4. (Optional) Resistivity and hyper-resistivity terms
    5. Stores results in custom MultiFabs for diagnostic output
    """
    
    level = 0
    
    # Get existing fields
    Ex = sim.fields.get("Efield_fp", dir='x', level=level)
    Ey = sim.fields.get("Efield_fp", dir='y', level=level)
    Ez = sim.fields.get("Efield_fp", dir='z', level=level)
    
    Bx = sim.fields.get("Bfield_fp", dir='x', level=level)
    By = sim.fields.get("Bfield_fp", dir='y', level=level)
    Bz = sim.fields.get("Bfield_fp", dir='z', level=level)
    
    # Total current (from Ampere's law: J = curl B / mu0 - J_ext)
    Jx = sim.fields.get("hybrid_current_fp_plasma", dir='x', level=level)
    Jy = sim.fields.get("hybrid_current_fp_plasma", dir='y', level=level)
    Jz = sim.fields.get("hybrid_current_fp_plasma", dir='z', level=level)
    
    # Ion current (deposited from ion particles)
    Jix = sim.fields.get("current_fp", dir='x', level=level)
    Jiy = sim.fields.get("current_fp", dir='y', level=level)
    Jiz = sim.fields.get("current_fp", dir='z', level=level)
    
    # Charge density and electron pressure
    rho = sim.fields.get("rho_fp", level=level)
    Pe = sim.fields.get("hybrid_electron_pressure_fp", level=level)
    
    # Get or create custom fields for Ohm's law terms
    # Note: These should have been allocated in afterInitEsolve callback
    hall_x = sim.fields.get("hall_term", dir='x', level=level)
    hall_y = sim.fields.get("hall_term", dir='y', level=level)
    hall_z = sim.fields.get("hall_term", dir='z', level=level)
    
    pressure_x = sim.fields.get("pressure_term", dir='x', level=level)
    pressure_y = sim.fields.get("pressure_term", dir='y', level=level)
    pressure_z = sim.fields.get("pressure_term", dir='z', level=level)
    
    # ========================================================================
    # Compute Hall term: (J - Ji) x B / (ne)
    # ========================================================================
    # Note: This is a simplified version. In reality, you need to:
    # 1. Interpolate fields to a common staggering (nodal)
    # 2. Handle the division by charge density carefully (floor values)
    # 3. Account for the electron charge (ne = -rho/q_e for quasi-neutrality)
    
    from scipy.constants import elementary_charge as q_e
    
    # Simple version using global indexing (copies data)
    # For production, iterate over MFIter for better performance
    
    # Get all data (this returns numpy arrays via global indexing)
    bx_data = Bx[...]
    by_data = By[...]
    bz_data = Bz[...]
    
    jx_data = Jx[...]
    jy_data = Jy[...]
    jz_data = Jz[...]
    
    jix_data = Jix[...]
    jiy_data = Jiy[...]
    jiz_data = Jiz[...]
    
    rho_data = rho[...]
    
    # Electron current: Je = J - Ji
    jex = jx_data - jix_data
    jey = jy_data - jiy_data
    jez = jz_data - jiz_data
    
    # Cross product: Je x B
    je_cross_b_x = jey * bz_data - jez * by_data
    je_cross_b_y = jez * bx_data - jex * bz_data
    je_cross_b_z = jex * by_data - jey * bx_data
    
    # Electron density (assuming quasi-neutrality: ne = rho / q_e)
    # Apply floor to avoid division by zero
    n_floor = 1e6  # Adjust based on your simulation
    ne = np.maximum(np.abs(rho_data) / q_e, n_floor)
    
    # Hall term = (J - Ji) x B / (ne)
    hall_term_x = je_cross_b_x / ne
    hall_term_y = je_cross_b_y / ne
    hall_term_z = je_cross_b_z / ne
    
    # Store in custom MultiFabs
    hall_x[...] = hall_term_x
    hall_y[...] = hall_term_y
    hall_z[...] = hall_term_z
    
    # ========================================================================
    # Compute pressure gradient term: -grad(Pe) / (ne)
    # ========================================================================
    # This requires computing finite differences
    # Simplified version - in production, use proper stencils matching WarpX
    
    pe_data = Pe[...]
    
    # Compute gradients (2nd order central differences)
    # Note: You need to handle boundaries properly
    grad_pe_x = np.gradient(pe_data, axis=0)  # Simplified
    grad_pe_y = np.gradient(pe_data, axis=1)
    grad_pe_z = np.gradient(pe_data, axis=2)
    
    # Pressure term = -grad(Pe) / (ne)
    pressure_term_x = -grad_pe_x / ne
    pressure_term_y = -grad_pe_y / ne
    pressure_term_z = -grad_pe_z / ne
    
    # Store in custom MultiFabs
    pressure_x[...] = pressure_term_x
    pressure_y[...] = pressure_term_y
    pressure_z[...] = pressure_term_z
    
    # ========================================================================
    # Optional: Compute resistivity term (eta * J) and hyper-resistivity
    # ========================================================================
    # These require access to the resistivity parameters from HybridPICModel
    # and additional field operations (Laplacian for hyper-resistivity)
    
    print(f"Computed Ohm's law terms at step {sim.extension.warpx.getistep(0)}")


# ============================================================================
# Callback setup
# ============================================================================

@callbacks.installafterInitEsolve
def setup_ohm_law_fields():
    """
    Allocate custom MultiFabs for storing Ohm's law terms.
    These will be added to the diagnostic output.
    """
    level = 0
    
    # Use E-field as template for staggering and grid properties
    Ex = sim.fields.get("Efield_fp", dir='x', level=level)
    
    # Allocate Hall term components
    for dir_str in ['x', 'y', 'z']:
        sim.fields.alloc_init(
            name="hall_term",
            dir=dir_str,
            level=level,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=0.0,
            redistribute=True,
            redistribute_on_remake=True
        )
        
        sim.fields.alloc_init(
            name="pressure_term",
            dir=dir_str,
            level=level,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=0.0,
            redistribute=True,
            redistribute_on_remake=True
        )
    
    # Add these fields to diagnostic output
    sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=level)
    sim.extension.warpx.add_field_to_diagnostic("diag1", "pressure_term", lev=level)
    
    print("Allocated Ohm's law diagnostic fields")


@callbacks.callfromafterstep
def update_ohm_law_terms():
    """
    Callback that runs after each time step.
    Recomputes the Ohm's law terms so they're up-to-date in the output.
    """
    compute_ohm_law_terms()


# ============================================================================
# Run simulation
# ============================================================================

sim.step()

print("Simulation complete. Ohm's law terms written to diagnostic output.")
print("Output fields include:")
print("  - hall_term_x, hall_term_y, hall_term_z")
print("  - pressure_term_x, pressure_term_y, pressure_term_z")
