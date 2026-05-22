.. _developers-diagnostics-internal:

Exposing Internal Computations for Diagnostics
===============================================

This guide shows how advanced users can expose internal intermediate calculations
(that are normally computed and discarded within kernels) for diagnostic output.

Overview
--------

WarpX computes many intermediate quantities during field solves that are not
normally saved. Examples include:

- Individual terms in Ohm's law (Hall term, pressure gradient, resistivity)
- Intermediate values in implicit solvers
- Terms in the PSATD solver
- Subcycling intermediate states

For debugging or analysis, you may want to output these quantities. This guide
shows the pattern for making them available to diagnostics.

The General Pattern
-------------------

**Step 1: Allocate a persistent MultiFab**

Allocate a named MultiFab from Python that will persist across timesteps and be
available for diagnostic output:

.. code-block:: python

   from pywarpx import callbacks
   
   @callbacks.installafterInitEsolve
   def allocate_diagnostic_fields():
       # Use an existing field as template
       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       
       # Allocate persistent diagnostic field
       hall_term = sim.fields.alloc_init(
           name="hall_term",
           dir='x',
           level=0,
           ba=Ex.box_array(),
           dm=Ex.dm(),
           ncomp=1,
           ngrow=Ex.n_grow_vect,
           initial_value=0.0,
           redistribute=True,
           redistribute_on_remake=True
       )
       
       # Add to diagnostic output
       sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=0)

**Step 2: Copy or compute values into the persistent MultiFab**

This can be done either from Python callbacks or by modifying C++ code.

**Python approach** (recompute in callback):

.. code-block:: python

   @callbacks.callfromafterstep
   def compute_diagnostic_fields():
       """Runs every step - compute derived quantities to store for next output"""

       # Get input fields
       Bx = sim.fields.get("Bfield_fp", dir='x', level=0)
       Jx = sim.fields.get("current_fp", dir='x', level=0)
       rho = sim.fields.get("rho_fp", level=0)

       # Get diagnostic field to write to
       hall_term = sim.fields.get("hall_term", dir='x', level=0)

       # Compute and store (simplified example)
       # Note: This uses global indexing for simplicity
       hall_term[...] = (Jx[...] * Bx[...]) / rho[...]

**C++ approach** (copy during kernel execution):

Modify the relevant C++ kernel to optionally store intermediate values when
the diagnostic field exists. See the complete example below.

**Step 3: Output the field**

The field will automatically be included in diagnostic output since it was
added with ``add_field_to_diagnostic()``.

.. note::

   **Alternative: PICMI ``data_list`` pass-through**

   If you prefer to declare all desired output fields up front, you can
   include your custom field name directly in the ``data_list`` of a
   :py:class:`~pywarpx.picmi.FieldDiagnostic`:

   .. code-block:: python

      diag = picmi.FieldDiagnostic(
          name="diag1",
          grid=grid,
          period=10,
          data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "custom_scalar", "custom_vector"],
          warpx_format="openpmd",
      )

   Names that are not recognised WarpX built-in fields are passed through to
   the C++ diagnostic layer.  The field **must exist** in the
   ``MultiFabRegister`` at the time the diagnostic is flushed (typically
   allocated in an ``afterInitEsolve`` or ``afterInitatRestart`` callback).
   This approach is equivalent to calling ``add_field_to_diagnostic()`` and
   produces identical output.

Complete Example: Hall Term in Ohm's Law
-----------------------------------------

This example shows how to expose the Hall term from the hybrid-PIC Ohm's law solver.

Option A: Pure Python Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recompute the Hall term in a Python callback:

.. code-block:: python

   from pywarpx import picmi, callbacks
   import numpy as np
   
   sim = picmi.Simulation(
       max_steps=100,
       warpx_hybrid_pic_model=True,
   )
   
   # Create diagnostic
   diag = picmi.FieldDiagnostic(
       name="diag1",
       period=10,
       data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
   )
   sim.add_diagnostic(diag)
   
   @callbacks.installafterInitEsolve
   def setup_hall_diagnostics():
       """Allocate fields for Hall term diagnostic"""
       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       
       # Allocate all three components
       for dir_str in ['x', 'y', 'z']:
           sim.fields.alloc_init(
               name="hall_term",
               dir=dir_str,
               level=0,
               ba=Ex.box_array(),
               dm=Ex.dm(),
               ncomp=1,
               ngrow=Ex.n_grow_vect,
               initial_value=0.0,
               redistribute=True,
               redistribute_on_remake=True
           )
       
       # Add to diagnostic output
       sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=0)
       print("Hall term diagnostics enabled")
   
   @callbacks.callfromafterstep
   def compute_hall_term():
       """Compute Hall term: (J - Ji) x B / (ne)"""
       from scipy.constants import elementary_charge as q_e
       
       level = 0
       
       # Get magnetic field
       Bx = sim.fields.get("Bfield_fp", dir='x', level=level)
       By = sim.fields.get("Bfield_fp", dir='y', level=level)
       Bz = sim.fields.get("Bfield_fp", dir='z', level=level)
       
       # Get plasma current (from Ampere's law)
       Jx = sim.fields.get("hybrid_current_fp_plasma", dir='x', level=level)
       Jy = sim.fields.get("hybrid_current_fp_plasma", dir='y', level=level)
       Jz = sim.fields.get("hybrid_current_fp_plasma", dir='z', level=level)
       
       # Get ion current
       Jix = sim.fields.get("current_fp", dir='x', level=level)
       Jiy = sim.fields.get("current_fp", dir='y', level=level)
       Jiz = sim.fields.get("current_fp", dir='z', level=level)
       
       # Get charge density
       rho = sim.fields.get("rho_fp", level=level)
       
       # Get data using global indexing
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
       
       # Compute electron current: Je = J - Ji
       jex = jx_data - jix_data
       jey = jy_data - jiy_data
       jez = jz_data - jiz_data
       
       # Cross product: Je x B
       je_cross_b_x = jey * bz_data - jez * by_data
       je_cross_b_y = jez * bx_data - jex * bz_data
       je_cross_b_z = jex * by_data - jey * bx_data
       
       # Electron density (quasi-neutrality: ne = |rho| / q_e)
       n_floor = 1e6  # Adjust based on your simulation
       ne = np.maximum(np.abs(rho_data) / q_e, n_floor)
       
       # Store Hall term = Je x B / (ne)
       hall_x = sim.fields.get("hall_term", dir='x', level=level)
       hall_y = sim.fields.get("hall_term", dir='y', level=level)
       hall_z = sim.fields.get("hall_term", dir='z', level=level)
       
       hall_x[...] = je_cross_b_x / ne
       hall_y[...] = je_cross_b_y / ne
       hall_z[...] = je_cross_b_z / ne
   
   sim.step()

**Pros:** No C++ changes needed, flexible, easy to prototype

**Cons:** Recomputes values (slower), requires understanding of field locations

Option B: C++ Modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For better performance, modify the C++ kernel to store intermediate values
when a diagnostic field is registered.

**Step 1:** In ``HybridPICSolveE.cpp``, check if diagnostic field exists and store values:

.. code-block:: cpp

   // In Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp
   
   void FiniteDifferenceSolver::HybridPICSolveECartesian (...) {
       // ... existing code ...
       
       // Check if diagnostic field was allocated from Python
       auto& warpx = WarpX::GetInstance();
       bool const store_hall_term = warpx.m_fields.has("hall_term", Direction::x, lev);
       
       Array4<Real> hall_x_diag, hall_y_diag, hall_z_diag;
       if (store_hall_term) {
           hall_x_diag = warpx.m_fields.get("hall_term", Direction::x, lev).array(mfi);
           hall_y_diag = warpx.m_fields.get("hall_term", Direction::y, lev).array(mfi);
           hall_z_diag = warpx.m_fields.get("hall_term", Direction::z, lev).array(mfi);
       }
       
       // In the kernel where Hall term is computed:
       amrex::ParallelFor(tex, tey, tez,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               // ... existing computation of enE (Hall term) ...
               auto const enE_x = (jey - jiy) * bz - (jez - jiz) * by;
               auto const hall_term_x = enE_x / ne;
               
               // Store for diagnostics if field was allocated
               if (store_hall_term) {
                   hall_x_diag(i, j, k) = hall_term_x;
               }
               
               // Continue with E-field update (existing code)
               Ex(i, j, k) = hall_term_x + /* other terms */;
           },
           // ... similar for y and z ...
       );
   }

**Step 2:** From Python, allocate the field as in Option A. The C++ code will
automatically populate it.

**Pros:** Minimal performance overhead, values computed only once

**Cons:** Requires C++ changes and rebuilding WarpX

When to Use Each Approach
--------------------------

**Python-only** (Option A) recommended for:

- Quick debugging and exploration
- Prototyping new diagnostics
- Infrequent output (diagnostic overhead is acceptable)
- When you cannot rebuild WarpX

**C++ modifications** (Option B) recommended for:

- Frequent diagnostic output
- Production runs where performance matters
- Fields computed deep in GPU kernels
- Values that are expensive to recompute

Additional Examples
-------------------

**Example: Copy an internal temporary field**

.. code-block:: python

   @callbacks.installafterInitEsolve  
   def setup_temp_field_diagnostic():
       """Copy a temporary internal field to persistent diagnostic field"""
       
       # Get internal temporary field as template
       temp = sim.fields.get("hybrid_rho_fp_temp", level=0)
       
       # Allocate persistent copy
       diag_copy = sim.fields.alloc_init(
           name="rho_temp_diagnostic",
           level=0,
           ba=temp.box_array(),
           dm=temp.dm(),
           ncomp=1,
           ngrow=temp.n_grow_vect,
           redistribute=True,
           redistribute_on_remake=True
       )
       
       sim.extension.warpx.add_field_to_diagnostic("diag1", "rho_temp_diagnostic")
   
   @callbacks.callfromafterstep
   def copy_temp_field():
       """Copy temporary field data to diagnostic field"""
       temp = sim.fields.get("hybrid_rho_fp_temp", level=0)
       diag = sim.fields.get("rho_temp_diagnostic", level=0)
       
       # Copy using ParallelCopy or global indexing
       diag[...] = temp[...]

**Example: Compute multiple derived fields efficiently**

.. code-block:: python

   @callbacks.callfromafterstep
   def compute_all_diagnostics():
       """Compute multiple derived fields at once"""
       
       # Get all needed inputs once
       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       Ey = sim.fields.get("Efield_fp", dir='y', level=0)
       Ez = sim.fields.get("Efield_fp", dir='z', level=0)
       
       # Compute E magnitude
       e_mag = sim.fields.get("E_magnitude", level=0)
       e_mag[...] = np.sqrt(Ex[...]**2 + Ey[...]**2 + Ez[...]**2)
       
       # Compute E parallel to B (requires B field too)
       # ... more derived quantities ...

Best Practices
--------------

1. **Naming convention:** Use descriptive names like ``hall_term``, ``pressure_grad``, ``E_magnitude``

2. **Performance:** Only compute diagnostics when needed:

   - Use ``callfromafterstep`` to update diagnostic fields each step so they
     are ready when the diagnostic system writes them (the diagnostic period
     filter is applied by the flush mechanism, not by your callback)
   - For expensive recomputations you can check the step number against the
     diagnostic period manually inside your callback

3. **Memory:** Diagnostic fields consume memory - only allocate what you need

4. **Documentation:** Comment your code explaining what each diagnostic represents

5. **Validation:** Compare computed values with known solutions or conservation laws

See Also
--------

- :ref:`usage-python-extend` - Python field access and manipulation
- :ref:`developers-fields` - Internal field structure
- :ref:`developers-diagnostics` - Diagnostic system overview
