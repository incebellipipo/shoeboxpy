Getting Started
===============

Installation
------------

Install the latest development version directly from GitHub::

   pip install git+https://github.com/incebellipipo/shoeboxpy.git

Or (after cloning) install in editable mode::

   pip install -e .

Quick Example (6-DOF)
---------------------

.. code-block:: python

   import numpy as np
   from shoeboxpy.model6dof import Shoebox

   box = Shoebox(L=1.0, B=0.3, T=0.03,
                 eta0=np.array([0,0,0, 0.05,0.05,0.05]),
                 GM_phi=0.2, GM_theta=0.2)

   dt = 0.01
   tau = np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.1])
   for _ in range(1000):  # 10 seconds
       box.step(tau=tau, dt=dt)

   eta, nu = box.get_states()
   print("Final pose:", eta)

3-DOF Variant
-------------

.. code-block:: python

   from shoeboxpy.model3dof import Shoebox as Shoebox3
   b3 = Shoebox3(L=1.0, B=0.3, T=0.03)
   for _ in range(1000):
       b3.step(tau=np.array([1.0, 0.0, 0.05]), dt=0.01)
   print(b3.get_states())

Choosing Parameters
-------------------

* Added mass coefficients (alpha_*) are dimensionless multipliers on rigid-body mass/inertia. Start with 0.05–0.2.
* Damping factors (beta_*) are linear; increase to reduce oscillations.
* GM_phi / GM_theta govern roll and pitch stiffness; too large => stiff dynamics.

What Next?
----------

* Read :doc:`theory` for derivations.
* Browse :doc:`modules` for full API.
* Use ``shoeboxpy.animate.animate_history`` to visualize 6-DOF trajectories.

FAQ
---

**Why diagonal added mass?**  Simplicity and speed; you can extend to a full matrix if needed.

**Can I add nonlinear damping?**  Yes; subclass and override ``step``/``_dynamics_into`` inserting extra forces.

**Stability issues?**  Reduce dt or increase damping / metacentric heights.
