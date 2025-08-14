Welcome to shoeboxpy's documentation!
=====================================

`shoeboxpy` provides lightweight 3-DOF and 6-DOF rigid-body vessel models ("shoebox" approximation) with added mass, Coriolis, linear damping and simple hydrostatic restoring. It is intended for quick prototyping, teaching and algorithm development where a full CFD model would be overkill.

.. note::
   The models use small-angle roll/pitch restoring and diagonal added mass/damping approximations. They are NOT a substitute for high-fidelity design / certification analyses.

Getting started
---------------
If this is your first time, read :doc:`getting_started` for installation, a minimal example and common tasks. For the underlying equations, see :doc:`theory`.

API overview
------------
The two core classes are :class:`shoeboxpy.model3dof.Shoebox` and :class:`shoeboxpy.model6dof.Shoebox`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   theory
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
