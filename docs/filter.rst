###################################################
Filtering
###################################################

=====================================
Applying highpass / lowpass filters
=====================================

.. plot:: ../examples/filter/pass.py
   :include-source:

=====================================
Applying an FIR filter
=====================================

.. plot:: ../examples/filter/fir.py
   :include-source:

=====================================
Matched Filter SNR
=====================================

.. plot:: ../examples/filter/snr.py
   :include-source:

=====================================
Chisq time series
=====================================

.. plot:: ../examples/filter/chisq.py
   :include-source:

==============================================================
Faithfulness with higher harmonics, averaged over the sky
==============================================================

:func:`pycbc.filter.faithfulness.sky_averaged_unfaithfulness` computes the
SNR-weighted, sky-and-polarization-averaged unfaithfulness of Ramos-Buades et
al. 2023 (arXiv:2303.18046, Eqs. 30-36) between a signal and a template given
as spherical-harmonic modes: the faithfulness is maximised over the template's
coalescence time and phase numerically and its effective polarization
analytically, then averaged over the source's inclination, phase and effective
polarization with SNR^3 weights. :func:`pycbc.filter.faithfulness.sky_averaged_unfaithfulness_map`
maps it over many binaries in parallel.

.. literalinclude:: ../examples/filter/faithfulness.py
   :language: python
