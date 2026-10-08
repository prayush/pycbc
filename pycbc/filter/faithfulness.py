# Copyright (C) 2026 Prayush Kumar
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Faithfulness of waveform models with higher harmonics, maximised over the
template's extrinsic parameters and averaged over the source's orientation,
sky position and polarization, weighted by the signal's SNR.

The quantities are those of Ramos-Buades et al. 2023, "SEOBNRv5PHM: Next
generation of accurate and efficient multipolar precessing-spin
effective-one-body waveforms for binary black holes", Phys. Rev. D 108,
124037 (arXiv:2303.18046), Sec. IV A, in their notation.

Detector strain, Eq. (30), with the effective polarization angle kappa of
Capano, Pan & Buonanno 2014 (arXiv:1311.1286), which absorbs the sky position
and the polarization angle; the overall amplitude A(theta, phi) cancels in
every normalised quantity below:

.. math::

    h(t) = \\mathcal{A}(\\theta, \\phi)\\,(h_+ \\cos\\kappa + h_\\times \\sin\\kappa)

Polarizations from the modes, Eq. (31), with pycbc's (LAL's) harmonics, as
:func:`pycbc.waveform.waveform_modes.sum_modes` builds them:

.. math::

    h_+ - i h_\\times = \\sum_{\\ell, m} {}_{-2}Y_{\\ell m}(\\iota, \\varphi_0)\\, h_{\\ell m}(t)

Inner product, Eq. (32):

.. math::

    \\langle h_1, h_2 \\rangle = 4\\,\\mathrm{Re} \\int_{f_\\mathrm{in}}^{f_\\mathrm{max}}
        df\\, \\frac{\\tilde h_1(f)\\, \\tilde h_2^*(f)}{S_n(f)}

Faithfulness, Eq. (33): the normalised overlap, maximised over the
template's coalescence time :math:`t_c` and azimuthal phase
:math:`\\varphi_{0t}` (numerically) and its effective polarization
:math:`\\kappa_t` (analytically), with the template's inclination equal to the
source's and the intrinsic parameters equal:

.. math::

    \\mathcal{F}(M_s, \\iota_s, \\varphi_{0s}, \\kappa_s) = \\max_{t_c, \\varphi_{0t}, \\kappa_t}
        \\left[ \\frac{\\langle h_s | h_t \\rangle}
        {\\sqrt{\\langle h_s | h_s \\rangle \\langle h_t | h_t \\rangle}}
        \\right]_{\\iota_t = \\iota_s}

SNR-weighted average over the source's orientation, Eq. (36) -- the cube root
of the ratio, as the paper writes it -- with :math:`\\mathrm{SNR} =
\\sqrt{\\langle h_s, h_s \\rangle}` the *signal's* SNR at that orientation:

.. math::

    \\bar{\\mathcal{F}}_\\mathrm{SNR}(M_s) = \\left(
        \\frac{\\int_{-1}^{1} d\\cos\\iota_s \\int_0^{2\\pi} d\\kappa_s
               \\int_0^{2\\pi} d\\varphi_{0s}\\; \\mathcal{F}^3\\, \\mathrm{SNR}^3}
              {\\int_{-1}^{1} d\\cos\\iota_s \\int_0^{2\\pi} d\\kappa_s
               \\int_0^{2\\pi} d\\varphi_{0s}\\; \\mathrm{SNR}^3}
    \\right)^{1/3},
    \\qquad
    \\bar{\\mathcal{M}}_\\mathrm{SNR} = 1 - \\bar{\\mathcal{F}}_\\mathrm{SNR}.

:func:`sky_averaged_unfaithfulness` returns
:math:`\\bar{\\mathcal{M}}_\\mathrm{SNR}`.

The maximisation over :math:`\\kappa_t` is the closed form of Harry, Calderon
Bustillo & Nitz 2017 (arXiv:1709.09181), the statistic of
:func:`pycbc.filter.compute_max_snr_over_sky_loc_stat_no_phase`: with
:math:`\\hat P`, :math:`\\hat C` the overlaps of the signal with the normalised
plus and cross templates and :math:`I = \\langle \\hat h_+, \\hat h_\\times
\\rangle`,

.. math::

    \\max_{\\kappa_t} \\frac{\\langle h_s | h_t \\rangle}{\\sqrt{\\langle h_t | h_t \\rangle}}
    = \\sqrt{\\frac{\\hat P^2 + \\hat C^2 - 2 \\hat P \\hat C I}{1 - I^2}} .

No maximisation over an overall phase of the template is made:
:math:`\\varphi_{0t}` is explicit and is maximised numerically, as Eq. (33)
requires; with higher harmonics a constant phase rotation of the whole
template is not a change of any physical parameter.

How it is computed
------------------
Every polarization at every :math:`(\\iota, \\varphi_0, \\kappa)` is a real
linear combination of the real and imaginary parts of the modes. The
correlation series between each component of the signal and each component of
the template are computed once, by FFT, oversampled in lag; every source
orientation, template phase and template polarization is then a linear
combination of those series. The coalescence time is searched on the
oversampled lag grid, within a window around the best lag of a pre-pass,
and refined by a parabola; :math:`\\varphi_{0t}` on a grid, refined by a
golden-section search. For aligned spins (``aligned_spin=True``) the modes
with :math:`m < 0` follow from :math:`h_{\\ell,-m} = (-1)^\\ell h^*_{\\ell m}`,
which halves the number of components, and the inclination integral is taken
over :math:`[0, \\pi/2]` only: the source at :math:`(\\pi - \\iota, \\varphi_0,
\\kappa)` is the source at :math:`(\\iota, \\varphi_0, -\\kappa)`, and the
template's plus/cross span is unchanged, so the integrand averaged over
:math:`\\kappa` is symmetric.

This module assumes the template's intrinsic parameters are the signal's and
does not optimise over the in-plane spin rotation of Eq. (34), which matters
only for precessing systems.
"""

import logging
import math

import numpy
from scipy import fft as sp_fft
from scipy.special import roots_legendre

from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.waveform.waveform_modes import get_glm

logger = logging.getLogger('pycbc.filter.faithfulness')

__all__ = ['sky_averaged_unfaithfulness', 'faithfulness_at_orientation',
           'max_over_polarization', 'snr_weighted_average',
           'nr_low_frequency_cutoff', 'sky_averaged_unfaithfulness_map']

_INV_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


# ------------------------------------------------------------------ pieces
def max_over_polarization(P, C, I):
    """The overlap maximised over the template's effective polarization.

    For a template :math:`h_t = \\cos\\kappa_t\\, h_+ + \\sin\\kappa_t\\, h_\\times`
    and real overlaps :math:`\\hat P = \\langle s, \\hat h_+ \\rangle`,
    :math:`\\hat C = \\langle s, \\hat h_\\times \\rangle` of the data with the
    normalised plus and cross templates, whose own overlap is :math:`I`,

    .. math::

        \\max_{\\kappa_t} \\frac{\\langle s, h_t \\rangle}{\\|h_t\\|}
        = \\sqrt{\\frac{\\hat P^2 + \\hat C^2 - 2 \\hat P \\hat C I}{1 - I^2}},

    the statistic of :func:`pycbc.filter.compute_max_snr_over_sky_loc_stat_no_phase`
    (arXiv:1709.09181), vectorised over any shape. It is the norm of the
    projection of the data onto the span of :math:`h_+, h_\\times`. The
    maximising angle is returned too, for the *normalised* basis: the optimum
    is :math:`\\cos\\kappa\\, \\hat h_+ + \\sin\\kappa\\, \\hat h_\\times`.

    Parameters
    ----------
    P, C : array_like
        Real overlaps with the normalised plus and cross templates.
    I : array_like
        Real overlap of the normalised plus and cross templates,
        broadcastable against ``P``; ``|I| < 1``.

    Returns
    -------
    rho : numpy.ndarray
        The maximised (non-negative) normalised overlap.
    kappa : numpy.ndarray
        The maximising effective polarization angle, in [0, 2 pi).
    """
    P = numpy.asarray(P, dtype=float)
    C = numpy.asarray(C, dtype=float)
    I = numpy.asarray(I, dtype=float)
    if numpy.any(numpy.abs(I) >= 1.0):
        raise ValueError("the plus and cross templates are parallel (|I| >= 1)")
    den = 1.0 - I * I
    rho2 = (P * P + C * C - 2.0 * P * C * I) / den
    rho = numpy.sqrt(numpy.maximum(rho2, 0.0))
    # the maximising direction: (alpha, beta) proportional to M^-1 (P, C) in
    # the normalised basis, with M = [[1, I], [I, 1]]
    alpha = (P - I * C) / den
    beta = (C - I * P) / den
    kappa = numpy.mod(numpy.arctan2(beta, alpha), 2.0 * numpy.pi)
    return rho, kappa


def snr_weighted_average(faithfulness, snr, weights=None):
    """Eq. (36): the cube root of the SNR^3-weighted mean of F^3.

    Parameters
    ----------
    faithfulness, snr : array_like
        Per source orientation, of equal shape. ``snr`` is the signal's.
    weights : array_like, optional
        Quadrature weights of the orientations (default: equal).

    Returns
    -------
    float
        :math:`\\bar{\\mathcal{F}}_\\mathrm{SNR}`.
    """
    F = numpy.asarray(faithfulness, dtype=float).ravel()
    s3 = numpy.asarray(snr, dtype=float).ravel() ** 3
    w = (numpy.ones_like(F) if weights is None
         else numpy.broadcast_to(numpy.asarray(weights, dtype=float),
                                 numpy.shape(faithfulness)).ravel())
    num = numpy.sum(w * F ** 3 * s3)
    den = numpy.sum(w * s3)
    if not den > 0:
        raise ValueError("the signal has no SNR at any orientation")
    return float(numpy.cbrt(num / den))


def nr_low_frequency_cutoff(hlm, delta_t=None, factor=1.35):
    """The starting frequency the paper uses for a short (e.g. NR) signal.

    "For cases where [both waveforms are not in band] (e.g., the NR waveforms
    are used), we employ f_in = 1.35 f_peak, where f_peak corresponds to the
    peak amplitude of the frequency-domain strain of the signal, and the factor
    1.35 accounts for possible artifacts coming from the Fourier transform of
    the time domain waveforms" (arXiv:2303.18046, Sec. IV A).

    Parameters
    ----------
    hlm : complex TimeSeries or array
        The signal's dominant mode, normally the (2, 2); its real part (the
        face-on plus polarization of that mode, up to a constant) is
        transformed.
    delta_t : float, optional
        Required if ``hlm`` is an array.
    factor : float
        The paper's 1.35.

    Returns
    -------
    float
        ``factor`` times the frequency of the peak of ``|FFT(Re hlm)|``.
    """
    if isinstance(hlm, TimeSeries):
        delta_t = float(hlm.delta_t)
        h = numpy.asarray(hlm.numpy())
    else:
        if delta_t is None:
            raise ValueError("delta_t is required for an array")
        h = numpy.asarray(hlm)
    x = numpy.real(h).astype(float)
    n = len(x)
    amp = numpy.abs(numpy.fft.rfft(x))
    f = numpy.fft.rfftfreq(n, d=delta_t)
    k = int(numpy.argmax(amp[1:])) + 1
    return float(factor * f[k])


# ------------------------------------------------------------------- basis
class _ModeBasis:
    """The real component series a mode set is linear in, and the
    coefficients that build h_+ and h_x from them at (iota, phi0).

    Unfolded, every mode (l, m) contributes Re h_lm and Im h_lm. Folded
    (aligned spins), a pair (l, +-m) contributes Re h_lm and Im h_lm of the
    m > 0 member only, the m < 0 member following from
    h_{l,-m} = (-1)^l conj(h_lm).
    """

    def __init__(self, modes, aligned_spin, label):
        entries = []
        if aligned_spin:
            keys = set(modes)
            for (l, m), h in sorted(modes.items()):
                if m > 0:
                    if (l, -m) in keys:
                        partner = numpy.asarray(modes[(l, -m)])
                        expect = (-1) ** l * numpy.conj(h)
                        scale = max(numpy.max(numpy.abs(h)), 1e-300)
                        if (len(partner) != len(h) or
                                numpy.max(numpy.abs(partner - expect)) > 1e-8 * scale):
                            raise ValueError(
                                f"{label} mode ({l}, {-m}) is not (-1)^l conj of "
                                f"({l}, {m}): these modes are not aligned-spin; "
                                f"use aligned_spin=False")
                    entries.append((l, m, True, numpy.asarray(h)))
                elif m == 0:
                    entries.append((l, 0, False, numpy.asarray(h)))
                elif (l, -m) not in keys:
                    # only the m < 0 member given: carry its m > 0 partner
                    entries.append((l, -m, True, (-1) ** l * numpy.conj(numpy.asarray(h))))
        else:
            for (l, m), h in sorted(modes.items()):
                entries.append((l, m, False, numpy.asarray(h)))
        if not entries:
            raise ValueError(f"no {label} modes")
        self.entries = [(l, m, folded) for l, m, folded, _ in entries]
        self.series = [h for _, _, _, h in entries]
        self.n = 2 * len(entries)

    def components(self, length):
        """The real component series, zero-padded to `length`, as rows."""
        out = numpy.zeros((self.n, length))
        for i, h in enumerate(self.series):
            out[2 * i, :len(h)] = numpy.real(h)
            out[2 * i + 1, :len(h)] = numpy.imag(h)
        return out

    def polarization_coefficients(self, iota, phi):
        """`(d_plus, d_cross)`, each of shape `phi.shape + (n,)`: at inclination
        `iota` (scalar) and azimuth(s) `phi`, h_+ = d_plus . e and
        h_x = d_cross . e, with e the component series."""
        phi = numpy.asarray(phi, dtype=float)
        dp = numpy.zeros(phi.shape + (self.n,))
        dc = numpy.zeros(phi.shape + (self.n,))
        for i, (l, m, folded) in enumerate(self.entries):
            c, s = numpy.cos(m * phi), numpy.sin(m * phi)
            y = get_glm(l, m, iota)
            if folded:
                ym = (-1) ** l * get_glm(l, -m, iota)
                yp, yx = y + ym, -y + ym
                # h_+ : (y + (-1)^l y_{l,-m}) (cos(m phi) a - sin(m phi) b)
                # h_x : (-y + (-1)^l y_{l,-m}) (sin(m phi) a + cos(m phi) b)
                dp[..., 2 * i] = yp * c
                dp[..., 2 * i + 1] = -yp * s
                dc[..., 2 * i] = yx * s
                dc[..., 2 * i + 1] = yx * c
            else:
                # h_+ - i h_x gains y e^{i m phi} (a + i b)
                dp[..., 2 * i] = y * c
                dp[..., 2 * i + 1] = -y * s
                dc[..., 2 * i] = -y * s
                dc[..., 2 * i + 1] = -y * c
        return dp, dc


# ------------------------------------------------------------------ engine
def _as_modes(modes, delta_t, label):
    out, dt = {}, delta_t
    for key, h in modes.items():
        l, m = int(key[0]), int(key[1])
        if isinstance(h, TimeSeries):
            if dt is None:
                dt = float(h.delta_t)
            elif abs(float(h.delta_t) - dt) > 1e-9 * dt:
                raise ValueError(f"{label} modes have different delta_t")
            arr = numpy.asarray(h.numpy())
        else:
            arr = numpy.asarray(h)
        out[(l, m)] = arr.astype(complex)
    if dt is None:
        raise ValueError(f"delta_t is required when the {label} modes are arrays")
    return out, dt


def _psd_on_grid(psd, length, delta_f, f_low):
    """The one-sided PSD on k * delta_f, k < length (None: white)."""
    if psd is None:
        return numpy.ones(length)
    if isinstance(psd, str):
        from pycbc.psd import from_string
        series = from_string(psd, length, delta_f, f_low)
        return numpy.asarray(series.numpy(), dtype=float)
    if isinstance(psd, FrequencySeries):
        from pycbc.psd import interpolate
        series = interpolate(psd, delta_f)
        out = numpy.zeros(length)
        n = min(length, len(series))
        out[:n] = numpy.asarray(series.numpy()[:n], dtype=float)
        return out
    if callable(psd):
        return numpy.asarray(psd(numpy.arange(length) * delta_f), dtype=float)
    raise TypeError("psd must be None, a PSD name, a FrequencySeries or a callable")


class _Faithfulness:
    """Shared machinery: component spectra, Gram matrices, correlation series
    on an oversampled lag grid, and the maximisations of Eq. (33)."""

    def __init__(self, signal_modes, template_modes, psd=None,
                 low_frequency_cutoff=None, high_frequency_cutoff=None,
                 delta_t=None, aligned_spin=False, oversample=4,
                 tc_window=0.01, n_template_phase=32, fft_workers=1):
        smodes, dt = _as_modes(signal_modes, delta_t, "signal")
        tmodes, dt2 = _as_modes(template_modes, dt, "template")
        if abs(dt2 - dt) > 1e-9 * dt:
            raise ValueError("signal and template modes have different delta_t")
        self.delta_t = dt
        self.aligned_spin = bool(aligned_spin)
        self.sig = _ModeBasis(smodes, self.aligned_spin, "signal")
        self.tmp = _ModeBasis(tmodes, self.aligned_spin, "template")
        n = max(max(len(h) for h in self.sig.series), max(len(h) for h in self.tmp.series))
        # zero-padded to at least twice the longer waveform: the circular
        # correlation does not wrap within +-n samples of lag
        N = 1 << int(math.ceil(math.log2(2 * n)))
        self.N = N
        self.delta_f = 1.0 / (N * dt)
        kmin, kmax = get_cutoff_indices(low_frequency_cutoff, high_frequency_cutoff,
                                        self.delta_f, N)
        kmin = max(kmin, 1)                     # no DC: the real inverse FFT needs it out
        self.kmin, self.kmax = kmin, kmax
        # an analytic PSD is zeroed below its cutoff, possibly at the band's first
        # bin (int(f_in / df) * df <= f_in): build it from two bins lower
        Sn = _psd_on_grid(psd, N // 2 + 1, self.delta_f, max(kmin - 2, 1) * self.delta_f)
        band = Sn[kmin:kmax]
        if numpy.any(band <= 0):
            raise ValueError("the PSD is not positive across the band")
        w = 4.0 * self.delta_f / band                      # the inner product's weight
        self.fft_workers = int(fft_workers)
        es = sp_fft.rfft(self.sig.components(N), axis=-1, workers=self.fft_workers)[:, kmin:kmax] * dt
        et = sp_fft.rfft(self.tmp.components(N), axis=-1, workers=self.fft_workers)[:, kmin:kmax] * dt
        self._es, self._et, self._w = es, et, w
        # zero-lag Gram matrices, Eq. (32)
        self.Gs = numpy.real((es * w) @ numpy.conj(es).T)
        self.Gt = numpy.real((et * w) @ numpy.conj(et).T)
        # the lag grid: the correlation of band-limited series is band-limited to
        # f_max = kmax * df, so it is sampled at oversample x its Nyquist rate,
        # whatever the waveforms' own rate
        self.oversample = int(oversample)
        Nt = 1 << int(math.ceil(math.log2(max(2 * kmax * self.oversample, 2 * kmax + 2))))
        self.Ntau = Nt
        self.dtau = 1.0 / (Nt * self.delta_f)
        self.tc_window = tc_window
        self.n_template_phase = int(n_template_phase)
        self._phi_grid = 2.0 * numpy.pi * numpy.arange(self.n_template_phase) / self.n_template_phase
        self._lags = None
        self._R = None

    # ---- correlation series
    def _lag_of(self, idx):
        """Lag, in seconds, of oversampled-grid index idx (circular)."""
        idx = numpy.asarray(idx)
        return numpy.where(idx < self.Ntau // 2, idx, idx - self.Ntau) * self.dtau

    def _ifft_band(self, X):
        """Re sum_k X[..., k] exp(2 pi i k n / Ntau) over the band k in [kmin,
        kmax), n = 0..Ntau-1, by a real inverse FFT: the correlation of real
        series is real, and the band holds neither DC (kmin >= 1) nor Nyquist
        (kmax <= Ntau / 2)."""
        pad = numpy.zeros(X.shape[:-1] + (self.Ntau // 2 + 1,), dtype=complex)
        pad[..., self.kmin:self.kmax] = X
        return sp_fft.irfft(pad, n=self.Ntau, axis=-1, workers=self.fft_workers) * (0.5 * self.Ntau)

    # ---- the template as a trigonometric polynomial in its phase
    def _trig(self, iota):
        """`(Dp, Dc)`, each (Bt, 2E): the template's plus and cross coefficients
        are `Dp @ T(phi0_t)` and `Dc @ T(phi0_t)`, with the features
        T = [cos(m_1 phi), sin(m_1 phi), ..., cos(m_E phi), sin(m_E phi)] over the
        template's E basis entries -- every coefficient is a harmonic's
        weight times one of them (`_ModeBasis.polarization_coefficients`)."""
        E = len(self.tmp.entries)
        Dp = numpy.zeros((self.tmp.n, 2 * E))
        Dc = numpy.zeros((self.tmp.n, 2 * E))
        for i, (l, m, folded) in enumerate(self.tmp.entries):
            y = get_glm(l, m, iota)
            if folded:
                ym = (-1) ** l * get_glm(l, -m, iota)
                yp, yx = y + ym, -y + ym
                Dp[2 * i, 2 * i], Dp[2 * i + 1, 2 * i + 1] = yp, -yp
                Dc[2 * i, 2 * i + 1], Dc[2 * i + 1, 2 * i] = yx, yx
            else:
                Dp[2 * i, 2 * i], Dp[2 * i + 1, 2 * i + 1] = y, -y
                Dc[2 * i, 2 * i + 1], Dc[2 * i + 1, 2 * i] = -y, -y
        return Dp, Dc

    def _features(self, phi):
        m = numpy.array([e[1] for e in self.tmp.entries], dtype=float)
        ang = numpy.asarray(phi, dtype=float)[..., None] * m
        T = numpy.empty(ang.shape[:-1] + (2 * len(m),))
        T[..., 0::2] = numpy.cos(ang)
        T[..., 1::2] = numpy.sin(ang)
        return T

    def _locate_peak(self, iotas, n_phase=32):
        """A pre-pass over every lag, for a few source orientations and a grid
        of template phases, giving the lag the window is centred on. (Too
        coarse a phase grid lets a side lobe one cycle away out-score a main
        lobe sampled off its best phase: 16 points did, on a toy.) Phases are
        taken a few at a time, so memory stays a few series."""
        best_val, best_idx = -1.0, 0
        grid = 2.0 * numpy.pi * numpy.arange(n_phase) / n_phase
        T = self._features(grid)
        for iota in iotas:
            Dp, Dc = self._trig(iota)
            dp_all, dc_all = T @ Dp.T, T @ Dc.T                                    # (p, Bt)
            sp2 = numpy.einsum('pk,kl,pl->p', dp_all, self.Gt, dp_all)
            sc2 = numpy.einsum('pk,kl,pl->p', dc_all, self.Gt, dc_all)
            spc = numpy.einsum('pk,kl,pl->p', dp_all, self.Gt, dc_all)
            dps, dcs = self.sig.polarization_coefficients(iota, numpy.array([0.0]))
            for kappa in (0.0, 0.5 * numpy.pi):
                cs = numpy.cos(kappa) * dps + numpy.sin(kappa) * dcs              # (1, Bs)
                ss2 = float(cs[0] @ self.Gs @ cs[0])
                X = (cs @ self._es * self._w)[0][None, :] * numpy.conj(self._et)
                Q = self._ifft_band(X)                                            # (Bt, Ntau)
                for c in range(0, len(grid), 8):
                    sl = slice(c, c + 8)
                    rho2 = self._rho2_from(dp_all[sl] @ Q, dc_all[sl] @ Q,
                                           sp2[sl], sc2[sl], spc[sl])             # (<=8, Ntau)
                    k = numpy.unravel_index(int(numpy.argmax(rho2)), rho2.shape)
                    val = rho2[k] / ss2
                    if val > best_val:
                        best_val, best_idx = val, int(k[1])
        return best_idx

    def _prepare_window(self, iotas=None, centre=None, half=None):
        """Correlation series R[j, k, w] = <e_s[j], e_t[k] delayed by lag w> on
        the window of lags the coalescence time is searched in: every lag if
        tc_window is None, else +-half around centre (the pre-pass's peak)."""
        if self.tc_window is None:
            idx = numpy.arange(self.Ntau)
            self._circular = True
        else:
            if centre is None:
                centre = self._locate_peak(iotas)
            if half is None:
                half = int(math.ceil(self.tc_window / self.dtau))
            if 2 * half + 1 >= self.Ntau:
                idx = numpy.arange(self.Ntau)
                self._circular = True
            else:
                idx = numpy.mod(numpy.arange(centre - half, centre + half + 1), self.Ntau)
                self._circular = False
            self._centre, self._half = centre, half
        self._lags = idx
        # one signal component at a time, to bound the memory
        R = numpy.empty((self.sig.n, self.tmp.n, len(idx)))
        for j in range(self.sig.n):
            R[j] = self._ifft_band((self._es[j] * self._w)[None, :] * numpy.conj(self._et))[:, idx]
        self._R = R

    # ---- the statistic
    def _rho2(self, Q, dp, dc):
        """max over kappa_t of <h_s, h_t>^2 / <h_t, h_t>, for Q[o, k, w] the
        correlations of each source orientation's strain with the template
        components, and template coefficients dp, dc of shape (p, k).
        Returns (o, p, w)."""
        P = numpy.einsum('okw,pk->opw', Q, dp, optimize=True)
        C = numpy.einsum('okw,pk->opw', Q, dc, optimize=True)
        sp2 = numpy.einsum('pk,kl,pl->p', dp, self.Gt, dp)
        sc2 = numpy.einsum('pk,kl,pl->p', dc, self.Gt, dc)
        spc = numpy.einsum('pk,kl,pl->p', dp, self.Gt, dc)
        return self._rho2_from(P, C, sp2, sc2, spc)

    @staticmethod
    def _rho2_from(P, C, sp2, sc2, spc):
        """The closed form of `max_over_polarization`, squared, from unnormalised
        overlaps; where the cross template vanishes (aligned spins edge-on) the
        span is one-dimensional and the maximum is |P| / |h_+|."""
        sp2 = numpy.asarray(sp2)[..., None]
        sc2 = numpy.asarray(sc2)[..., None]
        spc = numpy.asarray(spc)[..., None]
        tiny = 1e-24 * numpy.maximum(sp2, sc2)
        one_d = (sc2 <= tiny) | (sp2 <= tiny)
        safe_sp2 = numpy.where(sp2 > tiny, sp2, 1.0)
        safe_sc2 = numpy.where(sc2 > tiny, sc2, 1.0)
        Ph = P / numpy.sqrt(safe_sp2)
        Ch = C / numpy.sqrt(safe_sc2)
        I = numpy.where(one_d, 0.0, spc / numpy.sqrt(safe_sp2 * safe_sc2))
        rho2 = (Ph * Ph + Ch * Ch - 2.0 * Ph * Ch * I) / (1.0 - I * I)
        rho2_1d = numpy.where(sp2 > tiny, Ph * Ph, Ch * Ch)
        return numpy.where(one_d, rho2_1d, rho2)

    def evaluate(self, iota, phis, kappas, n_candidates=3):
        """`evaluate_all` at one inclination."""
        return self.evaluate_all([iota], phis, kappas, n_candidates)

    def evaluate_all(self, iotas, phis, kappas, n_candidates=3):
        """F, SNR, and the maximising t_c, phi0_t, kappa_t for every source
        orientation: each inclination in `iotas` with every (phi0_s, kappa_s) in
        the outer product of `phis` and `kappas` (inclination major, then
        phi0_s, then kappa_s); `edge` flags an orientation whose best lag is at
        the edge of the window.

        For each orientation the (template phase x lag) grid is searched and
        its `n_candidates` highest local maxima are refined -- the phase by
        golden section, the lag by climbing to the best of its neighbours and a
        quartic through five refined values -- and the best is kept, so a side
        lobe sampled closer to its own best phase cannot stand in for the main
        one. Every candidate of every orientation is refined at once."""
        phis = numpy.atleast_1d(numpy.asarray(phis, dtype=float))
        kappas = numpy.atleast_1d(numpy.asarray(kappas, dtype=float))
        T_grid = self._features(self._phi_grid)                                # (npg, 2E)
        trig = [self._trig(iota) for iota in iotas]
        Dps = numpy.stack([d[0] for d in trig])                                # (G, Bt, 2E)
        Dcs = numpy.stack([d[1] for d in trig])
        Mpp = numpy.einsum('gka,kl,glb->gab', Dps, self.Gt, Dps)
        Mcc = numpy.einsum('gka,kl,glb->gab', Dcs, self.Gt, Dcs)
        Mpc = numpy.einsum('gka,kl,glb->gab', Dps, self.Gt, Dcs)
        circ = self._circular
        K = int(n_candidates)
        Qs, ss2s, PC, WC, VALID, GS = [], [], [], [], [], []
        for g, iota in enumerate(iotas):
            dps, dcs = self.sig.polarization_coefficients(iota, phis)          # (nphi, Bs)
            cs = (numpy.cos(kappas)[None, :, None] * dps[:, None, :]
                  + numpy.sin(kappas)[None, :, None] * dcs[:, None, :]).reshape(-1, self.sig.n)
            ss2 = numpy.einsum('oj,jl,ol->o', cs, self.Gs, cs)
            Q = numpy.einsum('oj,jkw->okw', cs, self._R, optimize=True)        # (o, Bt, W)
            r = self._rho2(Q, T_grid @ Dps[g].T, T_grid @ Dcs[g].T)            # (o, p, W)
            no, npg, nw = r.shape
            # local maxima: periodic in phase, and in lag only if circular
            ninf = numpy.full((no, npg, 1), -numpy.inf)
            if circ:
                lt, rt = numpy.roll(r, 1, axis=2), numpy.roll(r, -1, axis=2)
            else:
                lt = numpy.concatenate([ninf, r[:, :, :-1]], axis=2)
                rt = numpy.concatenate([r[:, :, 1:], ninf], axis=2)
            peak = ((r >= numpy.roll(r, 1, axis=1)) & (r >= numpy.roll(r, -1, axis=1))
                    & (r >= lt) & (r >= rt))
            masked = numpy.where(peak, r, -numpy.inf).reshape(no, -1)
            Kg = min(K, masked.shape[1])
            top = numpy.argpartition(-masked, Kg - 1, axis=1)[:, :Kg]          # (o, K)
            pc, wc = numpy.unravel_index(top, (npg, nw))
            Qs.append(Q)
            ss2s.append(ss2)
            PC.append(pc)
            WC.append(wc)
            VALID.append(numpy.isfinite(numpy.take_along_axis(masked, top, axis=1)))
            GS.append(numpy.full(no, g))
        Q = numpy.concatenate(Qs)
        ss2 = numpy.concatenate(ss2s)
        pc = numpy.concatenate(PC)
        wc = numpy.concatenate(WC)
        valid = numpy.concatenate(VALID).ravel()
        g_of = numpy.concatenate(GS)
        no, nw = len(ss2), Q.shape[-1]
        Kg = pc.shape[1]
        oo = numpy.repeat(numpy.arange(no), Kg)
        gg = g_of[oo]
        pc, wc = pc.ravel(), wc.ravel()
        Dp_c, Dc_c = Dps[gg], Dcs[gg]                                           # (n, Bt, 2E)
        Mpp_c, Mcc_c, Mpc_c = Mpp[gg], Mcc[gg], Mpc[gg]                         # (n, 2E, 2E)

        def uvecs(wv):
            q = Q[oo, :, wv]                                                    # (n, Bt)
            return (numpy.einsum('nk,nka->na', q, Dp_c),
                    numpy.einsum('nk,nka->na', q, Dc_c))

        def f(phi, uP, uC):
            T = self._features(phi)                                             # (n, 2E)
            P = numpy.einsum('na,na->n', uP, T)
            C = numpy.einsum('na,na->n', uC, T)
            sp2 = numpy.einsum('na,nab,nb->n', T, Mpp_c, T)
            sc2 = numpy.einsum('na,nab,nb->n', T, Mcc_c, T)
            spc = numpy.einsum('na,nab,nb->n', T, Mpc_c, T)
            return self._rho2_from(P[:, None], C[:, None], sp2, sc2, spc)[:, 0]

        def golden(uP, uC, lo, hi, iters=30):
            a, b = numpy.array(lo, dtype=float), numpy.array(hi, dtype=float)
            c = b - _INV_GOLDEN * (b - a)
            d = a + _INV_GOLDEN * (b - a)
            fc, fd = f(c, uP, uC), f(d, uP, uC)
            for _ in range(iters):
                left = fc > fd
                b = numpy.where(left, d, b)
                a = numpy.where(left, a, c)
                c_new = b - _INV_GOLDEN * (b - a)
                d_new = a + _INV_GOLDEN * (b - a)
                # the surviving interior point is reused
                x_new = numpy.where(left, c_new, d_new)
                f_new = f(x_new, uP, uC)
                c, d = numpy.where(left, c_new, d), numpy.where(left, c, d_new)
                fc, fd = numpy.where(left, f_new, fd), numpy.where(left, fc, f_new)
            x = 0.5 * (a + b)
            return f(x, uP, uC), x

        # The phase is refined in a bracket about the optimum of the adjacent
        # lag, not of the grid: the optimum drifts with the lag (by ~0.04 rad a
        # step on a 4x-oversampled grid), and a bracket of one grid step would
        # stop containing it as the phase grid is refined. Maxima in phi0_t lie
        # >~ 2 pi / (2 m_max) apart, so +-0.25 rad stays unimodal.
        half = max(2.0 * numpy.pi / len(self._phi_grid), 0.25)

        def at(wv, centre_phi):
            if circ:
                wv = numpy.mod(wv, nw)
            ok = (wv >= 0) & (wv < nw)
            uP, uC = uvecs(numpy.clip(wv, 0, nw - 1))
            v, ph = golden(uP, uC, centre_phi - half, centre_phi + half)
            return numpy.where(ok, v, -numpy.inf), ph

        # The optimal phase moves with the lag, so the grid's local maximum can
        # sit a lag away from the refined one: climb in lag, the phase
        # re-refined about its last optimum at every step, until the centre is
        # the best of its two neighbours.
        v0, ph0 = at(wc, self._phi_grid[pc].astype(float))
        for _ in range(8):
            vm, phm = at(wc - 1, ph0)
            vp, php = at(wc + 1, ph0)
            up = (vp > v0) & (vp >= vm)
            down = (vm > v0) & ~up
            if not numpy.any(up | down):
                break
            wc = numpy.where(up, wc + 1, numpy.where(down, wc - 1, wc))
            if circ:
                wc = numpy.mod(wc, nw)
            v0 = numpy.where(up, vp, numpy.where(down, vm, v0))
            ph0 = numpy.where(up, php, numpy.where(down, phm, ph0))
        # the peak in lag: a quartic through five phase-refined values where the
        # window allows (its x^4 term cancels the parabola's leading error),
        # else the parabola through three
        interior = numpy.ones_like(wc, dtype=bool) if circ else (wc > 0) & (wc < nw - 1)
        five = numpy.ones_like(wc, dtype=bool) if circ else (wc > 1) & (wc < nw - 2)
        with numpy.errstate(invalid='ignore', over='ignore'):
            curv = vm - 2.0 * v0 + vp
            concave = interior & (curv < 0) & (v0 >= vm) & (v0 >= vp)
            delta = numpy.where(concave, 0.5 * (vm - vp) / numpy.where(concave, curv, 1.0), 0.0)
            peakv = numpy.where(concave, v0 - 0.25 * (vm - vp) * delta,
                                numpy.maximum(numpy.maximum(vm, v0), vp))
            quart = concave & five
            if numpy.any(quart):
                vmm, _ = at(wc - 2, phm)
                vpp, _ = at(wc + 2, php)
                # the interpolating quartic in x (lag offset in grid steps)
                # through x = -2..2, its coefficients from the values exactly
                c0 = v0
                c1 = (vmm - 8.0 * vm + 8.0 * vp - vpp) / 12.0
                c2 = (-vmm + 16.0 * vm - 30.0 * v0 + 16.0 * vp - vpp) / 24.0
                c3 = (-vmm + 2.0 * vm - 2.0 * vp + vpp) / 12.0
                c4 = (vmm - 4.0 * vm + 6.0 * v0 - 4.0 * vp + vpp) / 24.0
                x = numpy.where(quart, delta, 0.0)
                for _ in range(20):                              # Newton on p'(x) = 0
                    d1 = c1 + 2 * c2 * x + 3 * c3 * x ** 2 + 4 * c4 * x ** 3
                    d2 = 2 * c2 + 6 * c3 * x + 12 * c4 * x ** 2
                    x = numpy.where(quart & (d2 < 0), x - d1 / numpy.where(d2 < 0, d2, -1.0), x)
                ok = quart & (numpy.abs(x) <= 1.0)
                pq = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
                delta = numpy.where(ok, x, delta)
                peakv = numpy.where(ok, pq, peakv)
        peakv = numpy.where(valid, peakv, -numpy.inf)
        best = numpy.argmax(peakv.reshape(no, Kg), axis=1)
        sel = numpy.arange(no) * Kg + best
        F = numpy.sqrt(numpy.maximum(peakv[sel], 0.0) / ss2)
        edge = ~five[sel] if not circ else numpy.zeros(no, dtype=bool)
        # the maximisers, for diagnostics: phi0_t and kappa_t optimal at the grid
        # lag tc_grid, where a template built with them has faithfulness_grid
        w_best, phi_t = wc[sel], ph0[sel]
        uP, uC = uvecs(wc)
        T = self._features(phi_t)
        P = numpy.einsum('na,na->n', uP[sel], T)
        C = numpy.einsum('na,na->n', uC[sel], T)
        sp2 = numpy.einsum('na,nab,nb->n', T, Mpp_c[sel], T)
        sc2 = numpy.einsum('na,nab,nb->n', T, Mcc_c[sel], T)
        spc = numpy.einsum('na,nab,nb->n', T, Mpc_c[sel], T)
        two = sc2 > 1e-24 * sp2
        safe = numpy.where(two, sc2, 1.0)
        Ph = P / numpy.sqrt(sp2)
        Ch = numpy.where(two, C / numpy.sqrt(safe), 0.0)
        I = numpy.where(two, spc / numpy.sqrt(sp2 * safe), 0.0)
        # the optimum is alpha h+^ + beta hx^ in the normalised basis, i.e.
        # proportional to cos(kappa_t) h_+ + sin(kappa_t) h_x with
        # tan(kappa_t) = (beta / |h_x|) / (alpha / |h_+|), Eq. (30)'s angle
        alpha = Ph - I * Ch
        beta = numpy.where(two, Ch - I * Ph, 0.0)
        kap = numpy.mod(numpy.arctan2(beta / numpy.sqrt(safe), alpha / numpy.sqrt(sp2)),
                        2.0 * numpy.pi)
        tc_grid = self._lag_of(self._lags[w_best])
        return dict(faithfulness=F, snr=numpy.sqrt(ss2), tc=tc_grid + delta[sel] * self.dtau,
                    template_phi0=numpy.mod(phi_t, 2.0 * numpy.pi), template_kappa=kap,
                    tc_grid=tc_grid, faithfulness_grid=numpy.sqrt(numpy.maximum(v0[sel], 0.0) / ss2),
                    edge=edge, lag_index=self._lags[w_best])


# ------------------------------------------------------------------ public
def _evaluate_all(eng, iotas, phis, kappas, max_widen=6):
    """Evaluate every inclination; while any orientation's best lag sits on
    the edge of the window, double the window about the same centre (so no
    earlier peak leaves it) and evaluate again."""
    eng._prepare_window(iotas=[iotas[0], iotas[-1]])
    for _ in range(max_widen + 1):
        res = eng.evaluate_all(iotas, phis, kappas)
        if eng._circular or not numpy.any(res['edge']):
            return res
        eng._prepare_window(centre=eng._centre, half=2 * eng._half)
    logger.warning("the best coalescence time is still at the edge of the "
                   "tc_window after widening it %d times", max_widen)
    return res


def _quadrature(n_inclination, n_phase, n_polarization, aligned_spin):
    # Gauss-Legendre on [0, 1] -- and, without the aligned-spin symmetry, on
    # [-1, 0] too. The rule is split at cos(iota) = 0 because the integrand
    # can have a kink there: for aligned spins h_x is odd in cos(iota), so at
    # kappa = pi/2 SNR^3 goes as |cos(iota)|^3, and one rule across [-1, 1]
    # converges only algebraically through it.
    x, wx = roots_legendre(int(n_inclination))
    cos_iota, w_iota = 0.5 * (x + 1.0), 0.5 * wx
    if not aligned_spin:
        cos_iota = numpy.concatenate([-cos_iota[::-1], cos_iota])
        w_iota = numpy.concatenate([w_iota[::-1], w_iota])
    phis = 2.0 * numpy.pi * numpy.arange(int(n_phase)) / int(n_phase)
    # kappa_s + pi flips the sign of h_s, which kappa_t absorbs: [0, pi) suffices
    kappas = numpy.pi * numpy.arange(int(n_polarization)) / int(n_polarization)
    return numpy.arccos(cos_iota), w_iota, phis, kappas


def sky_averaged_unfaithfulness(signal_modes, template_modes, psd=None,
                                low_frequency_cutoff=None,
                                high_frequency_cutoff=None, delta_t=None,
                                aligned_spin=False, n_inclination=8,
                                n_phase=8, n_polarization=8,
                                n_template_phase=32, tc_window=0.01,
                                oversample=4, fft_workers=1,
                                return_details=False):
    """The SNR-weighted, sky-and-polarization-averaged unfaithfulness.

    :math:`\\bar{\\mathcal{M}}_\\mathrm{SNR} = 1 - \\bar{\\mathcal{F}}_\\mathrm{SNR}`
    of arXiv:2303.18046, Eqs. (30)-(36); see the module docstring for every
    equation. The signal is ``signal_modes`` (the reference), the template is
    ``template_modes`` (the model), both at the same intrinsic parameters.

    Parameters
    ----------
    signal_modes, template_modes : dict
        ``(l, m) -> complex TimeSeries`` (or complex array with ``delta_t``),
        both at the same sample rate. Pycbc's convention for the modes:
        ``h_+ - i h_x = sum_lm -2Y_lm(iota, phi0) h_lm``. Each should be
        tapered at its start by the caller.
    psd : None, str, FrequencySeries or callable
        The noise PSD: a name for :func:`pycbc.psd.from_string` (the paper:
        ``'aLIGOZeroDetHighPower'``), a FrequencySeries (interpolated), a
        function of frequency, or None for white noise.
    low_frequency_cutoff, high_frequency_cutoff : float
        :math:`f_\\mathrm{in}` and :math:`f_\\mathrm{max}`: the paper uses 10 and
        2048 Hz when both waveforms are in band, and
        :func:`nr_low_frequency_cutoff` otherwise.
    aligned_spin : bool
        If True, modes with m < 0 are taken from
        :math:`h_{\\ell,-m} = (-1)^\\ell h_{\\ell m}^*` (given ones are checked
        against it), and cos(iota) is integrated over [0, 1] by symmetry.
        Use False for precessing modes.
    n_inclination : int
        Gauss-Legendre nodes in cos(iota_s) on [0, 1] (and as many again on
        [-1, 0] when ``aligned_spin`` is False).
    n_phase, n_polarization : int
        Uniform nodes in phi0_s on [0, 2 pi) and kappa_s on [0, pi) (the
        integrand has period pi in kappa_s).
    n_template_phase : int
        Grid in phi0_t before the golden-section refinement.
    tc_window : float or None
        Half-width, in seconds, of the coalescence-time search around the
        best lag of a pre-pass (None: every lag; slow).
    oversample : int
        The lag grid samples the correlations at this multiple of the rate
        their band needs; the peak is refined by a quartic through five
        points. Measured against a brute-force maximisation on a toy: 4 (the
        default) agrees to ~1e-10 in F, 2 to ~1e-8 with inverse FFTs half as
        long. With higher harmonics reaching 1-2 kHz (SEOBNRv5HM's seven modes
        at 20 solar masses, band to 2048 Hz) 4 and 8 differ by 5e-5 of U.
    fft_workers : int
        Threads for scipy's FFTs.
    return_details : bool
        Also return the per-orientation faithfulness, SNRs, maximisers and
        quadrature.

    Returns
    -------
    float, or (float, dict)
        :math:`\\bar{\\mathcal{M}}_\\mathrm{SNR}`, and with ``return_details``
        a dict with ``faithfulness_snr`` (Eq. 36), ``faithfulness_mean``
        (Eq. 35), and per orientation ``iota``, ``phi0``, ``kappa``,
        ``faithfulness``, ``snr``, ``tc``, ``template_phi0``,
        ``template_kappa``, ``weight``.
    """
    eng = _Faithfulness(signal_modes, template_modes, psd=psd,
                        low_frequency_cutoff=low_frequency_cutoff,
                        high_frequency_cutoff=high_frequency_cutoff,
                        delta_t=delta_t, aligned_spin=aligned_spin,
                        oversample=oversample, tc_window=tc_window,
                        n_template_phase=n_template_phase,
                        fft_workers=fft_workers)
    iotas, w_iota, phis, kappas = _quadrature(n_inclination, n_phase, n_polarization,
                                              aligned_spin)
    out = _evaluate_all(eng, iotas, phis, kappas)
    n = len(phis) * len(kappas)
    P, K = numpy.meshgrid(phis, kappas, indexing='ij')
    rec = {'iota': numpy.repeat(iotas, n), 'phi0': numpy.tile(P.ravel(), len(iotas)),
           'kappa': numpy.tile(K.ravel(), len(iotas)), 'weight': numpy.repeat(w_iota, n)}
    for k in ('faithfulness', 'snr', 'tc', 'template_phi0', 'template_kappa', 'tc_grid',
              'faithfulness_grid'):
        rec[k] = out[k]
    F_snr = snr_weighted_average(rec['faithfulness'], rec['snr'], rec['weight'])
    U = 1.0 - F_snr
    if not return_details:
        return U
    rec['faithfulness_snr'] = F_snr
    rec['faithfulness_mean'] = float(numpy.sum(rec['weight'] * rec['faithfulness'])
                                     / numpy.sum(rec['weight']))
    rec['unfaithfulness'] = U
    return U, rec


def faithfulness_at_orientation(signal_modes, template_modes, iota, phi0,
                                kappa, psd=None, low_frequency_cutoff=None,
                                high_frequency_cutoff=None, delta_t=None,
                                aligned_spin=False, n_template_phase=32,
                                tc_window=0.01, oversample=4, fft_workers=1):
    """Eq. (33) at one source orientation.

    Parameters are as for :func:`sky_averaged_unfaithfulness`; ``iota``,
    ``phi0`` and ``kappa`` are the source's.

    Returns
    -------
    dict
        ``faithfulness`` and ``snr`` (the signal's); ``tc``, the template's
        delay (s) at the interpolated peak; and ``template_phi0``,
        ``template_kappa``, the template's phase and Eq. (30) polarization at
        the grid lag ``tc_grid`` nearest it, where a template built with them
        and delayed by ``tc_grid`` has the overlap ``faithfulness_grid``.
    """
    eng = _Faithfulness(signal_modes, template_modes, psd=psd,
                        low_frequency_cutoff=low_frequency_cutoff,
                        high_frequency_cutoff=high_frequency_cutoff,
                        delta_t=delta_t, aligned_spin=aligned_spin,
                        oversample=oversample, tc_window=tc_window,
                        n_template_phase=n_template_phase,
                        fft_workers=fft_workers)
    out = _evaluate_all(eng, [float(iota)], [float(phi0)], [float(kappa)])
    return {k: (bool(v[0]) if k == 'edge' else float(v[0])) for k, v in out.items()}


class _MapWorker:
    """Picklable: generate one pair in the worker, then compute U."""

    def __init__(self, generator, kwargs):
        self.generator = generator
        self.kwargs = kwargs

    def __call__(self, item):
        pair = self.generator(item)
        kwargs = dict(self.kwargs)
        if len(pair) == 3:
            signal, template, extra = pair
            kwargs.update(extra)
        else:
            signal, template = pair
        return sky_averaged_unfaithfulness(signal, template, **kwargs)


def sky_averaged_unfaithfulness_map(generator, items, processes=1, **kwargs):
    """Map :func:`sky_averaged_unfaithfulness` over many binaries in parallel.

    The waveforms are generated inside the workers, so only each ``item`` is
    sent to a worker, not the waveforms.

    Parameters
    ----------
    generator : callable
        A picklable (module-level) function of one item returning
        ``(signal_modes, template_modes)`` or
        ``(signal_modes, template_modes, extra_kwargs)``, the last overriding
        ``kwargs`` for that item (e.g. its frequency cutoffs).
    items : iterable
        One per binary.
    processes : int
        Worker processes (:func:`pycbc.pool.choose_pool`); 1 runs serially.
    **kwargs
        Passed to :func:`sky_averaged_unfaithfulness`.

    Returns
    -------
    list
        One result per item, in order.
    """
    worker = _MapWorker(generator, kwargs)
    items = list(items)
    if processes is None or int(processes) <= 1:
        return [worker(it) for it in items]
    from pycbc.pool import choose_pool
    pool = choose_pool(int(processes))
    try:
        return list(pool.map(worker, items))
    finally:
        pool.close()
        pool.join()
