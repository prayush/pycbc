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
Tests of pycbc.filter.faithfulness: the SNR-weighted, sky-and-polarization
averaged unfaithfulness of arXiv:2303.18046, Eqs. (30)-(36).

The fast implementation is checked against things that can only be right:
pycbc's own mode sum and match, the identity, exact invariances, and a slow
brute-force implementation written here from the paper's definitions with
explicit polarizations, explicit time shifts, pycbc's overlap and a numerical
maximisation over every template parameter -- no closed form anywhere.

Set PYCBC_TEST_SLOW=1 to also run the tests with physical waveforms.
"""
import math
import os
import unittest

import numpy
from scipy.optimize import minimize

from pycbc.types import TimeSeries
from pycbc.filter import (match, overlap, sigma,
                          compute_max_snr_over_sky_loc_stat_no_phase)
from pycbc.filter.faithfulness import (
    sky_averaged_unfaithfulness, faithfulness_at_orientation,
    max_over_polarization, snr_weighted_average, nr_low_frequency_cutoff,
    sky_averaged_unfaithfulness_map, _ModeBasis)
from pycbc.waveform.waveform_modes import sum_modes
from pycbc.psd import from_string

DT = 1.0 / 4096
F_LOW, F_HIGH = 20.0, 1024.0
PSD = 'aLIGOZeroDetHighPower'
SLOW = bool(os.environ.get('PYCBC_TEST_SLOW'))


# ------------------------------------------------------------------- toys
def toy_modes(n=6000, f_start=24.0, f_end=180.0, phases=None, drop=(),
              shift=0, amps=None, negative_m=False):
    """An analytic aligned-spin multimode chirp.

    The (2,2) frequency sweeps smoothly from f_start to f_end; the orbital
    phase is half the (2,2) phase; each mode is A_lm a(t) exp(-i m Phi(t) +
    i delta_lm) with a(t) ~ f^(2/3), a smooth start taper and a smooth end.
    `negative_m` adds h_{l,-m} = (-1)^l conj(h_lm)."""
    t = numpy.arange(n) * DT
    T = t[-1]
    x = t / T
    f22 = f_start + (f_end - f_start) * x ** 3
    phi_orb = numpy.pi * numpy.cumsum(f22) * DT
    win = numpy.ones(n)
    k = n // 10
    win[:k] = 0.5 * (1 - numpy.cos(numpy.pi * numpy.arange(k) / k))
    win[-k:] = 0.5 * (1 + numpy.cos(numpy.pi * numpy.arange(k) / k))
    a = (f22 / f_start) ** (2.0 / 3.0) * win
    amps = amps or {(2, 2): 1.0, (2, 1): 0.15, (3, 3): 0.25, (4, 4): 0.1}
    phases = phases or {}
    out = {}
    for (l, m), A in amps.items():
        if (l, m) in drop:
            continue
        h = numpy.zeros(n + shift, dtype=complex)
        h[shift:] = A * a * numpy.exp(-1j * m * phi_orb + 1j * phases.get((l, m), 0.0))
        out[(l, m)] = TimeSeries(h, delta_t=DT)
        if negative_m:
            out[(l, -m)] = TimeSeries((-1) ** l * numpy.conj(h), delta_t=DT)
    return out


def with_negative_m(modes):
    out = dict(modes)
    for (l, m), h in modes.items():
        if m > 0:
            out[(l, -m)] = TimeSeries((-1) ** l * numpy.conj(h.numpy()), delta_t=h.delta_t)
    return out


def _arrays(modes):
    return {k: numpy.asarray(v.numpy() if hasattr(v, 'numpy') else v) for k, v in modes.items()}


def strain(modes, iota, phi0, kappa, length):
    """h = cos(kappa) h_+ + sin(kappa) h_x from pycbc's own sum_modes, padded."""
    hpc = numpy.asarray(sum_modes(_arrays(modes), iota, phi0))
    h = numpy.cos(kappa) * numpy.real(hpc) - numpy.sin(kappa) * numpy.imag(hpc)
    out = numpy.zeros(length)
    out[:len(h)] = h
    return TimeSeries(out, delta_t=DT)


def pol(modes, iota, phi0, length):
    hpc = numpy.asarray(sum_modes(_arrays(modes), iota, phi0))
    hp, hc = numpy.zeros(length), numpy.zeros(length)
    hp[:len(hpc)] = numpy.real(hpc)
    hc[:len(hpc)] = -numpy.imag(hpc)
    return TimeSeries(hp, delta_t=DT), TimeSeries(hc, delta_t=DT)


def psd_for(length):
    df = 1.0 / (length * DT)
    return from_string(PSD, length // 2 + 1, df, F_LOW - 2 * df)


def brute_force_faithfulness(signal, template, iota, phi0, kappa, length=16384,
                             n_phase=48, n_kappa=36):
    """Eq. (33) the obvious way: explicit signal and template strains from
    sum_modes, pycbc's overlap, the template delayed explicitly in the
    frequency domain, and every template parameter (t_c, phi0_t, kappa_t)
    maximised numerically -- a grid, then Nelder-Mead from its best point."""
    psd = psd_for(length)
    s = strain(signal, iota, phi0, kappa, length).to_frequencyseries()
    f = s.sample_frequencies.numpy()
    snr = sigma(s, psd=psd, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH)
    cache = {}

    def pols(ph):
        key = round(float(ph), 15)
        if key not in cache:
            hp, hc = pol(template, iota, ph, length)
            cache[key] = (hp.to_frequencyseries(), hc.to_frequencyseries())
        return cache[key]

    def value(ph, tc, kap):
        hp, hc = pols(ph)
        ht = numpy.cos(kap) * hp + numpy.sin(kap) * hc
        ht = ht * numpy.exp(-2j * numpy.pi * f * tc)
        return overlap(s, ht, psd=psd, low_frequency_cutoff=F_LOW,
                       high_frequency_cutoff=F_HIGH, normalized=True)

    # coarse: the best lag from pycbc's match of the plus polarisations
    hp0, _ = pols(phi0)
    _, idx = match(s, hp0, psd=psd, low_frequency_cutoff=F_LOW,
                   high_frequency_cutoff=F_HIGH)
    lag0 = (idx if idx < length // 2 else idx - length) * DT
    best = (-2.0, None)
    for ph in 2 * numpy.pi * numpy.arange(n_phase) / n_phase:
        for kap in 2 * numpy.pi * numpy.arange(n_kappa) / n_kappa:
            for tc in lag0 + DT * numpy.arange(-2, 3):
                v = value(ph, tc, kap)
                if v > best[0]:
                    best = (v, (ph, tc, kap))
    x0 = numpy.array(best[1])
    scale = numpy.array([2 * numpy.pi / n_phase, DT, 2 * numpy.pi / n_kappa])
    res = minimize(lambda z: -value(*(x0 + z * scale)), numpy.zeros(3),
                   method='Nelder-Mead',
                   options={'xatol': 1e-9, 'fatol': 1e-14, 'maxiter': 4000})
    return -res.fun, snr


# ------------------------------------------------------------------ tests
class TestPolarizationMaximum(unittest.TestCase):

    def test_closed_form_is_the_projection_norm(self):
        rng = numpy.random.default_rng(1)
        for _ in range(20):
            s, hp, hc = rng.normal(size=(3, 50))
            hp /= numpy.linalg.norm(hp)
            hc /= numpy.linalg.norm(hc)
            P, C, I = s @ hp, s @ hc, hp @ hc
            rho, kap = max_over_polarization(P, C, I)
            # the norm of the projection of s onto span(hp, hc)
            B = numpy.stack([hp, hc], 1)
            proj = B @ numpy.linalg.lstsq(B, s, rcond=None)[0]
            self.assertAlmostEqual(float(rho), numpy.linalg.norm(proj), places=12)
            # and the maximiser points along it
            u = numpy.cos(kap) * hp + numpy.sin(kap) * hc
            self.assertAlmostEqual(float(s @ u / numpy.linalg.norm(u)), float(rho), places=12)

    def test_equals_a_brute_force_scan(self):
        rng = numpy.random.default_rng(2)
        P, C = rng.normal(size=(2, 30))
        I = rng.uniform(-0.95, 0.95, 30)
        rho, _ = max_over_polarization(P, C, I)
        k = numpy.linspace(0, 2 * numpy.pi, 200001)[:, None]
        scan = numpy.max((numpy.cos(k) * P + numpy.sin(k) * C)
                         / numpy.sqrt(1 + 2 * numpy.sin(k) * numpy.cos(k) * I), axis=0)
        numpy.testing.assert_allclose(rho, scan, rtol=1e-9)

    def test_equals_pycbcs_sky_location_statistic(self):
        rng = numpy.random.default_rng(3)
        P, C = rng.normal(size=(2, 40))
        for I in (-0.6, 0.0, 0.3, 0.9):
            rho, _ = max_over_polarization(P, C, I)
            stat = compute_max_snr_over_sky_loc_stat_no_phase(
                P.astype(complex), C.astype(complex), I, hpnorm=1.0, hcnorm=1.0)
            numpy.testing.assert_allclose(rho, numpy.asarray(stat), rtol=1e-12)

    def test_no_phase_freedom(self):
        # a template that is the signal phase-rotated by 90 degrees, (h_+, h_x) -> (h_x, -h_+)
        # in a one-dimensional span, is NOT matched: only kappa_t is free
        rng = numpy.random.default_rng(4)
        s = rng.normal(size=60)
        hp = s / numpy.linalg.norm(s)
        rho, _ = max_over_polarization(s @ hp, 0.0, 0.0)
        self.assertAlmostEqual(float(rho), numpy.linalg.norm(s), places=12)
        rng2 = rng.normal(size=60)
        q = rng2 - (rng2 @ hp) * hp
        q /= numpy.linalg.norm(q)
        rho, _ = max_over_polarization(s @ q, 0.0, 0.0)
        self.assertLess(float(rho), 1e-12)


class TestHelpers(unittest.TestCase):

    def test_snr_weighted_average(self):
        F = numpy.full(17, 0.987)
        snr = numpy.random.default_rng(5).uniform(1, 9, 17)
        self.assertAlmostEqual(snr_weighted_average(F, snr), 0.987, places=14)
        F = numpy.random.default_rng(6).uniform(0.9, 1.0, 17)
        w = numpy.random.default_rng(7).uniform(0.1, 1.0, 17)
        expect = (numpy.sum(w * F ** 3 * snr ** 3) / numpy.sum(w * snr ** 3)) ** (1 / 3)
        self.assertAlmostEqual(snr_weighted_average(F, snr, w), expect, places=14)
        # the cube root: a plain SNR^3-weighted mean of F would differ
        self.assertNotAlmostEqual(snr_weighted_average(F, snr, w),
                                  numpy.sum(w * F * snr ** 3) / numpy.sum(w * snr ** 3), places=6)

    def test_nr_low_frequency_cutoff(self):
        n = 8192
        t = numpy.arange(n) * DT
        win = numpy.hanning(n)
        h = win * numpy.exp(-2j * numpy.pi * 50.0 * t)
        f = nr_low_frequency_cutoff(TimeSeries(h, delta_t=DT))
        self.assertAlmostEqual(f, 1.35 * 50.0, delta=1.35 / (n * DT))

    def test_mode_basis_reproduces_pycbcs_sum_modes(self):
        """The folded (aligned-spin) and unfolded bases build exactly the
        polarizations pycbc's sum_modes builds, from LAL's harmonics."""
        positive = toy_modes(n=2000)
        full = with_negative_m(positive)
        for aligned, modes in ((False, full), (True, positive), (True, full)):
            b = _ModeBasis({k: v.numpy() for k, v in modes.items()}, aligned, 'x')
            e = b.components(2000)
            for iota in (0.2, 1.1, math.pi / 2, 2.6):
                for phi in (0.0, 0.9, 4.1):
                    dp, dc = b.polarization_coefficients(iota, numpy.array([phi]))
                    ref = sum_modes({k: v.numpy() for k, v in full.items()}, iota, phi)
                    numpy.testing.assert_allclose(dp[0] @ e, numpy.real(ref), atol=1e-13)
                    numpy.testing.assert_allclose(dc[0] @ e, -numpy.imag(ref), atol=1e-13)

    def test_aligned_spin_check_rejects_precessing_like_modes(self):
        modes = with_negative_m(toy_modes(n=1000))
        modes[(3, -3)] = TimeSeries(modes[(3, -3)].numpy() * numpy.exp(0.2j), delta_t=DT)
        with self.assertRaises(ValueError):
            _ModeBasis({k: v.numpy() for k, v in modes.items()}, True, 'x')


class TestFaithfulness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sig = toy_modes()
        cls.tmp = toy_modes(f_end=181.0, phases={(3, 3): 0.3}, drop=((4, 4),))
        cls.kw = dict(psd=PSD, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                      aligned_spin=True, n_inclination=4, n_phase=6, n_polarization=4)

    def test_identity(self):
        U, d = sky_averaged_unfaithfulness(self.sig, self.sig, return_details=True, **self.kw)
        self.assertLess(abs(U), 1e-10)
        self.assertLess(numpy.max(numpy.abs(1 - d['faithfulness'])), 1e-10)
        numpy.testing.assert_allclose(d["tc"], 0.0, atol=1e-7)       # < 1e-3 of a sample

    def test_22_only_reduces_to_match(self):
        """With the (2,2) and (2,-2) alone the template's span {h_+, h_x} is the
        22 waveform at every phase, so maximising over phi0_t and kappa_t is
        maximising over a phase: F equals pycbc's match of the face-on plus
        polarizations, at every orientation. Exactly so only where h_x is the
        Hilbert transform of h_+, which the match assumes; here they differ at
        the 1e-7 level, and the brute force shows that spread is in F itself,
        not in the numerics."""
        sig = {(2, 2): self.sig[(2, 2)]}
        tmp = {(2, 2): self.tmp[(2, 2)]}
        U, d = sky_averaged_unfaithfulness(sig, tmp, return_details=True, **self.kw)
        F = d['faithfulness']
        self.assertLess(numpy.ptp(F), 1e-6)
        length = 16384
        hs = strain(with_negative_m(sig), 0.0, 0.0, 0.0, length)
        ht = strain(with_negative_m(tmp), 0.0, 0.0, 0.0, length)
        m, _ = match(hs, ht, psd=psd_for(length), low_frequency_cutoff=F_LOW,
                     high_frequency_cutoff=F_HIGH, subsample_interpolation=True)
        self.assertGreater(1 - m, 1e-4)                      # not a trivial agreement
        self.assertLess(abs(F.mean() - m), 1e-6)
        self.assertLess(abs(U - (1 - m)), 1e-6)
        for k in (int(numpy.argmin(F)), int(numpy.argmax(F))):
            Fb, _ = brute_force_faithfulness(with_negative_m(sig), with_negative_m(tmp),
                                             d['iota'][k], d['phi0'][k], d['kappa'][k])
            self.assertLess(abs(F[k] - Fb), 1e-9)

    def test_invariances(self):
        U0 = sky_averaged_unfaithfulness(self.sig, self.tmp, **self.kw)
        self.assertGreater(U0, 1e-4)
        # the template delayed by whole samples
        Ut = sky_averaged_unfaithfulness(self.sig, toy_modes(f_end=181.0, phases={(3, 3): 0.3},
                                                             drop=((4, 4),), shift=37), **self.kw)
        self.assertLess(abs(Ut - U0), 1e-9)
        # the template's orbital phase rotated: h_lm -> h_lm exp(i m delta)
        rot = {(l, m): TimeSeries(h.numpy() * numpy.exp(1j * m * 1.234), delta_t=DT)
               for (l, m), h in self.tmp.items()}
        self.assertLess(abs(sky_averaged_unfaithfulness(self.sig, rot, **self.kw) - U0), 1e-10)
        # either waveform rescaled
        big = {k: TimeSeries(v.numpy() * 3.7, delta_t=DT) for k, v in self.sig.items()}
        small = {k: TimeSeries(v.numpy() * 0.02, delta_t=DT) for k, v in self.tmp.items()}
        self.assertLess(abs(sky_averaged_unfaithfulness(big, small, **self.kw) - U0), 1e-12)

    def test_sub_sample_delay(self):
        """A template delayed by a fraction of a sample, by a frequency-domain
        phase: the coalescence time is found between samples."""
        U0 = sky_averaged_unfaithfulness(self.sig, self.tmp, **self.kw)
        shifted = {}
        for k, h in self.tmp.items():
            x = numpy.zeros(8192, dtype=complex)
            x[:len(h)] = h.numpy()
            f = numpy.fft.fftfreq(len(x), DT)
            shifted[k] = TimeSeries(numpy.fft.ifft(numpy.fft.fft(x) * numpy.exp(-2j * numpy.pi * f * 0.37 * DT)),
                                    delta_t=DT)
        self.assertLess(abs(sky_averaged_unfaithfulness(self.sig, shifted, **self.kw) - U0), 2e-8)

    def test_against_brute_force(self):
        """Eq. (33) at each orientation, and Eq. (36), against the brute-force
        implementation above."""
        kw = dict(self.kw, n_inclination=2, n_phase=2, n_polarization=2)
        U, d = sky_averaged_unfaithfulness(self.sig, self.tmp, return_details=True, **kw)
        sig, tmp = with_negative_m(self.sig), with_negative_m(self.tmp)
        Fb, Sb = [], []
        for iota, phi0, kappa in zip(d['iota'], d['phi0'], d['kappa']):
            F, snr = brute_force_faithfulness(sig, tmp, iota, phi0, kappa)
            Fb.append(F)
            Sb.append(snr)
        Fb, Sb = numpy.array(Fb), numpy.array(Sb)
        self.assertGreater(numpy.ptp(Fb), 1e-4)             # orientation-dependent
        numpy.testing.assert_allclose(d['faithfulness'], Fb, rtol=0, atol=1e-9)
        numpy.testing.assert_allclose(d['snr'], Sb, rtol=1e-10)
        Ub = 1 - snr_weighted_average(Fb, Sb, d['weight'])
        self.assertLess(abs(U - Ub), 1e-9)
        self.assertGreater(U, 1e-4)

    def test_template_phase_and_polarization_maximisers(self):
        """The refined phi0_t and kappa_t reproduce the faithfulness when the
        template is built explicitly with them."""
        iota, phi0, kappa = 1.3, 0.8, 0.4
        r = faithfulness_at_orientation(self.sig, self.tmp, iota, phi0, kappa,
                                        psd=PSD, low_frequency_cutoff=F_LOW,
                                        high_frequency_cutoff=F_HIGH, aligned_spin=True)
        length = 16384
        psd = psd_for(length)
        s = strain(with_negative_m(self.sig), iota, phi0, kappa, length).to_frequencyseries()
        ht = strain(with_negative_m(self.tmp), iota, r['template_phi0'], r['template_kappa'],
                    length).to_frequencyseries()
        f = ht.sample_frequencies.numpy()
        ht = ht * numpy.exp(-2j * numpy.pi * f * r['tc_grid'])
        o = overlap(s, ht, psd=psd, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH)
        # the reported phase and Eq. (30) polarization rebuild the overlap at
        # the grid lag exactly; the interpolated peak lies above it, by little
        self.assertLess(abs(o - r['faithfulness_grid']), 1e-10)
        self.assertGreaterEqual(r['faithfulness'], r['faithfulness_grid'] - 1e-12)
        self.assertLess(r['faithfulness'] - r['faithfulness_grid'], 1e-3)
        self.assertLess(abs(r['tc'] - r['tc_grid']), DT)
        F_bf, _ = brute_force_faithfulness(with_negative_m(self.sig), with_negative_m(self.tmp),
                                           iota, phi0, kappa)
        self.assertLess(abs(F_bf - r['faithfulness']), 1e-9)

    def test_weight_is_the_signals_snr(self):
        U, d = sky_averaged_unfaithfulness(self.sig, self.tmp, return_details=True, **self.kw)
        length = 16384
        psd = psd_for(length)
        for i in (0, 7, len(d['snr']) - 1):
            s = strain(with_negative_m(self.sig), d['iota'][i], d['phi0'][i], d['kappa'][i], length)
            self.assertAlmostEqual(d['snr'][i] / sigma(s, psd=psd, low_frequency_cutoff=F_LOW,
                                                      high_frequency_cutoff=F_HIGH), 1.0, places=10)
        self.assertAlmostEqual(U, 1 - snr_weighted_average(d['faithfulness'], d['snr'], d['weight']),
                               places=14)
        # the signal's SNR, not the template's: swapping the roles changes U
        U_swap = sky_averaged_unfaithfulness(self.tmp, self.sig, **self.kw)
        self.assertGreater(abs(U_swap - U) / U, 1e-3)

    def test_kappa_period_is_pi(self):
        a = faithfulness_at_orientation(self.sig, self.tmp, 1.0, 0.5, 0.3, psd=PSD,
                                        low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                                        aligned_spin=True)
        b = faithfulness_at_orientation(self.sig, self.tmp, 1.0, 0.5, 0.3 + math.pi, psd=PSD,
                                        low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                                        aligned_spin=True)
        self.assertAlmostEqual(a['faithfulness'], b['faithfulness'], places=12)
        self.assertAlmostEqual(a['snr'] / b['snr'], 1.0, places=12)

    def test_inclination_reflection(self):
        """Aligned spins: the source at (pi - iota, phi0, kappa) is the source
        at (iota, phi0, -kappa), so F and SNR agree -- the symmetry the
        aligned-spin quadrature over cos(iota) in [0, 1] rests on."""
        kw = dict(psd=PSD, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                  aligned_spin=False)
        sig, tmp = with_negative_m(self.sig), with_negative_m(self.tmp)
        for iota, phi0, kappa in ((0.4, 0.3, 0.7), (1.2, 2.0, 2.5)):
            a = faithfulness_at_orientation(sig, tmp, iota, phi0, kappa, **kw)
            b = faithfulness_at_orientation(sig, tmp, math.pi - iota, phi0, -kappa, **kw)
            self.assertAlmostEqual(a['faithfulness'], b['faithfulness'], places=11)
            self.assertAlmostEqual(a['snr'] / b['snr'], 1.0, places=12)

    def test_half_range_quadrature_equals_the_full_one(self):
        """cos(iota) over [0, 1] (aligned_spin) and over [-1, 1] (explicit
        negative-m modes, no symmetry used) converge to the same U."""
        kw = dict(psd=PSD, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                  n_phase=6, n_polarization=4)
        U_half = sky_averaged_unfaithfulness(self.sig, self.tmp, aligned_spin=True,
                                             n_inclination=8, **kw)
        U_full = sky_averaged_unfaithfulness(with_negative_m(self.sig), with_negative_m(self.tmp),
                                             aligned_spin=False, n_inclination=16, **kw)
        self.assertLess(abs(U_half - U_full) / U_half, 1e-6)

    def test_settings_do_not_move_the_answer(self):
        U0 = sky_averaged_unfaithfulness(self.sig, self.tmp, **self.kw)
        for extra in (dict(tc_window=0.2), dict(tc_window=None), dict(oversample=8),
                      dict(n_template_phase=128)):
            U = sky_averaged_unfaithfulness(self.sig, self.tmp, **dict(self.kw, **extra))
            self.assertLess(abs(U - U0) / U0, 1e-6, msg=str(extra))

    def test_a_too_small_window_widens_itself(self):
        """A coalescence-time window far narrower than the peak's spread over
        orientations (0.5 ms against ~2.4 ms here) is widened until no
        orientation's best lag sits on its edge, and the answer is unchanged."""
        U0 = sky_averaged_unfaithfulness(self.sig, self.tmp, **self.kw)
        U = sky_averaged_unfaithfulness(self.sig, self.tmp, **dict(self.kw, tc_window=5e-4))
        self.assertLess(abs(U - U0) / U0, 1e-9)

    def test_angular_quadrature_converges(self):
        kw = dict(self.kw)
        U1 = sky_averaged_unfaithfulness(self.sig, self.tmp,
                                         **dict(kw, n_inclination=8, n_phase=8, n_polarization=8))
        U2 = sky_averaged_unfaithfulness(self.sig, self.tmp,
                                         **dict(kw, n_inclination=16, n_phase=16, n_polarization=16))
        self.assertLess(abs(U2 - U1) / U2, 1e-2)


def _toy_pair(item):
    """Module level, so that it pickles for a process pool."""
    f_end, d33 = item
    return toy_modes(), toy_modes(f_end=f_end, phases={(3, 3): d33}, drop=((4, 4),))


class TestMap(unittest.TestCase):

    def test_parallel_map_equals_serial(self):
        items = [(180.5, 0.1), (181.0, 0.3), (182.0, 0.0)]
        kw = dict(psd=PSD, low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
                  aligned_spin=True, n_inclination=2, n_phase=3, n_polarization=2)
        serial = sky_averaged_unfaithfulness_map(_toy_pair, items, processes=1, **kw)
        par = sky_averaged_unfaithfulness_map(_toy_pair, items, processes=2, **kw)
        numpy.testing.assert_allclose(par, serial, rtol=0, atol=1e-14)
        direct = [sky_averaged_unfaithfulness(*_toy_pair(it), **kw) for it in items]
        numpy.testing.assert_allclose(serial, direct, rtol=0, atol=0)


@unittest.skipUnless(SLOW, "set PYCBC_TEST_SLOW=1 to run tests with physical waveforms")
class TestPhysicalWaveforms(unittest.TestCase):
    """SEOBNRv5HM (pyseobnr) as the signal against IMRPhenomTHM (LAL) as the
    template, aligned spins, 60 solar masses, from 20 Hz."""

    @classmethod
    def setUpClass(cls):
        cls.sig, cls.tmp = physical_pair(q=2.0, chi1=0.3, chi2=0.1, total_mass=60.0)

    def test_finite_small_and_stable(self):
        kw = dict(psd=PSD, low_frequency_cutoff=20.0, high_frequency_cutoff=1024.0,
                  aligned_spin=True)
        U1 = sky_averaged_unfaithfulness(self.sig, self.tmp, n_inclination=8, n_phase=8,
                                         n_polarization=8, **kw)
        U2 = sky_averaged_unfaithfulness(self.sig, self.tmp, n_inclination=16, n_phase=16,
                                         n_polarization=16, **kw)
        self.assertTrue(numpy.isfinite(U1) and 0 < U1 < 0.05)
        self.assertLess(abs(U2 - U1) / U1, 1e-2)


def physical_pair(q, chi1, chi2, total_mass, f_lower=18.0, delta_t=DT):
    """(SEOBNRv5HM modes, IMRPhenomTHM modes), both tapered at the start, the
    four modes the two share with m > 0 -- (2,2), (2,1), (3,3), (4,4)."""
    import lal
    import lalsimulation as ls
    from pyseobnr.generate_waveform import GenerateWaveform
    m1 = total_mass * q / (1 + q)
    m2 = total_mass - m1
    keep = ((2, 2), (2, 1), (3, 3), (4, 4))
    settings = dict(mass1=m1, mass2=m2, spin1x=0., spin1y=0., spin1z=chi1, spin2x=0.,
                    spin2y=0., spin2z=chi2, deltaT=delta_t, f22_start=f_lower, f_ref=f_lower,
                    distance=1.0, inclination=0.0, approximant='SEOBNRv5HM',
                    ModeArray=list(keep))
    _, hlm = GenerateWaveform(settings).generate_td_modes()
    sig = {lm: TimeSeries(numpy.asarray(hlm[lm]), delta_t=delta_t) for lm in keep}
    hl = ls.SimInspiralChooseTDModes(0.0, delta_t, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0., chi1,
                                     0., 0., chi2, f_lower, f_lower, 1e6 * lal.PC_SI, None, 4,
                                     ls.IMRPhenomTHM)
    tmp = {}
    while hl is not None:
        if (hl.l, hl.m) in keep:
            tmp[(hl.l, hl.m)] = TimeSeries(numpy.asarray(hl.mode.data.data), delta_t=delta_t)
        hl = hl.next
    for d in (sig, tmp):
        for lm in d:
            re = TimeSeries(numpy.real(d[lm].numpy()), delta_t=delta_t).taper_timeseries('TAPER_START')
            im = TimeSeries(numpy.imag(d[lm].numpy()), delta_t=delta_t).taper_timeseries('TAPER_START')
            d[lm] = TimeSeries(re.numpy() + 1j * im.numpy(), delta_t=delta_t)
    return sig, tmp


if __name__ == '__main__':
    unittest.main()
