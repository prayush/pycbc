"""The SNR-weighted, sky-and-polarization-averaged unfaithfulness of
arXiv:2303.18046 (Eq. 36): what leaving out the higher harmonics of an
aligned-spin binary costs, averaged over the source's orientation."""
import lal
import lalsimulation as ls
from pycbc.types import TimeSeries
from pycbc.filter.faithfulness import sky_averaged_unfaithfulness

delta_t = 1.0 / 4096
m1, m2 = 40.0, 20.0
hl = ls.SimInspiralChooseTDModes(0.0, delta_t, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                                 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 18.0, 18.0,
                                 1e6 * lal.PC_SI, None, 4, ls.IMRPhenomTHM)
modes = {}
while hl is not None:
    if hl.m > 0:
        # start-tapered, as every waveform given to the function should be
        re = TimeSeries(hl.mode.data.data.real, delta_t=delta_t).taper_timeseries('TAPER_START')
        im = TimeSeries(hl.mode.data.data.imag, delta_t=delta_t).taper_timeseries('TAPER_START')
        modes[(hl.l, hl.m)] = TimeSeries(re.numpy() + 1j * im.numpy(), delta_t=delta_t)
    hl = hl.next

# the signal with every mode, the template with the (2, 2) alone; the m < 0
# modes follow from the aligned-spin symmetry
U, details = sky_averaged_unfaithfulness(
    modes, {(2, 2): modes[(2, 2)]}, psd='aLIGOZeroDetHighPower',
    low_frequency_cutoff=20.0, high_frequency_cutoff=2048.0, aligned_spin=True,
    return_details=True)
print('SNR-weighted unfaithfulness without the higher harmonics: %.2e' % U)
print('plain orientation average of the faithfulness:            %.6f'
      % details['faithfulness_mean'])
