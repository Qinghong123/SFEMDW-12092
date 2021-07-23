"""Defines PtcEvent class that performs power-to-contact search.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   collections import namedtuple
from   component.pattern import Track
from   core.data import DataError, DatumPack
import numpy as np
import production.base.metric as metric
from   production.base.event_pack import PackEvent
from   production.mixin.search import Search, SearchConfig, SearchError
import pynative
import system.path as path
from   system.wrap import chn, preamp
import time
from   typing import ClassVar, Dict, List, Tuple
from   util.control import KalmanFilter
from   util.dc_tcr import DctcrUtils

class PtcError(PackEvent.Error, SearchError):
    """Base class for PTC event errors"""

class PtcConfig(PackEvent.Config, SearchConfig):
    """ Defines DC DETCR PTC event configuration.
    Parameters
    ----------
    clean_amp: float
        dRdP amplitude threshold for triggering clean up
    clean_enable        : Tuple[bool, ...]
        flags to enable/disable cleaning for each track
    clean_offset: float
        the maximum heat offset to ptc for cleaning
    clean_time: float
        the time to dwell heads on disc for cleaning
    clean_rounds: int
        the max number of clean up rounds
    clean_tracks: tuple
        the range of clean up tracks in forms of [low, high]
    cont_bkoff: float
        the heat to be deducted from previous contact heat to get start heat
    cont_cutoff: float
        the heat to start contact detection
    cont_mode: int
        bit map of contact trigger enable flag; bit_1 = gap nsigma;
        bit_2 = amp slope; bit 3 = phase slope
    cont_nsigma: float
        the nsigma of prediction error of gap signal to trigger contact
    cont_slope: float
        the threshold of dRdP amplitude slope to trigger contact
    detcr_hp_hz: int
        the high-pass filter freq. for DETCR at contact detection mode
    detcr_lp_hz: int
        the low-pass filter freq. for DETCR at contact detection mode
    drdp_limit: float
        the lower limit of dRdP amplitude to trigger contact if no other
        triggers
    neg_pslope: float
        the lower limit of phase slope to trigger contact
    peak_nm: float
        the oscillation peak of sine wave in unit of nm
    pos_pslope: float
        the upper limit of phase slope to trigger contact
    prescan_mode: bool
        a flag to indicate the operation mode
    standalone: bool
        a flag indicating the event's operating mode; if False, prescan will
        be coupled with ptc; Note that ptc-prescan and ptc events should have
        identical setting on this.
    target_detcr_volts: float
        Detcr bias voltage to be used for detcr current calculation

    """
    clean_amp           : float
    clean_enable        : Tuple[bool, ...]
    clean_offset        : float
    clean_rounds        : int
    clean_time          : float
    clean_tracks        : Tuple[Track, ...]
    cont_bkoff          : float
    cont_cutoff         : float
    cont_mode           : int
    cont_nsigma         : float
    cont_slope          : float
    detcr_hp_hz         : int
    detcr_lp_hz         : int
    drdp_limit          : float
    neg_pslope          : float
    peak_nm             : float
    pos_pslope          : float
    prescan_mode        : bool
    standalone          : bool
    target_detcr_volts  : float

    # TODO: put following signal modes to json files SFEMDW-10162
    DC_CANCEL_MODE   : ClassVar[int]        = 4     # TCR_IBias=1;
    DCTCR_GAIN_MODES : ClassVar[Tuple[int]] = 3, 4  # TCR_IBias=1; db increasing order
    DETCR_BIAS_MODE  : ClassVar[int]        = 3     # TCR_IBias=1; TCR_High_Bias=0

    ADC_TOLERANCE    : ClassVar[int]   = 20                # Max allowed ADC offset from target
    CLEAN_PTC_DELTA  : ClassVar[int]   = 6
    FIT_COUNT        : ClassVar[int]   = 10
    MAX_CLEAN_TIME   : ClassVar[float] = 0.5
    MAX_TCR_BIAS     : ClassVar[float] = 2.82e-3
    MAX_WAVE         : ClassVar[int]   = 56
    MIN_CLEAN_OFFSET : ClassVar[int]   = 5
    NSIGMA           : ClassVar[int]   = 1                 # 3-bit Contact Mode: bit_1 = gap nsigma
    NUM_REVS         : ClassVar[int]   = 1
    OSC_FREQ         : ClassVar[float] = 1e3
    PHASE            : ClassVar[int]   = 4                 # 3-bit Contact Mode: bit 3 = phase slope
    PTC_PATTERN      : ClassVar[str]   = 'chp8pat.asm'
    SATURATION       : ClassVar[int]   = 9999              # when ADC reading is 255
    SLOPE            : ClassVar[int]   = 2                 # 3-bit Contact Mode: bit_2 = amp slope
    STARVATION       : ClassVar[int]   = -9999             # when ADC reading is 0
    WARMUP_OFFSET    : ClassVar[int]   = -5
    WARMUP_REV       : ClassVar[float] = 0.5
    XSRISC_ASM       : ClassVar[str]   = 'XsdRdP.asm'

    # Kalman filter settings
    KF_PARAMS : ClassVar[Tuple[List[float], ...]] = ([1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0],
                                                     [3.0, 3.0, 3.0])
    # t-score for 95% single tail
    T95 : ClassVar[Dict[int, float]] = \
        dict(zip(range(3, 31),
                 [2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833,
                  1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740,
                  1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708,
                  1.706, 1.703, 1.701, 1.699, 1.697]))

    # pylint:disable=attribute-defined-outside-init
    def _compile(self):
        """Initialize configuration. """
        if self.head_spec.heater == 'writer':
            self.heater_signal = preamp.DAC_HEATER_WRITER_POWER_DURING_READ
        else:
            raise PtcError('No support for reader heater now!')
        if self.head_spec.hamr:
            # First zone must be ID zone for head cleaning
            if self.tracks[0] != max(self.tracks):
                raise PtcError('HAMR PTC first zone must be ID zone for cleaning!')
            # Clean up time check
            if self.clean_time > self.MAX_CLEAN_TIME:
                raise PtcError(f'Head crash risk high with large clean up time: {self.clean_time}')
        self.metrics_in = (metric.DETCR_RESISTANCE,)
        if self.prescan_mode:
            self.metrics_out = (metric.EARLY_RAMP_SLOPE,)
        else:
            self.metrics_out = (metric.HEATER_WRITER_BIAS.tag('ptc', self.rpm),
                                metric.DETCRDC_CANCEL)
        # Convert real unit to DAC using heat slope
        ht_slope = (preamp.DacMapDigital(0, self.heater_signal, 0, 2) -
                    preamp.DacMapDigital(0, self.heater_signal, 0, 1))

        # Convert search range parameters from watts to dac
        self.dac_range = []
        for idx, value in enumerate(self.search_range):
            if idx < 2:
                # Convert to absolute heat DACs
                dac, _ = preamp.DacMapAnalog(0, self.heater_signal, 0, value)
            else:
                # relative DAC value only uses ht_slope to convert
                dac = int(value / ht_slope + 0.5)
            self.dac_range.append(dac)

        self.cont_cutoff_dac, _ = preamp.DacMapAnalog(0, self.heater_signal,
                                                      0, self.cont_cutoff)
        # Relative heat values
        bkoff_dac = int(self.cont_bkoff / ht_slope + 0.5)
        self.cont_bkoff_dac = max(bkoff_dac, self.FIT_COUNT * self.dac_range[-1])
        self.peak_dac = int(self.peak_nm*self.head_spec.hms_slope/ht_slope + 0.5)

        self.temp_file = path.cell.ramPath('ptc_tmp.dat')
        # flags indicating operation mode
        self.need_setup = self.standalone or self.prescan_mode
        self.need_restore = self.standalone or not self.prescan_mode

class Ptc(PackEvent, Search):
    """Defines base class for PTC calibration events. """

    Config = PtcConfig
    Error = PtcError

    # pylint:disable=attribute-defined-outside-init

    # Status code indicating head status
    DctcrStatus = namedtuple('DctcrStatus',
                             'search dcc_err saturation starvation early_ramp '
                             'nsigma slope phase limit')
    STATUS = DctcrStatus(*DctcrStatus._fields)

    # Oscillation ADC results
    OscData = namedtuple('OscData', 'heat amp phase gap')
    # Data record fields
    PtcRecord = namedtuple('PtcRecord', 'heat amp phase gap kf_amp kf_phase '
                                        'kf_gap slope pslope nsigma dcc dctcr_gain')
    RECORD = PtcRecord(*range(len(PtcRecord._fields)))

    def _action(self):
        """Perform PTC search """
        if len(self._heads) == 0:
            return []
        heads = self._heads
        clean_heads = []
        for self.clean_round in range(self.config.clean_rounds + 1):
            if clean_heads:
                self._cleanHeads(clean_heads)
                heads = clean_heads
            # Perform iterative search over search range
            start_heats = self._getStartHeats(heads)
            self._search(start_heats)
            # Clean heads if needed
            clean_heads = self._needClean(heads)
            if not clean_heads:
                break
        result = self._getDatums()
        # Record last track
        self.last_track = self._track
        return result

    def _acquire(self, x_values):
        """Acquires data for the given parameter values.
        Parameters
        ----------
        x_values: dict
            search variable values in forms of {head:value}

        Returns
        --------
        dict
            a dictionary of {head:named_tuple}
        """
        # Acquire ADC values
        adc_handle = self._sample(x_values)

        # Process data in forms of {head:(amp, phase, gap)}
        if pynative.localmode_q(): data = dict((hd, [0.0, 0.0, 0.0]) for hd in x_values)
        else:                      data = pynative.getchirpadc(adc_handle)

        result = {}
        for hd, values in data.items():
            result[hd] = self.OscData(x_values[hd], *values)

        return result

    def _adjustGain(self, head, direction):
        """Adjust DC DETCR Gain to accomodate target ADC value.

        Parameters
        ----------
        head: int
            head number
        direction: int
            the direction of gain adjustment: 1=increase, -1=decrease

        Returns
        -------
        bool
            whether gain is found to accommodate target_dcc

         """
        gain_signal = preamp.DAC_DETCRDC_GAIN
        mode, gain = preamp.DacRead(head, gain_signal)
        valid_modes = self.config.DCTCR_GAIN_MODES
        if mode not in valid_modes:
            raise PtcError(f'Invalid DC DETCR Gain mode {mode}')
        amin, amax, dmax, _ = preamp.DacGetRange(0, gain_signal, mode)
        # Indicator of adjust direction
        if direction == -1:
            if gain == 0:
                # Change mode if possible;
                if mode == valid_modes[0]:
                    self.log.debug(f'_adjustGain: Hd={head}: minimum gain '
                                   'reached, stop')
                    return False
                idx = valid_modes.index(mode)
                mode = valid_modes[idx - 1]
                gain = dmax
            else:
                gain -= 1
        else:
            # signal too weak, increase gain
            if gain == dmax:
                # Change mode with higher gain range
                if mode == valid_modes[-1]:
                    self.log.debug(f'_adjustGain: Hd={head}: maximum gain'
                                   'reached, stop')
                    return False
                idx = valid_modes.index(mode)
                mode = valid_modes[idx + 1]
                gain = 0
            else:
                gain += 1
        preamp.DacWrite(head, gain_signal, mode, gain)
        self.log.debug(f'_adjustGain: Hd={head} DCDetcrGain={gain} Mode={mode}')
        self.dctcr_gain[head] = gain
        return True

    def _calcCleanUpHeats(self, heads):
        clean_heats = {}
        msg = f'Clean up heat for round {self.clean_round}:\n '
        for head in heads:
            record = self._getRecordAtMinAmp(head)
            cont_heat = record[self.RECORD.heat]
            delta_amp = record[self.RECORD.amp] - self.config.clean_amp
            pch = self.predicted_ptc[head]
            if self.config.clean_amp < 1e-3:
                ratio = 1
            else:
                ratio = min(delta_amp / self.config.clean_amp, 1.0)
            delta_heat = int(ratio * self.config.clean_offset + 0.5)
            heat = int(pch + max(delta_heat, self.config.MIN_CLEAN_OFFSET))
            clean_heats[head] = min(heat, self.config.dac_range[1] - self.config.peak_dac)
            msg += f'Hd={head} ContHeat={cont_heat} PredictedCH={pch} CuHeat={clean_heats[head]}\n'
        self.log.debug(msg)
        return clean_heats

    def _calDcc(self, heats, target=127):
        """Adjust DC Cancel so that the measured ADC meets target range

        Parameters
        ----------
        heats: dict
            the dictionary of {head: heat} for DCC cal setup
        target: int
            target ADC value; usually close to middle of range

        Returns
        -------
        tuple
            head dictionaries of dcc_values and gain adjust flags

        """
        heads = list(heats.keys())
        result = {hd: None for hd in heads}
        gain_adjusted = {hd: False for hd in heads}
        _, _, max_dcc, _ = preamp.DacGetRange(0, preamp.DAC_DETCRDC_CANCEL,
                                              self.config.DC_CANCEL_MODE)
        for hd in heads:
            if hd not in self.dcc:
                dcc_func = self._searchDccFast
            else:
                dcc_func = self._searchDcc
            # Set heat
            preamp.DacWrite(hd, self.config.heater_signal, 0, heats[hd])
            direction, dcc = dcc_func(hd, heats[hd], max_dcc, target)
            last_direction = None
            count = 0
            while direction != 0:
                gain_adjusted[hd] = True
                if ((last_direction is not None and
                     direction != last_direction) or count > 30):
                    # Loop detected, stop
                    self.log.debug(f'calDcc: Hd={hd} ->loop detected')
                    dcc = -1
                    break
                if not self._adjustGain(hd, direction):
                    # Fail to adjust gain
                    dcc = -1
                    break
                last_direction = direction
                direction, dcc = dcc_func(hd, heats[hd], max_dcc, target)
                count += 1
            result[hd] = dcc
        return result, gain_adjusted

    def _checkContact(self, head):
        """Check if head triggers contact.

        Returns
        -------
        str
            action name for the next step; either go or finish
        """
        result = self.ACTION.go
        triggers = []
        for trigger_bit, trigger_func in zip(
                [self.config.NSIGMA, self.config.SLOPE, self.config.PHASE],
                [self._chkNSigma, self._chkSlope, self._chkPhase]):
            if self.config.cont_mode & trigger_bit:
                trigger = trigger_func(head)
                if trigger:
                    triggers.append(trigger)
        if (not triggers and
                self._data[head][-1][self.RECORD.amp] <= self.config.drdp_limit):
            triggers.append(self.STATUS.limit)
        if triggers:
            result = self.ACTION.finish
            self.status[head] = '_'.join(triggers)
        return result

    def _chkNSigma(self, head):
        """Check if gap signal triggers error nsigma"""
        n = len(self._data[head])
        if n > self.config.FIT_COUNT:
            hd_data = np.array(self._data[head])[-self.config.FIT_COUNT - 1:]
            x = hd_data[:-1, self.RECORD.heat]
            # Use measured gap
            y = hd_data[:-1, self.RECORD.gap]
            coeff, _, sigma = self._lstSqFit(x, y, order=1, mode=1)
            # calculate last data point projection error
            gap = hd_data[-1][self.RECORD.gap]
            heat = hd_data[-1][self.RECORD.heat]
            err = gap - (coeff[0] + coeff[1] * heat)
            nsigma = err / max(0.5, sigma)
            # Replace kf velocity for gap signal with nsigma
            self._data[head][-1][self.RECORD.nsigma] = nsigma
            if nsigma >= self.config.cont_nsigma:
                return self.STATUS.nsigma
        return ''

    def _chkPhase(self, head):
        """Check if amp signal triggers slope"""
        n = len(self._data[head])
        if n >= self.config.FIT_COUNT:
            hd_data = np.array(self._data[head])[-self.config.FIT_COUNT:]
            x = hd_data[:, self.RECORD.heat]
            # Use filtered signal for slope calculation
            y = hd_data[:, self.RECORD.kf_phase]
            coeff, _, _ = self._lstSqFit(x, y, order=2)
            slope = 2 * coeff[2] * x[-1] + coeff[1]
            # Save data
            self._data[head][-1][self.RECORD.pslope] = slope
            if slope >= self.config.pos_pslope or slope <= self.config.neg_pslope:
                return self.STATUS.phase
        return ''

    def _chkSlope(self, head):
        """Check if amp signal triggers slope"""
        n = len(self._data[head])
        if n >= self.config.FIT_COUNT:
            hd_data = np.array(self._data[head])[-self.config.FIT_COUNT:]
            x = hd_data[:, self.RECORD.heat]
            # Use filtered signal for slope calculation
            y = hd_data[:, self.RECORD.kf_amp]
            coeff, _, _ = self._lstSqFit(x, y, order=2)
            slope = 2 * coeff[2] * x[-1] + coeff[1]
            # Save data
            self._data[head][-1][self.RECORD.slope] = slope
            if slope >= self.config.cont_slope:
                return self.STATUS.slope
        return ''

    def _cleanHeads(self, heads):
        """Move the heads to specified clean zone and dwell on disc at the given heat levels
        for a certain amount of time.
        Parameters
        ----------
        heads: list
            the list of heads to be cleaned

        """
        # Calculate clean up heat
        heats = self._calcCleanUpHeats(heads)
        # Set clean up revs and offsets
        track = np.random.randint(self.config.clean_tracks[0], self.config.clean_tracks[1])
        self.log.debug(f'Moving to track {track} for cleaning...')
        self._moveToTrack(track)
        # Clean with writer heater
        try:
            for head in heads:
                preamp.DacWrite(head, self.config.heater_signal, 0, heats[head])
                time.sleep(self.config.clean_time)
                preamp.DacWrite(head, self.config.heater_signal, 0, 0)
        except Exception:
            for head in heads:
                preamp.DacWrite(head, self.config.heater_signal, 0, 0)
        # Update clean up status and roll back heat to redo ptc
        for head in heads:
            self.last_result[head] = self._result[head]
            self._clearData(head)
            self.status[head] = self.STATUS.search
            self.dcc.pop(head)
        self._moveToTrack(self._track)

    def _clearData(self, head):
        """Remove existing data in the case of re-calibrated DC Cancel
        and/or DC DETCR gain

        Parameters
        ----------
        head: int
            a head number

        """
        self._data[head].clear()

    def _decide(self, heads):
        """Check if contact is found or if there is error

        Parameters
        ----------
        heads: list
            a list of head numbers

        Returns
        -------
        dict
            dictionary of {head:action} where action is in self.ACTION
        """
        result = {head: self.ACTION.go for head in heads}
        heats = self.iterator.query(heads)
        for head, heat in heats.items():
            if self.status[head] in [self.STATUS.saturation, self.STATUS.starvation]:
                result[head] = self.ACTION.retry
            elif self.status[head] == self.STATUS.dcc_err:
                result[head] = self.ACTION.stop
            else:
                if not self.config.prescan_mode:
                    # Continue check contact
                    if heat > self.config.cont_cutoff_dac:
                        # Check contact
                        result[head] = self._checkContact(head)
        return result

    def _exitEventState(self):
        """Perform wrap up operations"""

        if self.config.need_restore:
            self._tearDown()
        if self.config.debug:
            host_file = path.host.join(self.getPathHostData(), self.name, f'{self.time_stamp}_ptc_data.csv')
            self._uploadFile(self.config.temp_file, host_file)

    def _finishAction(self, heads):
        """Terminate heads and save results."""
        self.iterator.terminate(heads)
        for hd in heads:
            cont_heat = self._result[hd] = int(self._getRecordAtMinAmp(hd)[self.RECORD.heat])
            if self._track == self.config.tracks[0]:
                # calculate pch for each head for the first zone(ID)
                pch = self._predictPtc(hd, cont_heat)
                if pch is None or pch < cont_heat:
                    pch = cont_heat
                elif pch > cont_heat + self.config.clean_offset // 2:
                    pch = cont_heat + self.config.clean_offset // 2
                self.predicted_ptc[hd] = pch

    def _getAdc(self, head, dcc_value, heat=None):
        """Returns ADC for ADC signal detcr_voltage
        Parameters
        ----------
        head: int
            head number
        dcc_value: int
            DC Cancel value
        heat: int
            the heat DAC to use for dRdP sampling
        Returns
        -------
        ADC value (int) and dRdP sampling amp

        """
        preamp.DacWrite(head, preamp.DAC_DETCRDC_CANCEL,
                        self.config.DC_CANCEL_MODE, dcc_value)
        time.sleep(0.001)
        adc = preamp.AdcRead(head, preamp.ADC_DETCR_VOLTAGE)
        amp = 0.0
        if heat is not None:
            amp = self._acquire({head: heat})[head].amp
        return adc, amp

    def _getDatums(self):
        """Extract metrics_out datums for the given data.

         Returns
         -------
         list
            a list of datums for metrics out
         """
        result = []
        signal = self.config.heater_signal
        if self.config.prescan_mode:
            units = ['unitless']
            values = [self._result]
        else:
            units = 'watts,unitless'.split(',')
            values = [self._result, self.dcc]
        for name, data_metric, unit in zip(self.config.metrics_out, values, units):
            for head in self._heads:
                try:
                    value = data_metric[head]
                    if unit == 'watts':
                        # Convert DAC to watts
                        value = preamp.DacMapDigital(head, signal, 0, value)
                except KeyError:
                    value = None
                result.append(DatumPack(name, unit, head, self._track, value,
                                        source=self.name))
        return result

    def _getEarlyRampSlope(self, heads):
        """Perform linear fit of dRdP amp and categorize heads based on
        t-ratio of linear slope.
        Parameters
        ----------
        heads: list
            the list of head numbers

        Returns
        -------
        dictionary
            the early ramp slope by head
        """
        result = {}
        msg = ''
        for hd in heads:
            records = np.array(self._data[hd])
            x = records[:, self.RECORD.heat]
            y = records[:, self.RECORD.amp]
            coeff, res, _ = self._lstSqFit(x, y, 1, 0)
            # Calculate t-ratio
            slope = coeff[1]
            n = len(x)
            x_delta = x - x.mean()
            std_err = np.sqrt(res[0] / (n - 2) / sum(x_delta * x_delta))
            t_ratio = slope / std_err
            if t_ratio >= self.config.T95[n - 2]:
                msg += (f'Hd={hd} Slope={slope:.2f} T-ratio={t_ratio:.2f}:'
                        ' early ramp\n')
            elif t_ratio > 0:
                # Replace slope with a dummy value to avoid fake early ramp
                slope = -1e-3
            result[hd] = slope
        if msg:
            self.log.debug(f'Early ramp check:\n{msg}')
        return result

    def _getRecordAtMinAmp(self, head):
        """Get the heat where amp is minimum from the deque
        Record columns are:
            heat, amp, phase, gap, kf_amp, kf_phase, kf_gap, amp_slope,
            phase_slope, gap_nsigma

        Parameters
        ----------
        head: int
            the head number

        Returns
        -------
        the record with min amplitude

        """
        hd_data = np.array(self._data[head])[-self.config.FIT_COUNT:]
        amp_values = hd_data[:, self.RECORD.amp]
        return hd_data[amp_values == min(amp_values)][0]

    def _getStartHeats(self, heads):
        """Get the start heats based on previous contact heat results"""
        if self.config.prescan_mode:
            return None
        result = {hd: self.config.dac_range[0] for hd in heads}
        try:
            contact_heats = self._data_out.filtBy(metric=self.config.metrics_out[0])
        except DataError:
            pass
        else:
            for hd in heads:
                ch = []
                for d in contact_heats.filtBy(head=hd):
                    if d.value is not None and d.track != self._track:
                        ch.append(d.value)
                if ch:
                    dac, _ = preamp.DacMapAnalog(0, self.config.heater_signal, 0, np.mean(ch))
                    if self.clean_round:
                        # If cleaning has been done, then use previous contact heat
                        dac = self._result[hd]
                    start_heat = dac - self.config.cont_bkoff_dac
                    result[hd] = max(result[hd], start_heat)
        return result

    @staticmethod
    def _getTimeStamp():
        """Get time stamp """
        t = time.localtime()
        time_stamp = ''.join([f'{v:02d}' for v in
                              (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)])
        return time_stamp

    def _initEventState(self):
        """Initialize event"""
        self.time_stamp = self._getTimeStamp()
        self.dctcr_utils = DctcrUtils()  # TODO: replace by ESF APIs
        self.default_gain = 1
        if self.config.need_setup:
            self._loadPtcPattern()
            self._setupPreamp()
            self._setupXsRisc()
            self._setupChn()
        with open(self.config.temp_file, 'w') as fs:
            headers = ('prescan,track,head,clean_rounds,' + ','.join(self.PtcRecord._fields)
                       + ',status\n')
            fs.write(headers)
        # Only stores predicted ptc at ID
        self.predicted_ptc = {}

    def _initTrackState(self):
        """Reset ptc data for given track"""
        if self._track != self.config.tracks[0] or not self.config.need_setup:
            # Reset dRdP if this is not the first time chn chirp API is called
            chn.ChirpResetAdc()

        self.log.debug('Chn Chirp API reset done')
        # Reset DC DETCR Gain in case it is changed during last DCC cal
        default_mode = self.config.DCTCR_GAIN_MODES[-1]
        for head in self._heads:
            preamp.DacWrite(head, preamp.DAC_DETCRDC_GAIN, default_mode,
                            self.default_gain)
        self.log.debug(f'DC DETCR Gain set to {self.default_gain} with '
                         f'mode {default_mode}')
        # Initialize ptc data
        self._initSearch(self._heads)
        # Initialize clean up data
        self.last_result = {head: 0 for head in self._heads}

    def _initSearch(self, heads):
        """Initialize search data"""
        super()._initSearch(heads)
        self.dcc = {}
        self.dctcr_gain = {}
        self.kf = {hd: KalmanFilter(*self.config.KF_PARAMS) for hd in heads}
        self.status = {head: self.STATUS.search for head in heads}

    def _isContactFound(self, head):
        """Return True if contact is found"""
        ST = self.STATUS
        cont_triggers = [ST.nsigma, ST.slope, ST.phase, ST.limit]
        for status in self.status[head].split('_'):
            if status in cont_triggers:
                return True
        return False

    def _loadPtcPattern(self):
        """Load dedicated PTC pattern"""
        self._loadAsm(self.config.PTC_PATTERN)
        # Calculate pattern-related parameters
        self.nsector = self.config.patt_spec.getParams('number_of_sectors')
        # rev_time in seconds
        rev_time = self.config.patt_spec.getParams('rev_time')
        # sample_freq in Hz
        sample_freq = self.nsector / rev_time
        samples = self.config.NUM_REVS * self.nsector
        # Number of samples per sine wave to achieve osc_freq
        n = int(sample_freq / self.config.OSC_FREQ + 0.5)
        # Upper bound by scratch pad space
        n = min(n, self.config.MAX_WAVE)
        # Upper bound by total number of cycles must be greater than 3
        n = min(n, int(samples / 3 + 0.5))
        # Lower bound by 16
        n = max(16, n)
        # n need to be multiples of 2
        self.smp_per_cycle = n + n % 2
        osc_freq = sample_freq / self.smp_per_cycle
        self.log.debug('\nHeat oscillation wave parameters: \n'
                         f'Peak={self.config.peak_dac} '
                         f'Length={self.smp_per_cycle} '
                         f'Samples={samples} '
                         f'OscFreq={osc_freq / 1000.0:.2f} KHz')

    def _logData(self, heads):
        """Logs latest PTC data for the given heads."""
        info_str = f'{self.config.prescan_mode},{self._track}'
        with open(self.config.temp_file, 'a') as fs:
            for hd in heads:
                if self.status[hd] in [self.STATUS.saturation, self.STATUS.starvation]:
                    # No data for overflow
                    continue
                try:
                    record = self._data[hd][-1]
                except IndexError:
                    continue
                line = info_str + f',{hd},{self.clean_round},'
                # first element is heat which is in DAC
                line += ','.join([f'{v:.2f}' if 0 < i < self.RECORD.dcc
                                  else str(int(v)) for i, v in enumerate(record)])
                line += f',{self.status[hd]}\n'
                fs.write(line)

    @staticmethod
    def _lstSqFit(x, y, order=1, mode=0):
        """Returns the linear least square fit coefficients, residual and
        the error standard deviation.
        Parameters
        ----------
        x: sequence
            the sequence of x values
        y: sequence
            the sequence of y values
        order: int
            the order of polynomial; 1=linear, 2=quadratic
        mode: int
            1= returns sigma of prediction error; if 0, return 0 sigma

        Returns
        -------
        tuple
            Coefficients, residue, sigma of error
        """
        sigma = 0
        # Obtain linear least square fit
        if order == 1:
            x_ = np.array([[1, v] for v in x])
        else:
            x_ = np.array([[1, v, v * v] for v in x])
        y_ = np.array(y)
        coeff, res, _, _ = np.linalg.lstsq(x_, y_, rcond=None)
        if mode:
            # Calculate projection errors
            errors = y_ - np.dot(x_, coeff)
            # Calculate stddev
            sigma = np.std(errors)
        return coeff, res, sigma

    def _needClean(self, heads):
        """Determine if given heads need cleaning.
        Parameters
        ----------
        heads: list
            a list of head numbers

        Returns
        -------
        a list of heads to be cleaned

        """
        result = []
        if self.config.clean_enable[self.config.tracks.index(self._track)]:
            for head in heads:
                if self._isContactFound(head):
                    record = self._getRecordAtMinAmp(head)
                    if (record[self.RECORD.amp] > self.config.clean_amp and
                            self._result[head] - self.last_result[head] > self.config.CLEAN_PTC_DELTA):
                        result.append(head)
        return result

    def _predictPtc(self, head, cont_heat):
        """Predict PTC by linear regression"""
        result = None
        data = np.array(self._data[head])
        x = data[:, self.RECORD.heat]
        y = data[:, self.RECORD.amp]
        flags = x <= cont_heat
        x = x[flags]
        y = y[flags]
        if len(x) >= 5:
            C, _, _ = self._lstSqFit(x, y)
            if abs(C[1]) > 1e-5:
                result = max(x[-1], (self.config.clean_amp - C[0]) / C[1])
        return result

    def _postProcess(self, data):
        """Post-process data and store results in self._data.

        Parameters
        ----------
        data: dict
            contains the head data in forms of {head:OscData}

        """
        ST = self.STATUS
        overflow = {self.config.SATURATION: ST.saturation,
                    self.config.STARVATION: ST.starvation}
        for hd, osc_data in data.items():
            if osc_data.amp in overflow:
                if self.status[hd] in overflow:
                    # DCC cal failed, terminate head
                    self.status[hd] = ST.dcc_err
                else:
                    # Indicate ADC overflow, need redo DCC
                    self.status[hd] = overflow[osc_data.amp]
            else:
                self.status[hd] = ST.search
                kf = self.kf[hd]
                if kf.isEmpty():
                    # Initialize state
                    kf.init(osc_data)
                else:
                    kf.predict(osc_data[0])
                    kf.update(osc_data)
                # Save both x, z, y, v and dcc, dctcr gain
                dcc = self.dcc.get(hd, -1)
                dctcr_gain = self.dctcr_gain.get(hd, -1)
                record = np.concatenate((osc_data, kf.state[1], kf.state[2],
                                         (dcc, dctcr_gain)))
                self._data[hd].append(record)

    def _retryAction(self, heads):
        """Recalibrate DC Cancel and backoff the heat"""
        # Note self.iterator.values contain the value for the next step
        # So if we want to offset 10 units relative to last step, we need to
        # minus one extra step
        heats = self.iterator.query(heads)
        dcc_values, gain_adjusted = self._calDcc(heats)
        self.dcc.update(dcc_values)
        offsets = {}
        fail_hds = []
        for hd, dcc in dcc_values.items():
            if dcc == -1:
                # Mark head as DCC failure
                self.status[hd] = self.STATUS.dcc_err
                fail_hds.append(hd)
            else:
                self.status[hd] = self.STATUS.search
                if gain_adjusted[hd]:
                    offsets[hd] = -(self.config.FIT_COUNT + 1) * self.iterator.step
                    # Exclude the previous data points from contact detection
                    self._clearData(hd)
                else:
                    offsets[hd] = -self.iterator.step
        if offsets:
            self.iterator.offset(offsets)
        if fail_hds:
            self._stopAction(fail_hds)

    @staticmethod
    def _sample(heats):
        """Sample ADC with heat oscillation around given heat values"""
        values = [f'{head},{heat}' for head, heat in heats.items()]
        cmd_str = ','.join(values)

        # TODO: will go away after SFEMDW-9826 completes
        return None if pynative.localmode_q() else chn.ChirpAcquireAdc(cmd_str)

    def _search(self, starts=None):
        """Perform PTC search

        Parameters
        ----------
        starts: dict
            a dictionary of {head:start_value}

        Returns
        -------
        dict
            the results found in forms of {head: value}
        """

        if starts is not None:
            heads = list(starts.keys())
        else:
            heads = self._heads
        self.iterator = self.Iterator(heads, *self.config.dac_range)
        self.log.debug(f'Starting PTC search at track {self._track}: clean up round {self.clean_round}')
        # Turn on heater circuit
        for head in heads:
            preamp.CircuitEnable(head, 'heater', 1)
        try:
            is_first = True
            start_heats = starts
            for heats in self.iterator:
                if start_heats is not None:
                    self.iterator.jump(start_heats)
                    start_heats = None
                    # Restart from given start heats
                    continue
                if is_first:
                    # Cal DCC first
                    self.dcc.update(self._calDcc(heats)[0])
                    is_first = False
                with self._stepState(heats):
                    self._doStep(heats)
            if self.config.prescan_mode:
                # Calculate early ramp slope for prescan
                self._result = self._getEarlyRampSlope(heads)
        except Exception:
            # Turn off heater circuit on error
            for hd in heads:
                preamp.CircuitEnable(hd, 'heater', 0)
            raise
        finally:
            msg = ''
            for hd in heads:
                try:
                    value = self._result[hd]
                except KeyError:
                    value = None
                msg += f'Head={hd} Result={value} Status={self.status[hd]}\n'
            if msg:
                self.log.debug(f'Prescan={self.config.prescan_mode}'
                                 f' Trk={self._track}:\n{msg}')

        return self._result

    def _searchDcc(self, head, heat, max_dcc, target):
        """ Perform binary search of DCC to bring ADC to the target range.

        Parameters
        ----------
        head: int
            the head number
        heat: int
            the heat dac for dRdP sampling
        max_dcc: int
            the maximum DCC value
        target: int
            the target ADC; usually in the middle of ADC range

        Returns
        -------
        tuple
            return direction, dcc where direction indicates DC DETCR Gain
            adjustment direction
        """
        if self.status[head] == self.STATUS.saturation:
            direction = 1
            last_amp = self.config.SATURATION
        else:
            direction = -1
            last_amp = self.config.STARVATION
        _, dcc = preamp.DacRead(head, preamp.DAC_DETCRDC_CANCEL)
        gain_adjustment = 0
        while True:
            adc, amp = self._getAdc(head, dcc, heat)

            if amp == self.config.SATURATION:
                if direction == 1:
                    # increase dcc
                    dcc += 1
                    if dcc > max_dcc:
                        # signal too strong, reduce gain
                        gain_adjustment = -1
                        break
                else:
                    # Stop, dcc bounce up => use last dcc if amp is good
                    if last_amp != self.config.STARVATION:
                        # Reset to last DCC
                        dcc += 1
                        preamp.DacWrite(head, preamp.DAC_DETCRDC_CANCEL,
                                        self.config.DC_CANCEL_MODE, dcc)
                    else:
                        # signal too strong, reduce gain
                        gain_adjustment = -1
                        self.status[head] = self.STATUS.saturation
                    break
            elif amp == self.config.STARVATION:
                if direction == -1:
                    # decrease dcc
                    dcc -= 1
                    if dcc < 0:
                        # Signal too weak, increase gain
                        gain_adjustment = 1
                        break
                else:
                    # Stop, dcc bounce down => use last dcc if amp is good
                    if last_amp != self.config.SATURATION:
                        # Reset to last DCC
                        dcc -= 1
                        preamp.DacWrite(head, preamp.DAC_DETCRDC_CANCEL,
                                        self.config.DC_CANCEL_MODE, dcc)
                    else:
                        # signal too strong, reduce gain
                        gain_adjustment = -1
                        # Update status
                        self.status[head] = self.STATUS.starvation
                    break
            elif adc > target + self.config.ADC_TOLERANCE:
                if direction == 1:
                    dcc += 1
                else:
                    # Stop
                    break
            elif adc < target - self.config.ADC_TOLERANCE:
                if direction == 1:
                    break
                else:
                    dcc -= 1
            else:
                break
            last_amp = amp
        if gain_adjustment:
            self.log.debug(f'calDcc: Hd={head} Heat={heat} DCC={dcc} '
                           f'ADC={adc} dRdP={amp:.1f}: target not found\n')
        return gain_adjustment, dcc

    def _searchDccFast(self, head, heat, max_dcc, target):
        """ Perform binary search of DCC to bring ADC to the target range.

        Parameters
        ----------
        head: int
            the head number
        heat: int
            the current heat for ADC reading
        max_dcc: int
            the maximum DCC value
        target: int
            the target ADC; usually in the middle of ADC range

        Returns
        -------
        tuple
            return direction, dcc where direction indicates DC DETCR Gain
            adjustment direction
        """
        dcc_lb = 0
        dcc_ub = max_dcc
        dcc = -1
        direction = 0
        while dcc_ub - dcc_lb > 1:
            dcc = (dcc_ub + dcc_lb) // 2
            adc, _ = self._getAdc(head, dcc)
            if adc > target + self.config.ADC_TOLERANCE:
                # increase dcc
                dcc_lb = dcc
            elif adc < target - self.config.ADC_TOLERANCE:
                # decrease dcc
                dcc_ub = dcc
            else:
                # Calibrate finished
                self.log.debug(f'calDcc: Hd={head} Heat={heat} DCC={dcc} '
                               f'ADC={adc}: target found\n')
                break
        else:
            self.log.debug(f'calDcc: Hd={head} Heat={heat} DCC={dcc} '
                           f'ADC={adc}: target not found\n')
        if adc < self.config.ADC_TOLERANCE:
            # signal too weak => increase gain
            direction = 1
        elif adc > 255 - self.config.ADC_TOLERANCE:
            # signal too strong => decrease gain
            direction = -1
        return direction, dcc

    def _setDetcrBias(self):
        """Calculate and set Detcr Bias amps """
        mode = self.config.DETCR_BIAS_MODE
        bias_signal = preamp.DAC_DETCR_BIAS
        bias_volts = self.config.target_detcr_volts
        msg = f'Setting DETCR Bias for target voltage {bias_volts}\n'
        for row in self._data_in.filtBy(head=self._heads):
            # calculate bias amps
            bias_amps = min(bias_volts / row.value, self.config.MAX_TCR_BIAS)
            bias_dac, _ = preamp.DacMapAnalog(row.head, bias_signal, mode,
                                              bias_amps)
            preamp.DacWrite(row.head, bias_signal, mode, bias_dac)
            msg += f'Head={row.head} DetcrBiasDac={bias_dac}\n'
        self.log.debug(msg)

    def _setupChn(self):
        """Setup Chn Chirp API assuming ptc pattern is loaded."""
        args = f'{self.config.NUM_REVS},{self.nsector},{self.smp_per_cycle},' \
               f'SIN,{self.config.peak_dac}'
        chn.SetupChirp(args)
        self.log.debug('Chn Chirp API setup done')

    def _setupPreamp(self):
        preamp.Reset()
        self._enableHeads()
        self._setDetcrBias()
        circuits = [preamp.CIRCUIT_DETCR, preamp.CIRCUIT_DETCRDC,
                    preamp.CIRCUIT_HEATER]
        # DAC signals to be written to preamp [signal, analog_value]
        cfg = self.config
        dac_signals = []
        for dac_signal, value in [(preamp.DAC_DETCR_HIGH_FREQ_CUTOFF, cfg.detcr_lp_hz),
                                  (preamp.DAC_DETCR_LOW_FREQ_CUTOFF, cfg.detcr_hp_hz)]:
            dac_value, _ = preamp.DacMapAnalog(0, dac_signal, 0, value)
            dac_signals.append((dac_signal, dac_value))

        for head in self._heads:
            for signal, dac_value in dac_signals:
                preamp.DacWrite(head, signal, 0, dac_value)
            for circuit in circuits:
                preamp.CircuitEnable(head, circuit, 1)
            # Set DC DETCR Gain to high gain first
            preamp.DacWrite(head, preamp.DAC_DETCRDC_GAIN,
                            self.config.DCTCR_GAIN_MODES[-1], self.default_gain)
            # TODO: need ESF API to set diagnostic test mode (SFEMDW-10183)
            preamp.AdcRead(head, preamp.ADC_DETCR_VOLTAGE)

    def _setupXsRisc(self):
        """Set up XsRisc scratch pad registers"""
        # TODO: implement in ESF module SFEMDW-9826
        if not pynative.localmode_q():
            self.dctcr_utils.setupAsm(heads=self._heads,
                                      asm_file=self.config.XSRISC_ASM,
                                      heater='writer',
                                      nrev=self.config.NUM_REVS,
                                      wu_rev=self.config.WARMUP_REV,
                                      wu_offset=self.config.WARMUP_OFFSET,
                                      wave_samples=self.smp_per_cycle,
                                      peak=self.config.peak_dac)

    def _setX(self, x_values):
        """Set heats"""
        signal = self.config.heater_signal
        for hd, heat in x_values.items():
            preamp.DacWrite(hd, signal, 0, heat)

    def _stopAction(self, heads):
        """Terminate heads"""
        self.iterator.terminate(heads)
        # Turn off heater circuit
        for head in heads:
            preamp.CircuitEnable(head, 'heater', 0)
            if self.status[head] == self.STATUS.dcc_err:
                # Clear dcc for failed heads
                self.dcc[head] = None

    def _tearDown(self):
        """Load default XsRisc file."""
        # TODO: this is not needed once DC DETCR feature is included in ESF
        self.dctcr_utils.loadAsm('XsPack48.asm')

        # Turn off DETCR cirucits to avoid burning DETCR sensor
        circuits = [preamp.CIRCUIT_DETCR, preamp.CIRCUIT_DETCRDC]
        for head in self._heads:
            for circuit in circuits:
                preamp.CircuitEnable(head, circuit, 0)
            # Clear heat
            preamp.DacWrite(head, self.config.heater_signal, 0, 0)
