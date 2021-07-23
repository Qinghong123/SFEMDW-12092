""" Tune TMU PID value (though only returns PI).

An Integral Criteria / Lambda Tuning using Minimum Integral of Absolute Error (IAE) constants from
Instrument Engineer’s Handbook Vol. 2,  Bela Liptak, p422

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   collections import deque
from   core.build import Builder
from   core.data import DatumTrack
from   datetime import datetime
from   mdwec import XYB_RC_APCOM_UNSPECIFIED_ERROR
import numpy as np
from   production.base.event_pack import PackEvent
import production.base.metric as metric
from   production.mixin.aperio import AperioMixin
import pynative
import system.path as path
from   system.wrap import chiller
import time
from   typing import ClassVar, List, Tuple
from   util.control import ThermalController

# pylint:disable=invalid-name,old-division,range-builtin-not-iterating,zip-builtin-not-iterating

class TuneTmuPidError(Exception): pass

class TuneTmuPidConfig(AperioMixin.Config, PackEvent.Config):
    """ Class to perform automated tuning of the TMU PID controller """

    chiller_dwell_s : float
    power_delay     : int
    power_step      : float
    spiroid_dwell_s : float
    step_dwell_s    : float
    temp_offset     : float

    GAIN_P_INIT           : ClassVar[float] = 100.0
    GAIN_I_INIT           : ClassVar[float] = 180.0
    GAIN_D_INIT           : ClassVar[float] = 0.0
    GAIN_N_INIT           : ClassVar[float] = 0.0
    GAIN_P_SPIROID_DETUNE : ClassVar[float] = 1.4 # kluge?
    IAE_P_A               : ClassVar[float] = 0.758
    IAE_P_B               : ClassVar[float] = -0.861
    IAE_I_A               : ClassVar[float] = 1.020
    IAE_I_B               : ClassVar[float] = -0.323
    IAE_D_A               : ClassVar[float] = 0.0
    IAE_D_B               : ClassVar[float] = 0.0
    PLANT_GAIN_MIN        : ClassVar[float] = 0.1
    POWER_LIMIT_ABS       : ClassVar[int]   = 100
    POWER_LIMIT_BAD_TE    : ClassVar[int]   = -60
    POWER_SAMPLE_PERIOD_S : ClassVar[int]   = 0.5
    POWER_SAMPLES         : ClassVar[int]   = 10
    SAMPLE_PERIOD_S       : ClassVar[float] = 2.0
    SMOOTH_WINDOW_AMBIENT : ClassVar[int]   = 25
    SMOOTH_WINDOW_MOTOR   : ClassVar[int]   = 7   # TODO: needs to be odd?
    SMOOTH_WINDOW_SPIROID : ClassVar[int]   = 7
    SMOOTH_WINDOW_TRACK   : ClassVar[int]   = 4
    TEMP_AMBIENT_FACTOR   : ClassVar[float] = 0.5 # plant and actuator are affected differently
    TEMP_LIMIT_HIGH       : ClassVar[int]   = 24
    TEMP_LIMIT_LOW        : ClassVar[int]   = 16
    TEMP_SAMPLE_PERIOD_S  : ClassVar[int]   = 10
    TRACK_IDLE_OFFSET     : ClassVar[int]   = 500 # TODO: some other way

    def _compile(self):

        self.ambient = Builder().hardware('ambient')
        self.cntrl = ThermalController(
            kp = self.GAIN_P_INIT,
            ki = 1 / self.GAIN_I_INIT if (0 < abs(self.GAIN_I_INIT) < 1) else self.GAIN_I_INIT,
            kd = self.GAIN_D_INIT,
            kn = self.GAIN_N_INIT,
            ts = self.SAMPLE_PERIOD_S)

        self.power_delay_orig = chiller.ReturnTargetPowerDelay()

        self.temp_sample_modulus = int(self.TEMP_SAMPLE_PERIOD_S / self.SAMPLE_PERIOD_S)
        if self.temp_sample_modulus < 0:
            raise TuneTmuPidError(f'Invalid sampling configuration')

        # confirm valid tracks
        for track in self.tracks: self.patt_spec.calcEndTrack(track, self.TRACK_IDLE_OFFSET)

        self.metrics_in  = (metric.READER_BIAS,)
        self.metrics_out = (
            metric.CHILLER_PID_GAIN_P,
            metric.CHILLER_PID_GAIN_I,
            metric.SPIROID_PID_GAIN_P,
            metric.SPIROID_PID_GAIN_I)

    @staticmethod
    def ftcHeadSort(heads: List[int]) -> List[int]:
        """ Sort ``heads`` from middle outwards.

        TODO: duplication of method in `write_seeds.py`
        """
        HSA_CENTER_HEAD = 23
        return sorted(heads, key=lambda head: abs(head - HSA_CENTER_HEAD))

    @staticmethod
    def genModels(n_pts: int, type_: str) -> Tuple[np.array, int, float, float]:
        """ Generates exponential models. """

        def modelStep(t, r, tau): return r - (r * np.exp(-t / tau))

        #              min     max    N
        DELAY_ARGS =   15  ,   45  ,  31
        R_ARGS     = (100.0, 1000.0,  91) if type_ == 'spiroid' \
                else (  0.6,   20.0,  98)
        TAU_ARGS   =  100.0,  600.0, 101

        DELAYS = np.linspace(*DELAY_ARGS, endpoint=True, dtype=int)
        RS     = np.linspace(*R_ARGS    , endpoint=True)
        TAUS   = np.linspace(*TAU_ARGS  , endpoint=True)

        time_n = np.arange(n_pts)

        for delay in DELAYS:
            time_d = time_n[:-delay]
            for r in RS:
                for tau in TAUS:
                    model = np.concatenate((np.zeros(delay), modelStep(time_d, r, tau)))

                    yield model, delay, r, tau

    @staticmethod
    def smooth(data: np.array, window: np.array) -> np.array:
        """ Matlab's smooth() of data with odd window. """
        data_cumsum = np.cumsum(np.insert(data, 0, 0)) / window
        data_smooth = data_cumsum[window:] - data_cumsum[:-window]

        # TODO: are these ``start`` & ``stop`` needed?
        r = np.arange(1, window - 1, 2)
        start = np.cumsum(data[:window - 1])[::2] / r
        stop = (np.cumsum(data[:-window: -1])[::2] / r)[::-1]

        return np.concatenate((start, data_smooth, stop))

class TuneTmuPid(PackEvent, AperioMixin):
    """ System identification and PID selection. """

    Config = TuneTmuPidConfig
    Error  = TuneTmuPidError

    def _action(self):

        # stabilize chiller
        self._setTargetTemp()
        if not pynative.localmode_q():

            with self.config.pos_spec.idle():
                self.log.debug(f'Thermal stabilization: {self.config.chiller_dwell_s:.1f} sec')
                time.sleep(self.config.chiller_dwell_s)

        head_ftc = self._selectFtcHead()

        # spiroid closed-loop stabilization
        self._collectSamples(head_ftc, self.config.spiroid_dwell_s, set_power=True)

        # step injection
        self._setTargetPower()
        tracks, temps_mot, temps_amb = self._collectSamples(head_ftc, self.config.step_dwell_s)

        # Reset target temperature to ambient
        temp_amb, _ = self.config.ambient.read()
        chiller.SetTargetTemperature(str(temp_amb))

        # PID calculations
        tracks_cond, temps_cond = self._conditionSignals(tracks, temps_mot, temps_amb)

        if not pynative.localmode_q():

            self._disableHeater()
            with self.config.pos_spec.idle():
                self.log.debug('Calculating PID values.')

                p_c, i_c, *_ = self._calcPidValues(temps_cond , type_='chiller')
                p_s, i_s, *_ = self._calcPidValues(tracks_cond, type_='spiroid')
            self._enableHeater()

        p_s *= self.config.GAIN_P_SPIROID_DETUNE

        return [
            DatumTrack(metric.CHILLER_PID_GAIN_P, 'gain', self._track, p_c, self.name),
            DatumTrack(metric.CHILLER_PID_GAIN_I, 'gain', self._track, i_c, self.name),
            DatumTrack(metric.SPIROID_PID_GAIN_P, 'gain', self._track, p_s, self.name),
            DatumTrack(metric.SPIROID_PID_GAIN_I, 'gain', self._track, i_s, self.name)]

    def _calcPidValues(self, step: np.array, type_: str) -> Tuple[float]:
        """ PID gain calculations """

        rmse_min = 1e9
        for model, delay, r, tau in self.config.genModels(len(step), type_):

            rmse = np.sqrt(np.mean((model - step)**2))

            if rmse < rmse_min:
                rmse_min = rmse

                delay_select = delay
                r_select     = r
                tau_select   = tau

        plant_gain = max(delay_select / tau_select, self.config.PLANT_GAIN_MIN)

        r   = r_select / abs(self.config.power_step) * 100.0
        tau = tau_select * self.config.SAMPLE_PERIOD_S

        p = 1 / ((self.config.IAE_P_A / r) * (plant_gain ** self.config.IAE_P_B))
        i = tau / (self.config.IAE_I_A * plant_gain ** self.config.IAE_I_B)
        d = tau * (self.config.IAE_D_A * plant_gain ** self.config.IAE_D_B)
        n = 0.0

        return p, i, d, n

    def _collectSamples(self, head: int, time_s: float, set_power: bool = False) \
            -> Tuple[np.array, np.array, np.array]:
        """ Collect data and update control. """

        movavg    = deque(maxlen=self.config.SMOOTH_WINDOW_TRACK)
        n_samples = int(time_s / self.config.SAMPLE_PERIOD_S)
        track_end = self.config.patt_spec.calcEndTrack(self._track, self.config.TRACK_IDLE_OFFSET)

        temps_amb = np.zeros(n_samples, dtype=float)
        temps_mot = np.zeros(n_samples, dtype=float)
        tracks    = np.zeros(n_samples, dtype=float)

        self._updateController(head) # always "re-zero" controller

        for ii in range(n_samples):

            time_start_loop = time.time()

            track = self._getTrack(head)

            movavg.append(track)
            if ii < self.config.SMOOTH_WINDOW_TRACK:
                track_smooth = track
            else:
                track_smooth = sum(movavg) / movavg.maxlen

            _, power_cntrl = self.config.cntrl.update(track_smooth)

            if set_power: chiller.SetTargetPower(str(power_cntrl))

            if ii % self.config.temp_sample_modulus == 0:
                temp_amb, _ = self.config.ambient.read()

            temp_mot = chiller.GetActualTemperature()

            if not self.config.TEMP_LIMIT_LOW < temp_mot < self.config.TEMP_LIMIT_HIGH:
                raise TuneTmuPidError(f'Motor temperature {temp_mot} too extreme '
                                      f'{self.config.TEMP_LIMIT_LOW} < T < '
                                      f'{self.config.TEMP_LIMIT_HIGH}')

            temps_amb[ii] = temp_amb
            temps_mot[ii] = temp_mot
            tracks[ii]    = track

            # TODO: not sure this is a safe practice
            #  intended to prevent HDI issues
            self._moveToTrack(track_end)
            self._moveToTrack()

            # ensure uniform sample period
            time_remain = self.config.SAMPLE_PERIOD_S - (time.time() - time_start_loop)
            if time_remain < 0.0:
                raise TuneTmuPidError(f'Sample loop time exceeding sample period '
                                      f'{self.config.SAMPLE_PERIOD_S}s')

            time.sleep(time_remain)

        if self.config.debug:

            names_data = zip(('tracks', 'motor', 'ambient'), (tracks, temps_mot, temps_amb))

            path.host.makedirs(self._path_host_data, exist_ok=True)
            timestamp = datetime.now().strftime('%H%M%S')
            fp_cell = path.cell.ramPath('temp.csv')

            for name, data in names_data:

                fp_host = path.host.join(self._path_host_data, f'{timestamp}_{name}.csv')
                np.savetxt(fp_cell, data, delimiter=',')
                path.host.uploadFile(fp_host, fp_cell)

            path.cell.remove(fp_cell)

        return tracks, temps_mot, temps_amb

    def _conditionSignals(self, tracks: np.array, temps_mot: np.array, temps_amb: np.array) \
            -> Tuple[np.array, np.array]:
        """ Pre-processing of raw signal data. """

        # condition tracks
        tracks_zeroed = tracks - tracks[0]
        tracks_smooth = self.config.smooth(tracks_zeroed, self.config.SMOOTH_WINDOW_SPIROID)
        if np.mean(tracks_smooth) < 0: tracks_smooth = -tracks_smooth

        # condition temperature
        temps_amb_smooth = self.config.smooth(temps_amb, self.config.SMOOTH_WINDOW_AMBIENT)
        temps_norm = temps_mot - self.config.TEMP_AMBIENT_FACTOR * (temps_amb_smooth - temps_amb[0])
        temps_zeroed = temps_norm - temps_norm[0]
        temps_smooth = self.config.smooth(temps_zeroed, self.config.SMOOTH_WINDOW_MOTOR)
        if np.mean(temps_smooth) < 0: temps_smooth = -temps_smooth

        return tracks_smooth, temps_smooth

    def _exitEventState(self) -> None:

        self._disableReader()
        self._disableHeads()

        temp_amb, _ = self.config.ambient.read()
        chiller.SetTargetTemperature(str(temp_amb))

        chiller.ConfigTargetPowerDelay(str(self.config.power_delay_orig))

        self.apo.forceServoZoneMode(enable=False)

        self._exitAperio(path_host_dir=self._path_host_data)

    def _getChillerPower(self) -> float:
        """ Read power for some number of samples, return average. """

        power_total = 0
        for _ in range(self.config.POWER_SAMPLES):
            power_total += chiller.GetActualPower()
            time.sleep(self.config.POWER_SAMPLE_PERIOD_S)

        return power_total / self.config.POWER_SAMPLES

    def _getTrack(self, head: int) -> float:

        self.apo.getHeadPos()
        results = self.apo.getResults(head=head)

        return float(results[0]) * 2.0

    def _initEventState(self) -> None:

        self._enableHeads()
        self._enableReader()

        self._initAperio(aperio_ini=self.config.patt_spec.doap_ini)

        self.apo.activateHeads(self._heads)
        self.apo.forceServoZoneMode()
        self.apo.forceServoZone()
        self.apo.writeFpga('SectorDecimation', 0)

        chiller.ResetAckFailures()
        chiller.ConfigTargetPowerDelay(str(self.config.power_delay))
        chiller.SetCoolingPid(
            f'{self.config.cntrl.kp},'
            f'{self.config.cntrl.ki},'
            f'{self.config.cntrl.kd},'
            f'{self.config.cntrl.kn}')

    def _selectFtcHead(self) -> int:

        self._moveToTrack()

        heads_pass, _ = self.apo.runAttenuatorCal()

        heads_ftc = self.config.ftcHeadSort(heads_pass)

        for head in heads_ftc:
            try:
                self._getTrack(head)
            except RcError as err:
                if err.rc != XYB_RC_APCOM_UNSPECIFIED_ERROR: raise err
            else:
                head_ftc = head
                break
        else:
            raise TuneTmuPidError(f'Failed to obtain FTC head from {heads_ftc=}')

        self.apo.activateHeads([head_ftc])

        return head_ftc

    def _setTargetPower(self) -> float:

        power_chiller = self._getChillerPower()
        power_step = self.config.power_step + power_chiller
        power_step = min(max(power_step, -self.config.POWER_LIMIT_ABS), self.config.POWER_LIMIT_ABS)

        chiller.SetTargetPower(str(power_step)) # puts TMU/Oasis into external loop control

        return power_step

    def _setTargetTemp(self) -> float:

        temp_amb, _ = self.config.ambient.read()
        temp_target = self.config.temp_offset + temp_amb
        chiller.SetTargetTemperature(f'{temp_target}')

        temp = chiller.GetTargetTemperature()

        return temp

    def _updateController(self, head: int):
        """ Update the TMU controller reference and feed-forward. """

        self._moveToTrack()

        power_chiller = self._getChillerPower()
        track_ref     = self._getTrack(head)

        # TODO: is this necessary or should we rely on health monitor?
        if power_chiller < self.config.POWER_LIMIT_BAD_TE:
            raise TuneTmuPidError(f'Low TMU power {power_chiller} indicative of bad TE element')

        self.config.cntrl.setPositionReference(track_ref)
        self.config.cntrl.setFeedforward(power_chiller)
