"""Fine laser current calibration based on magnetic writer width(MWW).
This event is for HAMR head and media only.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   core.data import DatumPack
import numpy as np
from   production.base.event_write import WriteEvent, WriteEventConfig
import production.base.metric as metric
from   production.mixin.search import Search, SearchConfig
from   production.mixin.sts import SyncTaaScan, SyncTaaScanConfig
import system.path as path
from   system.wrap import pack, pcc, preamp
import time
from   typing import ClassVar, Tuple

class LaserMwwCalError(Exception):
    """Laser coarse calibration exceptions"""
    pass

class LaserMwwCalConfig(WriteEventConfig, SearchConfig, SyncTaaScanConfig):
    """Configuraiton class for laser MWW cal"""
    pattern: str
    target_mww: float

    MAX_MWW_COUNT   : ClassVar[int] = 5
    MIN_MWW_COUNT   : ClassVar[int] = 3
    MODES_WRITE     : ClassVar[Tuple[str, ...]] = (WriteEventConfig.MODE_DC_LOW,)
    NUM_PASSES      : ClassVar[int] = 3

    def _compile(self):
        self.metrics_in = (metric.HEATER_WRITER_BIAS.tag(self.fly_height_nm, self.rpm),
                           metric.READER_BIAS,
                           metric.LASER_WRITE_BIAS.tag('coarse'))
        self.metrics_out = (metric.LASER_WRITE_BIAS.tag('mww'), metric.TAA.tag('mww'))
        self.temp_fcal = path.cell.ramPath('lsc_mww.tmp')
        self.temp_mww = path.cell.ramPath('mww_data.tmp')

class LaserMwwCal(WriteEvent, Search, SyncTaaScan):
    """This class implements the MWW laser current calibration function."""
    Config = LaserMwwCalConfig
    Error = LaserMwwCalError

    def _acquire(self, x_values):
        """Read MWW data """
        result = {}
        heads = sorted(list(x_values.keys()))
        self._setHeat(heads, 'read')
        retry_heads = heads
        with open(self.config.temp_mww, 'a') as fs_mww:
            for retry in range(self.config.NUM_PASSES):
                data = self._scan(heads=retry_heads,
                                  zone=self._track,
                                  fs=fs_mww,
                                  prefix=f'{self.time_stamp}',
                                  data=x_values,
                                  retry=retry)
                # Retry is helpful because sometimes TAA readings can be corrupted by noise
                retry_heads = []
                msg = f'TAA scan errors for pass {retry}:\n'
                for head, (mww, peak, status) in data.items():
                    result[head] = (x_values[head], mww, peak, status)
                    if status not in [self.config.ERR_OK, self.config.ERR_GAP]:
                        retry_heads.append(head)
                        msg += f'Hd={head} Status={status}\n'
                if not retry_heads:
                    break
                else:
                    self.log.debug(msg)
        return result

    def _action(self):
        data = []
        self._search()
        for head in self._heads:
            try:
                offset, taa = self._result[head]
                laser_mww = self.coarse_laser[head] + offset
            except KeyError:
                laser_mww = taa = None
            data.append(DatumPack(metric=self.config.metrics_out.LASER_WRITE_BIAS,
                                  units='amps',
                                  head=head,
                                  track=self._track,
                                  value=laser_mww,
                                  source=self.name))
            data.append(DatumPack(metric=self.config.metrics_out.TAA,
                                  units='mV',
                                  head=head,
                                  track=self._track,
                                  value=taa,
                                  source=self.name))
        return data

    def _decide(self, heads):
        """Check the MWW results obtained so far and decide if need to stop search"""
        actions = {head: self.ACTION.go for head in heads}
        start_offset, end_offset, step = self.config.search_range
        msg = ''
        for head in heads:
            values = []
            for offset, mww, taa, status in self._data[head]:
                if mww is None:
                    continue
                values.append((offset, mww, taa))
            n = len(values)
            if not n: continue
            values.sort(key=lambda a: a[0])
            stop = False
            if n >= self.config.MAX_MWW_COUNT and mww >= self.config.target_mww:
                actions[head] = self.ACTION.stop
                stop = True
            elif self.iterator.values[head] >= end_offset:
                # Search range is exhausted, stop
                stop = True
            if stop and n >= self.config.MIN_MWW_COUNT:
                target_offset, target_taa = self._fitMww(values)
                msg += f'Hd={head} TargetOffset={target_offset:.4f} TargetTAA={target_taa}\n'
                # Check if MWW values can extrapolate
                if target_offset < start_offset < values[0][0]:
                    # Down projection is invalid due to breakpoint at start
                    msg += ': down projection inhibited'
                elif values[-1][0] < self.iterator.values[head] - step < target_offset:
                    msg += ': up projection inhibited'
                else:
                    self._result[head] = target_offset, target_taa
        if msg:
            self.log.debug(msg)
        return actions

    def _exitEventState(self):
        self._disableLaser()
        self._disableWriter()
        self._disableHeater()
        self._disableReader()
        self._disableHeads()
        if self.config.debug:
            for key, temp_file in {'fcal': self.config.temp_fcal, 'mww': self.config.temp_mww}.items():
                host_file = path.host.join(self.getPathHostData(), self.name, f'{self.time_stamp}_{key}_data.csv')
                self._uploadFile(temp_file, host_file)

    def _fitMww(self, data):
        """Linear fit MWW data to obtain the target offset and max TAA at the target MWW
        Parameters
        ----------
        data: list
            a list of tuples of (offset, mww, taa)

        Returns
        -------
        a tuple of (offset, taa) at target MWW
        """
        data = np.array(data)
        offset = data[:, 0]
        mww = data[:, 1]
        taa = data[:, 2]
        # Find target offset
        mww_coeff = np.polyfit(offset, mww, 1)
        target_offset = (self.config.target_mww - mww_coeff[1]) / mww_coeff[0]
        # Find target taa by evaluate the 2nd order fit
        taa_fit = np.poly1d(np.polyfit(offset, taa, 2))
        target_taa = taa_fit(target_offset)
        return target_offset, target_taa

    @staticmethod
    def _getTimeStamp():
        """Get time stamp """
        t = time.localtime()
        time_stamp = ''.join([f'{v:02d}' for v in
                              (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)])
        return time_stamp

    def _initEventState(self):

        self.time_stamp = self._getTimeStamp()
        # Set laser bias while keeping laser disabled
        laser_bias_dac, _ = preamp.DacMapAnalog(0, preamp.DAC_LASER_BIAS_CURRENT, 0,
                                                self.config.head_spec.laser_bias_current_amps)
        for head in self._heads:
            preamp.CircuitEnable(head, preamp.CIRCUIT_LASER, 0)
            preamp.DacWrite(head, preamp.DAC_LASER_BIAS_CURRENT, 0, laser_bias_dac)
        # Load concentric pattern
        self._loadAsm(self.config.pattern)
        # Fit metrics in for event tracks
        self._data_in += self._fitDataToTracks(metrics=(self.config.metrics_in.HEATER_WRITER_BIAS,
                                                        self.config.metrics_in.LASER_WRITE_BIAS))
        self._initWrite()
        # Prepare logging file
        if self.config.debug:
            with open(self.config.temp_fcal, 'w') as fs:
                fs.write('TimeStamp,Track,Head,WrHeat,LaserCurrent,Offset,MWW,MaxTAA,Status\n')
            with open(self.config.temp_mww, 'w') as fs:
                fs.write('TimeStamp,Track,Head,Offset,Pass,Position,TAA\n')

        self.log.debug('Synchronous TAA scan settings: \n'
                       f'\tVelocity={self.config.scan_velocity} ips\n'
                       f'\tAccerleration={self.config.acceleration} counts/sample^2\n'
                       f'\tNumTracks={self.config.ntracks}\n'
                       f'\tSamplesPerTrk={self.config.ns_trk}\n'
                       f'\tStepSize={self.config.step_size} nm'
                       )
        # Prepare heads
        self._enableHeads()
        self._enableReader()
        self._enableWriter()

    def _initStepState(self, x_values):
        """Erase TAA band with DC_LOW pattern and then write a concentric track"""
        heads = sorted(list(x_values.keys()))
        # Set laser current and clearance
        self._setX(x_values)
        self._setHeat(heads, afh_mode='erase', laser_offsets=x_values)
        # DC erase track band
        self._writeTrack(heads)
        # Concentric write a single track
        self._writeTrack(heads, self._track)

    def _initTrackState(self):

        self._initSearch(self._heads)
        self.re_heat = {}
        self.wr_heat = {}
        for head, head_data in self._data_in.filtBy(metric=self.config.metrics_in.HEATER_WRITER_BIAS,
                                                    track=self._track).iterBy('head'):
            self.re_heat[head] = head_data[0].value
        self.coarse_laser = {}
        for head, head_data in self._data_in.filtBy(metric=self.config.metrics_in.LASER_WRITE_BIAS,
                                                    track=self._track).iterBy('head'):
            self.coarse_laser[head] = head_data[0].value

    def _logData(self, heads):
        """Log current fine cal data"""
        with open(self.config.temp_fcal, 'a') as fs:
            info_str = f'{self.time_stamp},{self._track}'
            msg = ''
            for head in heads:
                offset, mww, peak, status = self._data[head][-1]
                laser_current = self.coarse_laser[head] + offset
                if mww is None:
                    # Make it numeric
                    peak = mww = -1.0
                fs.write(f'{info_str},{head},{self.wr_heat[head]},{laser_current},'
                         f'{offset},{mww},{peak},{status}\n')
                msg += f'Hd={head} LC={laser_current:.4f} Offset={offset:.4f} '
                if mww is None:
                    msg += f'MWW=None MaxTAA=None Status={status}\n'
                else:
                    msg += f'MWW={mww:.2f} MaxTAA={peak:.2f} Status={status}\n'
        self.log.debug(msg)

    def _setHeat(self, heads, afh_mode, laser_offsets=None):
        """Set heat for write or read
        Parameters
        ----------
        heads: list
            the list of head numbers
        afh_mode: str
            can be 'read' or 'erase'
        laser_offsets: dict
            the laser current offsets to coarse laser current in forms of {head:offset}
        """
        for head in heads:
            # Set heat with offset
            heat_offset = self.config.head_spec.getWiwp(afh_mode)
            if laser_offsets is not None:
                laser_current = max(0, self.coarse_laser[head] + laser_offsets[head])
                heat_offset += self.config.head_spec.getLiwp(afh_mode, laser_current)
                self.wr_heat[head] = max(0, self.re_heat[head] + heat_offset)
            heat = max(0, self.re_heat[head] + heat_offset)
            dac, _ = preamp.DacMapAnalog(head, preamp.DAC_HEATER_WRITER_POWER_DURING_READ, 0, heat)
            preamp.DacWrite(head, preamp.DAC_HEATER_WRITER_POWER_DURING_READ, 0, dac)
            preamp.DacWrite(head, preamp.DAC_HEATER_WRITER_POWER_DURING_WRITE, 0, dac)

    def _setX(self, x_values):
        """Set laser current based on the given offset values"""

        for head, offset in x_values.items():
            # Set laser
            laser_current = max(0, self.coarse_laser[head] + offset)
            dac, _ = preamp.DacMapAnalog(head, preamp.DAC_LASER_WRITE_CURRENT, 0, laser_current)
            preamp.DacWrite(head, preamp.DAC_LASER_WRITE_CURRENT, 0, dac)

    def _stopAction(self, heads):
        """Stop searching given heads"""
        self.iterator.terminate(heads)

    def _writeTrack(self, heads, track=None):
        """Erase band for single track writing.
        Assumes heat and laser current are all set.
        """
        self._moveToTrack(track)
        try:
            for circuit in [preamp.CIRCUIT_HEATER, preamp.CIRCUIT_LASER]:
                for head in heads:
                    preamp.CircuitEnable(head, circuit, 1)
            if track is None:
                pcc.SelectWriteMode('spiral')
                self._write()
            else:
                self._writeSingleTrack(track)
        finally:
            # Disable circuits
            for circuit in [preamp.CIRCUIT_HEATER, preamp.CIRCUIT_LASER]:
                for head in heads:
                    preamp.CircuitEnable(head, circuit, 0)

    def _writeSingleTrack(self, track):
        pcc.SelectWriteMode('early')
        pack.Write(f'{track}, {track}, {self.config.START_GRAYCODE}')
