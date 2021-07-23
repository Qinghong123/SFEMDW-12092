"""DOAp basic acquire

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   production.base.event_pack import PackEvent
import production.base.metric as metric
from   production.mixin.aperio import AperioMixin
from   typing import ClassVar

class AcquireError(Exception): pass

class AcquireConfig(PackEvent.Config, AperioMixin.Config):
    """
    Parameters
    ----------
    interfaces : str
        String of comma separated names of channel interface(s)(ie - register(s) or DOAp alias (
        e.g.-"Seamless")) to read.
    n_revs : int
        Number of revs to collect data.
    """
    interfaces : str
    n_revs     : int

    AFH_MODE : ClassVar[str] = 'read'

    def _compile(self):

        if self.head_spec.heater == 'writer':
            self.metrics_in = (
                metric.READER_BIAS,
                metric.HEATER_WRITER_BIAS.tag(self.fly_height_nm, self.rpm))
        else:
            self.metrics_in = (
                metric.READER_BIAS,
                metric.HEATER_READER_BIAS.tag(self.fly_height_nm, self.rpm))

class Acquire(AperioMixin, PackEvent):
    """Event for running DOAp basic acquire"""

    Config = AcquireConfig
    Error  = AcquireError

    def _action(self):

        heads_pass, heads_fail = self.apo.acquire(
            n_revs        = self.config.n_revs,
            interfaces    = self.config.interfaces,
            path_host_dir = self._path_host_data)

        if self.config.debug:
            self.log.debug(f'{self.name}: apo p/f heads: {heads_pass=},  {heads_fail=}')

    def _exitEventState(self):

        self._disableReader()

        self._exitAperio(path_host_dir=self._path_host_data)

    def _exitTrackState(self):

        self.log.debug(f'{self._track=}')

        self._disableHeater()

        self._disableHeads()

    def _initEventState(self):

        if self.config.heat:
            self._data_in += self._fitDataToTracks(self.config.metrics_in[1])

        self._initAperio(aperio_ini=self.config.patt_spec.doap_ini)

        self._enableHeads()

        self._enableReader()

        self.apo.activateHeads(heads=self._heads)

    def _initTrackState(self):

        self._enableHeater()
