""" InlineRRO PV Event

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   component.pattern import getPatternSpec
from   core.data import DatumHead
from   core.config import ConfigName
import mdwec
from   production.base.event_hsa import HsaEvent
import production.base.metric as metric
from   production.mixin.aperio import AperioMixin
import pynative
from   typing import ClassVar, Dict

class InlineRroError(Exception): pass

class InlineRroConfig(HsaEvent.Config, AperioMixin.Config):
    """
    InlineRro should really be a ``PackEvent``, but the ``PackEventConfig.tracks`` definition
    is in the ``doap_ini`` file. The associated track moves are executed by the Aperio system and
    the returned data averaged across said tracks. To accomodate these facts, the remaining
    ``PackEventConfig`` parameters are duplicated, here, in ``InlineRroConfig``, to partially
    emulate a ``PackEvent``.

    Because the movements take place on the Aperio box, the minimum heat value is applied for each
    head.

    The responsibility of moving to each track should probably be refactored to be done here, in
    production code.

    Parameters
    ----------
    fly_height_nm : float
        Target distance between head and disk.
    heat : bool
        If ``True``, set heat during data collection.
    num_heads : int
        The number of heads that are desired to be tested for InlineRRO. If None, then test all hds.
    rpm : float
        Target operating RPM.
    """
    fly_height_nm : float
    heat          : bool
    num_heads     : int
    rpm           : float

    UNITS_MAP : ClassVar[Dict[ConfigName, str]] = {
        metric.SPIRAL_PES_WIRRO_MAX :       'tracks',
        metric.SPIRAL_PES_WIRRO_3STD :      'tracks',
        metric.SPIRAL_PES_CWIRRO_MAX :      'tracks',
        metric.SPIRAL_PES_CWIRRO_3STD :     'tracks',
        metric.SPIRAL_PES_IWIRRO_MAX :      'tracks',
        metric.SPIRAL_PES_IWIRRO_3STD :     'tracks',
        metric.SPIRAL_TIMING_WIRRO_MAX :    'timing',
        metric.SPIRAL_TIMING_WIRRO_3STD :   'timing',
        metric.SPIRAL_TIMING_CWIRRO_MAX :   'timing',
        metric.SPIRAL_TIMING_CWIRRO_3STD :  'timing',
        metric.SPIRAL_TIMING_IWIRRO_MAX :   'timing',
        metric.SPIRAL_TIMING_IWIRRO_3STD :  'timing'}

    def _compile(self):

        self.patt_spec = getPatternSpec() # kluge: see docstring

        # ``metrics_in`` is referenced by index in ``_initEventState``. Thus if the order of the
        # metrics in ``metrics_in`` is modified, so must the index.
        if self.head_spec.heater == 'writer':
            self.metrics_in = (
                metric.READER_BIAS,
                metric.HEATER_WRITER_BIAS.tag(self.fly_height_nm, self.rpm))
        else:
            self.metrics_in = (
                metric.READER_BIAS,
                metric.HEATER_READER_BIAS.tag(self.fly_height_nm, self.rpm))

        # metrics_out ordering should match the DOAp ordered results
        self.metrics_out = (metric.SPIRAL_PES_WIRRO_MAX,     metric.SPIRAL_PES_WIRRO_3STD,
                            metric.SPIRAL_PES_CWIRRO_MAX,    metric.SPIRAL_PES_CWIRRO_3STD,
                            metric.SPIRAL_PES_IWIRRO_MAX,    metric.SPIRAL_PES_IWIRRO_3STD,
                            metric.SPIRAL_TIMING_WIRRO_MAX,  metric.SPIRAL_TIMING_WIRRO_3STD,
                            metric.SPIRAL_TIMING_CWIRRO_MAX, metric.SPIRAL_TIMING_CWIRRO_3STD,
                            metric.SPIRAL_TIMING_IWIRRO_MAX, metric.SPIRAL_TIMING_IWIRRO_3STD)

class InlineRro(AperioMixin, HsaEvent):
    """Class for encapsulating HSA event Aperio mixin to perform Inline RRO"""
    Config = InlineRroConfig
    Error  = InlineRroError

    def _action(self):

        try:
            if pynative.localmode_q():
                # for some reason ``pyint.PyApiTrace('true')`` shows nothing returning from this
                # ``apcom.SendString``
                heads_fail = []
                heads_pass = self._heads[:]
            else:
                heads_pass,\
                heads_fail = self.apo.spiralInlineRro(path_host_dir=self._path_host_data)

            if self.config.debug:
                self.log.debug(f'{self.name}: apo p/f heads: {heads_pass=},  {heads_fail=}')

        except RcError as err:
            if err.rc != mdwec.XYB_RC_APCOM_UNSPECIFIED_ERROR: raise err

            if self.config.debug:
                self.log.debug(f'DOAp data collection failure during {self.name} event. Ignoring.')

            return self._retrieveData(heads_pass=[])    # During PV data collection, do NOT
                                                        # interrupt process for collection failure.
        else:
            return self._retrieveData(heads_pass=heads_pass)

    def _exitEventState(self):

        self._disableHeater()

        self._disableReader()
        self._disableHeads()
        self._exitAperio(path_host_dir=self._path_host_data)

    def _initEventState(self):

        self._initAperio(aperio_ini=self.config.patt_spec.doap_ini)
        if self.config.num_heads is not None: self._reduceHeadlist()
        self._enableHeads()
        self._enableReader()
        self.apo.activateHeads(heads=self._heads)

        if self.config.heat:
            # kluge: set min heat
            data_heat = self._data_in.filtBy(metric=self.config.metrics_in[1])
            for head, data_head in data_heat.iterBy('head'):
                self._enableHeater(min(d.value for d in data_head), heads=[head])

    def _reduceHeadlist(self):
        """ So-called SmartInlineRro headlist reduction algorithm intended to decimate the headlist
        input into InlineRRO test for the purpose of saving process cycle time. Algorithm works
        by selecting a span of length ``self.config.num_heads // 2`` at the top and bottom ends of
        the headstack to formulate an ideal reduced headlist. This list is then filtered by the
        currently-available heads defined by ``self._heads``.
        """
        lo_partition = list(range(0, self.config.num_heads//2))
        hi_partition = list(range(48-self.config.num_heads//2, 48))
        reduced_headlist = [head for head in self._heads if head in (lo_partition + hi_partition)]
        if not reduced_headlist:
            raise self.Error('Unable to define reduced headlist for InlineRRO - no heads available')
        self.log.debug(f'SmartInlineRro reduced headlist: {reduced_headlist}')
        self._heads = reduced_headlist

    #TODO: Common method.
    # Move to DoapMixin -or- HsaEvent (due to self._head usage)
    #                        ^^^^^^^^ NOT the same as other DOAp PV data collection events!
    def _retrieveData(self, heads_pass):

        data = []
        for head in self._heads:

            if head in heads_pass:
                results = self.apo.getResults(head=head)
            else:
                results = [None] * len(self.config.metrics_out)

            for metric_name, result in zip(self.config.metrics_out, results):
                data.append(
                    DatumHead(
                        metric = metric_name,
                        units  = self.config.UNITS_MAP[metric_name],
                        head   = head,
                        value  = float(result) if result else result,
                        source = self.name))

        return data

