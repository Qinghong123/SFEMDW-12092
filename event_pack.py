""" Provides base class for pack-dependent events.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   component.pattern import getPatternSpec, Pattern, Track
from   contextlib import contextmanager
from   core.data import DataContainer, DatumPack
from   hardware.positioner import getPositionerSpec, Positioner
from   hardware.sensors import Sensors
from   mdwec import XYB_PACK_RC_POSITIONER_TOO_SLOW, XYB_MEP_RC_ERROR_NOT_MOVING
from   production.base.event_hsa import HsaEvent
import pynative
import system.path as path
from   system.type import coroutine
from   system.wrap import mdic, mep, pcc, preamp
from   typing import ClassVar, Tuple, Union

class PackEventError(Exception): pass

class PackEventConfig(HsaEvent.Config):
    """ Base class for all pack-dependent event configurations.

    Defines the most basic configuration parameters, available for configuration by all events.

    Additionally, provides a tuple for metrics the event configuration is dependent on
    (`metrics_in`) as well as a tuple for metrics produced by the event (`metrics_out`). These
    tuples are not directly configurable, but may be configuration dependent. Therefore, the
    configuration class must be initialized before these tuples become valid.

    Parameters
    ----------
    fly_height_nm : float
        Target distance between head and disk.
    heat : bool
        If ``True``, set heat during data collection.
    rpm : float
        Target operating RPM.
    tracks : tuple
        Tracks at which to perform action.
    """
    fly_height_nm : float
    heat          : bool
    rpm           : float
    tracks        : Tuple[Track, ...]

    patt_spec     : ClassVar[Pattern]
    pos_spec      : ClassVar[Positioner]

    RPM_DELTA_MAX : ClassVar[int] = 2

    def __init__(self, **config):

        self.patt_spec = getPatternSpec()
        self.pos_spec  = getPositionerSpec()

        super().__init__(**config)

        if not all(0 <= track <= self.patt_spec.track_max for track in self.tracks):
            raise PackEventError(f'Invalid "tracks" {self.tracks}')

class PackEvent(HsaEvent):  # pylint:disable=abstract-method
    """ Base class for all pack-dependent events. """

    Config = PackEventConfig
    Error  = PackEventError

    # pylint:disable=attribute-defined-outside-init

    @property
    def _path_host_data(self):

        return path.host.join(super()._path_host_data, str(self._track))

    def run(self, heads):  # pylint:disable=arguments-differ
        """ Sole public method which executes event.

        Parameters
        ----------
        heads : List[int]
            Heads that the event will perform actions on.

        Returns
        -------
        data : DataContainer
            `DataContainer` of produced data, or `None`, depending on child class behavior.
        """
        event_by_track = self.runByTrack()

        while True:
            try:
                event_by_track.send((heads, False))
            except StopIteration:
                break

        return self._data_out

    @coroutine
    def runByTrack(self):
        """ Enables iterative processing of data by track.

        Yields
        ------
        data : DataContainer
            ``DataContainer`` of produced data, or ``None``, depending on child class behavior.
        """
        self._heads, _ = yield
        self._track    = self.config.tracks[0]

        with self._eventState():
            for self._track in self.config.tracks:
                while True:
                    self._moveToTrack()

                    with self._trackState():
                        data = self._action()
                        if data:
                            self._data_out += data
                            data = DataContainer(data)

                    self._heads, retry = yield data

                    if not retry: break

            if self._data_out:
                data = self._actionSummary(self._data_out)
                if data:
                    self._data_out += data
                    yield DataContainer(data)

    def _actionSummary(self, data_out: DataContainer) -> Union[None, list]:
        """ Optional method intended to be overwritten for Events which need to perform some level
        of post-processing on the Datums returned from the Event.run() across all test radii. In
        these cases, it is possible to generate extra Datums and append them to the output
        DataContainer. Should return a list of Datums.
        """

    def _disableHeater(self, heads=None):

        if self.config.heat:
            super()._disableHeater(heads)

    def _enableHeater(self, value_analog=None, heads=None, afh_mode='read'):

        if not self.config.heat:
            self.log.debug('config.heat is off, skip enabling heater')
            return

        if value_analog is None:
            wiwp = self.config.head_spec.getWiwp(afh_mode)
            if self.config.head_spec.hamr and afh_mode != 'read':
                # Add LIWP offset
                offset = {}
                heads = heads if heads else self._heads
                data = self._data_in.filtBy(metric=self.config.metrics_in.LASER_WRITE_BIAS,
                                            head=heads,
                                            track=self._track)
                for d in data:
                    offset[d.head] = wiwp + self.config.head_spec.getLiwp(afh_mode, d.value)
            else:
                offset = wiwp
            self._enableHeaterData(heads, offset)
        else:
            super()._enableHeater(value_analog, heads, afh_mode)

    def _enableSignalData(self, metric, circuit, signal, heads=None, offset=0.0):
        """ Map data loaded by event into preamp API calls. """

        data_metric = self._data_in.filtBy(metric=metric)

        # Some signal data may be track dependent, others may not.
        # The system should work with either.
        if 'track' not in data_metric.fields_meta:
            super()._enableSignalData(metric, circuit, signal, heads, offset)
            return

        data = data_metric.filtBy(
            head  = heads if heads else self._heads,
            track = self._track)

        *_, units_default = preamp.DacGetRange(0, signal, 0)
        is_dict = isinstance(offset, dict)
        for d in data:

            if d.units != units_default:
                raise self.Error(f'Invalid units {units_default} given default mode')
            if is_dict:
                val_offset = offset[d.head]
            else:
                val_offset = offset
            self._setSignal(d.head, circuit, signal, d.value + val_offset)

    @contextmanager
    def _eventState(self):

        if not Sensors.hsaLoaded(): raise PackEventError('HSA must be loaded to run a PackEvent')

        if abs(mdic.GetMotorRPM() - self.config.rpm) > self.config.RPM_DELTA_MAX:
            mdic.MotorChangeRPM(str(self.config.rpm))

        with super()._eventState(): yield

    def _exitTrackState(self):
        """ Configure system state after track action. """

    def _fitDataToTracks(self, metrics, tracks=None, fit_type='piecewise'):
        """Performs cross-track fit on the given metric

        Parameters
        ----------
        metrics : MetricContainer
            Iterable of metric names.
        tracks : list
            List of tracks to be extrapolated to. If ``None``, ``self.config.tracks`` is used.
        fit_type: str
            Fit type desired. Supported fit type includes 'linear', 'quadratic' and 'piecewise'.

        Returns
        -------
        DataContainer
            object containing the fitted data
        """
        tracks = self.config.tracks if tracks is None else tracks

        data_fit = []

        for metric, data_metric in self._data_in.filtBy(metric=metrics).iterBy('metric'):
            if 'head' in data_metric.fields_meta and 'track' in data_metric.fields_meta:
                units = data_metric[0].units
                for head, data_head in data_metric.iterBy('head'):
                    values_fit, _ = data_head.fit('track', tracks, fit_type)
                    for track, value_fit in zip(tracks, values_fit):
                        data_fit.append(
                            DatumPack(
                                metric = metric,
                                units  = units,
                                head   = head,
                                track  = track,
                                value  = value_fit,
                                source = self.name))
            else:
                data_fit.extend(data_metric)

        return DataContainer(data_fit)

    def _initTrackState(self):
        """ Configure system state before track action. """

    def _loadAsm(self, asm_file, *, force_load=False, load_local=False):
        """This method loads the specified ``asm_file`` file into the ESF system's pattern
        generator memory.

        The method checks the filename of the currently loaded .ASM file.  If it matches
        ``asm_file``, the load will will be skipped (unless ``force_load`` is used)

        Parameters
        ----------
        asm_file : str
            Filename of the .ASM file containing the desired pattern to be loaded into pattern
            memory.
            If no path is used in ``asm_file``, the ESF code assumes the file is in 1 of 2
            possible directories (depending on how ``load_local`` is set):
                on the cell @ d:/
                on the host @ c:/xyratex/stw/assembler
        force_load : bool
            If True, forces the reloading of ``asm_file`` even if the name is the same as the
            currently loaded .ASM
        load_local : bool
            Determines where the ESF API pcc.ChangePatternTStates() searches for ``asm_file``.
            If True, the API searches on the cell.
            If False, the API searches on the host.

        Notes
        -----
        TODO: KLUDGE: Move this method back into WriteEvent after SCSW-427 and SCSW-378 are
         complete. This method belongs in WriteEvent, but Ptc(PackEvent) currently needs this for
         pattern loading.
        """
        loaded_asm = pcc.GetLoadedProgram().lower()
        requested_asm = path.host.split(asm_file)[-1].lower().rstrip('.asm')
        if (loaded_asm == requested_asm) and not force_load:
            return

        local = '1' if load_local else '0'
        pcc.ChangePatternTStates(f'{asm_file}, 0, {local}')
        if self.config.debug:
            source = 'cell' if load_local else 'host'
            self.log.debug(f'Loaded "{requested_asm}" from {source} (replacing "{loaded_asm}")')

        if not pynative.localmode_q():
            #TODO: Use positoner context manager here? (SFEMDW-11862)
            # Can't use current version (pre-Serpente 0.2 Alpha) due to AFH conflict.
            mep.StartIdleSweep(f'{Track(0.25)},{Track(0.75)}')
            #TODO: SCSW-1295: pcc.FullClockWrite throws 1019
            #TODO: ESF says this full clock write is unnecessary when 2nd param of
            # pcc.ChangePatternTStates() is 0.  Investigate (SFEMDW-11863)
            pcc.FullClockWrite()
            mep.StopIdleSweep()
            self._moveToTrack()

    def _moveToTrack(self, track=None):
        """ Manages move state and move. """
        if track is None: track = self._track

        try:
            self.log.debug('Move to track %d' % track)

            mep.MoveToTrack('%d' % track) #pylint:disable=too-many-function-args
        except RcError as err:
            if err.rc not in (XYB_PACK_RC_POSITIONER_TOO_SLOW, XYB_MEP_RC_ERROR_NOT_MOVING):
                raise err

    @contextmanager
    def _trackState(self):
        """ Provides initialization and teardown of system at track. """
        try:
            self._initTrackState()
            yield
        finally:
            self._exitTrackState()
