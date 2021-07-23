""" Hardware module to control the ESF Positioner System.

"Copyright (C) Seagate Technology LLC, 2021. All rights reserved."
"""
from   component import Component
from   component.pattern import getPatternSpec, Track
from   contextlib import contextmanager
from   core.build import Builder
from   core.config import Config
import math
import mdwec
import pynative
from   system.type import coroutine
from   system.wrap import mep, pcc
import time

class PositionerError(Exception): pass

class PositionerConfig(Config):
    """
    Parameters
    ----------
    ff_profile : str
        The filename of the feed-forward spiral profile assembler file to use for this mep config.
    """
    ff_profile : str

    # Move params
    DEFAULT_MOVE_RETRIES   = 2
    DEFAULT_MOVE_TOLERANCE = 10  # servo tracks

    # Sweep params
    SWEEP_VEL_SCALE_LIMIT = 2.0

    MR_SEPARATION_NM = -5100 # nanometers

    # Idle params
    IDLE_OD_STROKE_PCT = 0.3
    IDLE_ID_STROKE_PCT = 0.7
    POLL_DELAY         = 2.0  # seconds

    # Stability check params
    STABILITY_TRACK_TOLERANCE = 3  # servo tracks
    STABILITY_MOVE_DELTA      = 10 # servo tracks

    # Stroke calibration params
    ID_STROKE_SCALE     = 1.01
    OD_SENSOR_CONTROL   = 'Stop'
    OD_SENSOR_LOC       = 'OuterEdge'
    OD_SENSOR_POS_ARCMM = -55.0
    TRACK_0_OFFSET_MAX  = 12.0

    MM_PER_INCH = 25.4

    def _compile(self):


        self.patt_spec = getPatternSpec()

        PIVOT_TO_GAP_MM    = pynative.getinifloatex('mep', 'PivotToGap_mm')     # millimeters
        START_TRACK_RADIUS = \
            pynative.getinifloatex('mep', 'StartTrackRadius') / self.MM_PER_INCH     # inches
        TRACK_PITCH_INCHES = \
            pynative.getinifloatex('mep', 'TrackPitch') / (self.MM_PER_INCH * 1000.0)# inches

        self.DEFAULT_MOTOR_DEMAND   = pynative.getinifloatex('mep', 'MaxMotorCurrentAmps') # amps
        self.NANO_RADIANS_PER_COUNT = \
            pynative.getinifloatex('mep', 'NanoRadiansPerCount') # unitless
        self.SAMPLE_PERIOD          = \
            pynative.getinifloatex('mep', 'SamplePeriod')        # nanoseconds
        self.STANDARD_VELOCITY      = \
            pynative.getiniintex('mep', 'StandardVelocity')    # counts/sample
        self.PIVOT_TO_CENTER        = \
            pynative.getinifloatex('mep', 'PivotToSpindleCentre_mm') / self.MM_PER_INCH # inches

        self.PIVOT_TO_GAP = PIVOT_TO_GAP_MM / self.MM_PER_INCH # inches
        self.TPI          = TRACK_PITCH_INCHES ** -1.0    # servo tracks per inch

        a = pow(self.PIVOT_TO_GAP, 2) + pow(self.PIVOT_TO_CENTER, 2) - pow(START_TRACK_RADIUS, 2)
        b = 2.0 * self.PIVOT_TO_CENTER * self.PIVOT_TO_GAP

        self.THETA_S = math.acos(a / b)

class Positioner(Component):
    """ Positioner hardware class.

    Hardware interface class to the MDW Positioner System. Main purpose is to wrap and combine MEP
    calls to provide higher-level functionality native inside the ESF python framework.
    """

    Config = PositionerConfig
    Error  = PositionerError

    async_target_track  = None
    idle_tracks         = None
    last_poll_timestamp = None
    target_track        = None

    def calcLinearVelocityForPitch(self, track_pitch, rpm=None):
        """ Method which calculates the linear velocity required from the positioner to achieve a
        scan rate that is equivalent to the desired ``track_pitch``. Typically used for erasure
        operations which wish to control the signal overlap amount by controlling track pitch.
        Common use for this function is to return a value for use with mep.SetSpiralVelocityLinear.

        Parameters
        ----------
        track_pitch : float
            The desired effective track pitch that the calculated linear velocity should achieve
        rpm : float
            An optical RPM override. Default = current RPM.

        Returns
        -------
        float
            Linear velocity in millimeters per second
        """
        rev_time = 60/rpm if rpm else self.config.patt_spec.getParams('rev_time')
        track_pitch *= 1e-6  # convert nanometers -> millimeters
        linear_vel = track_pitch / rev_time
        self.log.debug(f'Calculated {linear_vel=} for {rev_time=}, {track_pitch=}')

        return linear_vel

    def calibrateStroke(self, track_0_offset):

        if abs(track_0_offset) > self.config.TRACK_0_OFFSET_MAX:
            raise PositionerError(f'Supplied track_0_offset |{track_0_offset}| '
                                  f'exceeds internal limit of '
                                  f'{self.config.TRACK_0_OFFSET_MAX}')

        # First determine the position of the OD Opto Sensor to establish initial physical reference
        if pynative.localmode_q():
            # SCSW-1296: mep.PositionWithOptos returns None
            od_opto_pos = 0.0
        else:
            od_opto_pos = mep.PositionWithOptos(self.config.OD_SENSOR_POS_ARCMM,
                                                self.config.OD_SENSOR_LOC,
                                                self.config.OD_SENSOR_CONTROL)

        self.log.debug(f'Found OD Opto Sensor at {od_opto_pos} arc mm')

        # Use the supplied track_0_offset to jog further relative to the sensor, then set zero ref
        mep.MoveToPosition(od_opto_pos + track_0_offset)
        mep.SetZeroReferencePosition()
        track_0_pos = mep.GetCurrentPosition()
        mep.SetODEndStopPosition(track_0_pos)  # TODO: needed?
        self.log.debug('Set Track 0 Reference point')

        # Legacy "soft ID" sequence assumes positioner is at zero reference, then calculates an
        # angular offset to the maxmium ID position (usually 101% of max track) and defines this as
        # the software ID stop position
        soft_id_offset = \
            mep.CalculatePositionOffset(pcc.GetMaxNumberOfTracks()
                                        * self.config.ID_STROKE_SCALE)
        mep.SetIDEndStopPosition(soft_id_offset)
        self.log.debug(f'Set ID End Stop position: {soft_id_offset}')

        # Positioner is still at zero reference, now check positioner can reach max track
        self.move(pcc.GetMaxNumberOfTracks())
        self.log.debug('Seek to check max track has completed')

    def convertAngularVelocity(self, velocity, to_unit='ips'):
        """ General-purpose angular velocity unit converter function between two commonly-used units
        for expressing angular velocity of the positioner servo system: "ips" and "cps".

        ips = radial inches per second
        cps = counts per servo sample

        This method is commonly used in conjunction with the return from mep.GetSpiralVelocity(),
        which is in in units of "cps". Therefore the default for `to_unit` = "ips". In either case,
        the `velocity` input is assumed to be the opposite type of the `to_unit` definition.

        Parameters
        ----------
        velocity : float
            Spiral (twist) velocity in units opposite to ``to_unit`` argument
        to_unit : str
            Can be either "ips" or "cps". Must be opposite of ``velocity`` units. Default = "ips"

        Returns
        -------
        float
            Angular velocity in units ``to_unit``
        """
        if to_unit == 'ips':
            return velocity \
                   / self.config.SAMPLE_PERIOD \
                   * self.config.NANO_RADIANS_PER_COUNT \
                   * 1e-9 \
                   * self.config.PIVOT_TO_GAP

        elif to_unit == 'cps':
            return velocity \
                   * self.config.SAMPLE_PERIOD \
                   / self.config.NANO_RADIANS_PER_COUNT \
                   / 1e-9 \
                   / self.config.PIVOT_TO_GAP

        else:
            raise PositionerError(f'Unsupported unit type: {to_unit}. Must be "ips" or "cps"')

    def estimateJog(self, track, skew_angle_offset=0):
        """ Calculation API which accepts a track number and skew_angle_offset empirical argument.
        Returns the approximated MR Jog in units of tracks.

        Parameters
        ----------
        track : int
            The track number at which to estimate MR Jog
        skew_angle_offset : float
            An empirical amount of skew offset, based on geometry, to apply. Default = 0.

        Returns
        -------
        float
            The estimated MR Jog in units of tracks at ``track`` with ``skew_angle_offset``.
        """
        mr_sep = 1e-6 * self.config.MR_SEPARATION_NM / self.config.MM_PER_INCH  # -> microinches
        skew_angle_offset = skew_angle_offset * math.pi / 180

        theta = self.config.THETA_S - (track / self.config.TPI) / self.config.PIVOT_TO_GAP

        a = self.config.PIVOT_TO_GAP**2 + self.config.PIVOT_TO_CENTER**2
        b = 2 * self.config.PIVOT_TO_CENTER * self.config.PIVOT_TO_GAP * math.cos(theta)
        r = math.sqrt(a - b)

        a = self.config.PIVOT_TO_GAP**2 - self.config.PIVOT_TO_CENTER**2 + r**2
        b = 2 * self.config.PIVOT_TO_GAP * r

        skew = math.asin(a / b)
        skew += skew_angle_offset

        mr_jog = mr_sep * math.sin(skew) * self.config.TPI + 0.5
        return mr_jog

    @contextmanager
    def idle(self, track_start=None, track_end=None, velocity_scalar=None):
        """Convenience method that provides a context manager to help minimize head dwell time
        during operations that take more than a few seconds to return.  This method uses ESF APIs
        mep.StartIdleSweep and mep.StopIdleSweep to start the heads moving and stop them at the end.
        The context manager yields after starting the movement and stops the movement upon return.

        Parameters
        ----------
        track_start : Track
            A percent-stroke or explicit tracks to define the start idling boundary
        track_end : Track
            A percent-stroke or explicit tracks to define the end idling boundary
        velocity_scalar : float
            velocity_scalar to get the optimum speed during movement
            If velocity_scalar is None, speed is the default, 0.01 IPS
        """
        track_start = \
            Track(self.config.IDLE_OD_STROKE_PCT) if track_start is None else Track(track_start)
        track_end = \
            Track(self.config.IDLE_ID_STROKE_PCT) if track_end is None else Track(track_end)

        if pynative.localmode_q():
            yield
        else:
            self.log.debug(f'Beginning positioner.idle() Idling at track: {track_start}')
            try:
                if velocity_scalar is None:
                    mep.StartIdleSweep(track_start, track_end)
                else:
                    current_ips = mep.GetVelocityIPS()
                    vel_ips = velocity_scalar * current_ips
                    mep.StartIdleSweep(track_start, track_end, vel_ips)
                yield
            finally:
                mep.StopIdleSweep()
                self.log.debug(f'Idle stop at track: {mep.GetCurrentTrack()}')

    def move(self, track, tolerance=None, retries=None, **context_args):
        """ Public API for moving the positioner. Internally wraps an mep.MoveToTrack call with
        ``moveContext()`` context management, and applies the retry-handling with tolerance to the
        move operation itself. Note, this call is synchronous and therefore blocking.

        Parameters
        ----------
        track : int
            Track number to move to
        tolerance : float
            Amount of tolerance, in units of tracks, to allow before allowing an exception to raise
        retries : int
            Number of retries on the move command to enable more time to meet seek-settle criteria
        context_args : dict
            Additional optional arguments supported by the ``moveContext()`` context manager

        Examples
        --------
        >>> from hardware.positioner import Positioner
        >>> positioner = Positioner()
        >>> # ensure the heads are loaded first...
        >>> positioner.move(125000)
        >>> positioner.move(150000, afh_mode=('passive' or 'min_heat'))
        """
        with self.moveContext(**context_args):
            self._move(track, retries, tolerance)

    @contextmanager
    def moveContext(self, afh_mode='passive', velocity_mode='Standard', velocity_scalar=None):
        """ Public contextmanager API for use with general move operations. Enables flexibility for
        use with other MEP "move" APIs, such as mep.MoveToTrackDecimal by providing setup/teardown
        configuration of standard MEP attributes. Implements mep.SaveStatus and mep.RestoreStatus
        to save/restore the current velocity, acceleration and profile mode.  Enables override to
        velocity mode through mep.SetVelocityMode. Calls into the AFH module APIs to set an afh_mode
        for the move (passive or min are the only safe heat modes)..

        Note, all arguments to this command are optional.

        Parameters
        ----------
        afh_mode : str
            TODO: Add proper reference here when available
            afh_mode string defined by API. Default is None.
        velocity_mode : str
            A mode supported by mep.SetVelocityMode.
            Default is 'Standard'.
        velocity_scalar : float
            Scaling of the MEP "StandardVelocity", via mep.SetVelocity()

        Examples
        --------
        >>> from hardware.positioner import Positioner
        >>> # ensure heads are loaded
        >>> with Positioner().moveContext(afh_mode='servo', velocity_mode='spiral'):
        >>>     mep.MoveToDeltaTrackDecimal(-100)
        """
        self.log.debug(f'moveContext entered with context args: '
                       f'{afh_mode=}, {velocity_mode=}, {velocity_scalar=}')
        if velocity_mode != 'Standard' or velocity_scalar is not None:
            if not pynative.localmode_q(): mep.SaveStatus()  # Save velocity, accel & profile mode.

        try:
            mep.SetVelocityMode(velocity_mode)

            if velocity_scalar is not None:
                mep.SetVelocityIPS(velocity_scalar * mep.GetVelocityIPS())

            # TODO: rip this stuff out - heats are handled elsewhere by other mechanisms
            if afh_mode == 'passive': #  default action
                #self._setPassiveFlyHeight()
                pass
            elif afh_mode is not None: # set the requested mode
                #self._setAfhMode(afh_mode) # CAUTION anything other than min_heat can be dangerous.
                pass
            # else make no fly height adjustment

            yield

        finally:
            if velocity_mode != 'Standard' or velocity_scalar is not None:
                # Restore vel, accel & profile mode.
                if not pynative.localmode_q(): mep.RestoreStatus()
            self.log.debug('moveContext teardown complete')

    def _checkTrackInTolerance(self, target, tolerance):
        """ Helper method which returns whether or not the current track number obtained via
        mep.GetCurrentTrack() compared to input ``target`` is with ``tolerance``, in tracks.
        """
        if not (delta := abs(mep.GetCurrentTrack() - target)) <= tolerance:
            self.log.debug(f'Move with loose seek-settle failed, '
                           f'delta = {delta} tracks. Failure will be ignored.')
            return False

        return True

    def _move(self, track, retries=None, tolerance=None):
        """ Lowest-level wrapping of mep.MoveToTrack to implement retry-handling. Does not include
        context managerment like it's public counterpart, ``move()``.

        Parameters
        ----------
        track : int
            Target track of destination.
        retries : int
            Optional. Number of retries. Default = ``DEFAULT_MOVE_RETRIES``
        tolerance : int
            Optional. Tolerance of move error in units of tracks. Default=``DEFAULT_MOVE_TOLERANCE``
        """
        retry_handler = self.__retryHandler(track, retries, tolerance)

        while True:

            try:

                self.target_track = track
                mep.MoveToTrack(track)
                break

            except RcError as error:

                try:
                    retry_handler.send(error)

                except StopIteration:
                    break

            finally:
                self.target_track = None

        self.log.debug(f'Move to {track} is complete')

    def __pollIdling(self):
        """ __pollIdling() is a helper method for monitoring mep.AsyncMoveToTrack operation.
        It is returned by the ``moveContext()`` context manager, and when called, does the
        following things:

            1. Calls into to mep.VelocityMove_IsMoving()
            2. If any DLL errors are raised, they are handled via __retryHandler
            3. If __retryHandler does not re-raise, a new async move is started
            4. Else, if the return from the mep call is False, a new async move is started.

        Because this method is meant to be called in very specific situations which are managed
        at a higher level by public functions in this class, this method is private and should
        only be called internally to this class.
        """
        if (now := time.time()) - self.last_poll_timestamp > self.config.POLL_DELAY:
            self.last_poll_timestamp = now
            try:
                moving = mep.VelocityMove_IsMoving()

            except RcError as error:

                if error not in (mdwec.XYB_RC_TIMEOUT,
                                 mdwec.XYB_RC_VERIFY_SETTLE_FAILED,
                                 mdwec.XYB_MEP_RC_ERROR_NOT_MOVING):
                    raise PositionerError(f'Move failed for {error}')

                next_trk = next(self.idle_tracks)
                self.log.debug(f'Polling successful with retries.'
                               f'Beginning next move to {next_trk}')
                mep.AsyncMoveToTrack(next(self.idle_tracks))

            else:
                if not moving:
                    next_trk = next(self.idle_tracks)
                    self.log.debug(f'Polling successful, move complete. '
                                   f'Beginning next move to {next_trk}')
                    mep.AsyncMoveToTrack(next(self.idle_tracks))

    @coroutine
    def __retryHandler(self, target=None, retries=None, tolerance=None):
        """ Retry handling method to encapsulate the actions taken during expected MDWEC returns
        during normal positioner moves via mep.MoveToTrack. Will raise an RcError exception if the
        return code is unexpected, or will raise a PositionerError if the expected errors are
        received even after applying the retry loop and tolerance checks.
        """
        if tolerance is None:
            tolerance = self.config.DEFAULT_MOVE_TOLERANCE
        if retries is None:
            retries = self.config.DEFAULT_MOVE_RETRIES

        for attempt in range(retries):

            error = yield

            if error.rc not in (mdwec.XYB_RC_TIMEOUT,
                                mdwec.XYB_RC_VERIFY_SETTLE_FAILED,
                                mdwec.XYB_MEP_RC_ERROR_NOT_MOVING):
                raise PositionerError(f'Move failed for {error.rc}')

            if self._checkTrackInTolerance(target, tolerance):
                self.log.debug('Tolerance check successful')
                break

            self.log.debug('Settle not in tolerance. Retrying again...')

        else:
            raise PositionerError(f'Retries and tolerance checks not able to recover {error.rc}'
                                  f'using {retries=} and {tolerance=}')

def getPositionerSpec():
    """ Get the hardware-specified Positioner instance.

    Returns
    -------
    Positioner
        An instance of the Positioner class.
    """
    mep_config = '%d%d%d%d%d%d' % (
        pynative.getiniintex('mep', 'encodertypedef'),
        pynative.getiniintex('mep', 'poweramptypedef'),
        pynative.getiniintex('mep', 'actuatormotortypedef'),
        pynative.getiniintex('mep', 'mcbtypedef'),
        pynative.getiniintex('mep', 'systemtuningdef'),
        pynative.getiniintex('mep', 'actuatorblockdef'))

    return Builder().hardware(f'positioner-{mep_config}')
