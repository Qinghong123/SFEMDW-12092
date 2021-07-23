""" Base class for executing a sequence of events and performing grading.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   collections import defaultdict
from   contextlib import contextmanager
from   core.build import Builder
from   core.config import Config, ConfigName, ConfigObj
from   core.data import DataContainer, DataIo, Datum
from   datetime import datetime
from   production.base.event_hsa import HsaEvent
from   production.base.event_pack import PackEvent
from   production.base.metric import MetricIo, TIME_EXECUTION_TASK
from   system.type import coroutine
from   typing import ClassVar, Tuple

class TaskError(Exception): pass

class TaskConfig(Config, MetricIo):
    """Base configuration for all Tasks.

    Parameters
    ----------
    events : tuple
        An ordered list of event names to be executed during the task.
    grade_metrics : bool
        When ``True``, input and output metrics will be graded, when a spec is found in the
        ``grade.json``. When ``False``, metrics flow freely.
    metrics_persist_cell : tuple
        Metrics to be saved to the persistent memory.
    metrics_persist_hsa : tuple
        Metrics to be saved to the persistent memory, coupled to HSA SN.
    """
    events               : Tuple[ConfigName, ...]
    grade_metrics        : bool
    metrics_persist_cell : Tuple[ConfigName, ...]
    metrics_persist_hsa  : Tuple[ConfigName, ...]

    GRADE_IN_SPEC : ClassVar[None] = None

    def _compile (self):
        """ Compile task. """

        build = Builder()

        self.specs  = {}

        metrics_in  = set()
        metrics_out = set()
        for event_name in self.events:
            try:
                event = build.event(event_name)
            except Exception as err:
                print(f'Error building event "{event_name}"')
                raise err from None

            if self.grade_metrics:
                self.specs.update(self._buildSpecs(build, event))

            if metrics_in or metrics_out:
                metrics_in.update(set(event.config.metrics_in).difference(metrics_out))
            else:
                metrics_in.update(event.config.metrics_in)

            metrics_out.update(event.config.metrics_out)

        self.metrics_in  = metrics_in
        self.metrics_out = metrics_out

        metrics_persist = set(self.metrics_persist_cell + self.metrics_persist_hsa)
        # pylint:disable=dict-keys-not-iterating
        if metrics_miss := metrics_persist.difference(self.specs.keys()):
            raise TaskError(f'Persistent metrics must always be graded. Missing {metrics_miss}')

    @staticmethod
    def _buildSpecs(build, event):
        """ Check if all specs are valid.

        TODO: currently we require access of ``PackEvent.config.tracks`` parameter to implement
          unique grading for the same metric but different specs. This is not ideal. Ideally we
          can figure out a way to do this using "modifiers" within the spec.json.
        """
        n_tracks = 0
        if is_pack_event := isinstance(event, PackEvent):
            n_tracks = len(event.config.tracks)

        specs = {}
        for metric in event.config.metrics_in + event.config.metrics_out:
            try:
                spec = build.spec(metric)
            except Exception:
                continue

            if n_tracks:
                n_specs = len(spec)
                if n_specs not in [1, n_tracks] or (n_specs == n_tracks and not is_pack_event):
                    raise TaskError(f'{event.name} defines invalid by-track spec "{metric}"')

            specs[metric] = spec

        return specs

class Task(ConfigObj, DataIo):
    """A task is defined as an event or sequence of events that are required to be executed and
    in some cases compared against a spec before being graded.
    """
    Config = TaskConfig
    Error  = TaskError

    # pylint:disable=attribute-defined-outside-init

    _dirty : ClassVar[bool] = False

    @property
    def data(self):
        """ Returns DataContainer of all data accumulated by task """

        return self._data_out

    @property
    def dirty(self):
        """ Returns dirty flag indicating whether a write occured during the course of task
        execution.
        """

        return self._dirty

    def __call__(self, *args, **kargs):
        """ Convenience method intended for engineering mode only """

        config = self._config.asDict()
        config.update(kargs)

        self._config = self.Config(**config)

        return self.run(*args)

    def run(self, heads=None):
        """ Executes sequence of events defined in ``self.config.events``.

        It is possible for a task to contain only events that inherit from ``Event``. Thus it is
        *not* necessary for these events to be provided heads for input. The solution is an
        unfortunate bifurcation of behavior when a task is run standalone versus by a process.

        A process requires heads to run, as its core objective is to provide a final classification
        of each head/surface at completion. Therefore the current solution provides support for
        headless tasks only for the ``run`` method, not the ``runByEvent`` method.

        The following describes the behavior of a task run standalone when
        ``config.grade_metrics == True`` and out-of-spec (OOS) metrics are produced::

        - Without heads: task will throw an exception
        - With heads: task will disable heads

        Run from a process, a task will always assume heads are available, and will always defer to
        the process for disabling of said heads.

        With this implementation, slight behavioral differences can arise (though will likely be
        rare) as a process determines head eligibility from *grades*, not simply whether or not a
        metric was within specification. It is through this distinction that a process is able to
        keep heads enabled where a task would have otherwise disabled them.
        """
        # kluge: provide a dummy head to ``runByEvent`` to enable "headless" task execution
        heads = [-1] if (headless := heads is None) else heads

        grades_all    = list([] for _ in range(len(heads)))
        heads_start   = list(heads)
        task_by_event = self.runByEvent()

        while True:
            try:
                grades = task_by_event.send((heads, False))
            except StopIteration:
                break

            if grades is None: break # all tasks have completed

            for head, grades_head in zip(list(heads), grades):
                try:
                    # grab the first, if any, grade that is out-of-spec (oos)
                    grade_oos = next(
                        grade for grade in grades_head if grade is not self.config.GRADE_IN_SPEC)
                except StopIteration:
                    grade_oos = None

                if headless:
                    if grade_oos:
                        raise TaskError(f'Task {self.name} contains OOS metric(s), {grade_oos=}')
                else:
                    if grade_oos:
                        heads.remove(head)

                    grades_all[heads_start.index(head)] += grades_head

            if len(heads) == 0:
                task_by_event.close()
                break

        return None if headless else grades_all

    @coroutine
    def runByEvent(self):
        """ Coroutine enabling grading between events (and event actions).

        Expectation is, roughly::

            task ->    grades    -> caller
            task <- heads, retry <- caller

        """
        heads, _ = yield

        if len(heads) == 0: raise TaskError(f'Task {self.name} requires heads to operate')

        # pylint:disable=dict-keys-not-iterating
        metrics_grade = set(self.config.metrics_in).intersection(self.config.specs.keys())

        with self._taskState():

            if metrics_grade:

                # kluge: can't always guarantee that incoming data has same number of tracks.
                #  so apply first track's spec to all data
                data_in   = self.loadData(metrics_grade, filt_dup=True)
                grades, _ = self._gradeData(data_in, metrics_grade, heads, 0)

                heads, _ = yield grades # grades to be analyzed by caller. retry ignored.

            for event_name in self.config.events:

                event = self._build.event(event_name)
                heads = yield from self._runEvent(event, heads)

    def _exitTaskState(self):
        """ Configure system state after task completes. """

    def _gradeData(self, data, metrics, heads, step):

        grades = list([] for _ in heads)

        data_in_spec = []
        for metric in metrics:

            data_metric = data.filtBy(metric=metric)
            spec        = self.config.specs[metric]

            if is_head_datum := 'head' in data_metric.fields_meta:

                data_metric = data_metric.filtBy(head=heads)

            idx_spec = step if len(spec) > 1 else 0 # check if by-track grading

            grades_heads_oos = defaultdict(list)
            for datum in data_metric:

                grade = spec.grade(datum.value, idx_spec)

                if grade is self.config.GRADE_IN_SPEC:
                    data_in_spec.append(datum)
                else:
                    if is_head_datum: grades_heads_oos[grade].append(f'{datum.head:02d}')
                    else:             grades_heads_oos[grade] = None

                if is_head_datum: grades[heads.index(datum.head)].append(grade)
                else:             grades = [g + [grade] for g in grades]

            self.log.debug(str(spec))

            if grades_heads_oos:
                for grade_oos, heads_oos in grades_heads_oos.items():
                    str_oos = f'{metric} -> {grade_oos} @ heads -> ' \
                            + ', '.join(heads_oos) if heads_oos else 'all'
                    self.log.prod(str_oos)

        return grades, DataContainer(data_in_spec)

    def _initTaskState(self):
        """ Configure system state prior to task. """

        self._build    = Builder()
        self._data_out = DataContainer()

    def _runEvent(self, event, heads):

        is_hsa_event  = isinstance(event, HsaEvent)
        is_pack_event = isinstance(event, PackEvent)
        metrics_spec  = set(self.config.specs.keys())

        if is_pack_event: event_by_track = event.runByTrack()

        retry = False
        step  = 0
        while True:
            try:
                if is_pack_event:
                    try:
                        data = event_by_track.send((heads, retry))
                    except StopIteration:
                        break
                elif is_hsa_event:
                    data = event.run(heads)
                else:
                    data = event.run()

            finally:
                if is_hsa_event and event.dirty: self._dirty = True

            if data:
                self._data_out.extend(data)

                if metrics_grade := metrics_spec.intersection(data.metrics):
                    grades, data_in_spec = self._gradeData(data, metrics_grade, heads, step)

                    if data_in_spec: self._saveDataPersist(data_in_spec)

                    try:
                        heads, retry = yield grades # to be analyzed by caller
                    except GeneratorExit:
                        if is_pack_event: event_by_track.close()

            if not retry:
                if is_pack_event: step += 1
                else:             break

        return heads

    def _saveDataPersist(self, data):

        if metrics_persist_cell := set(self.config.metrics_persist_cell).intersection(data.metrics):

            data_metric = data.filtBy(metric=metrics_persist_cell)
            self.saveData(data_metric, 'cell', overwrite=True)

        if metrics_persist_hsa := set(self.config.metrics_persist_hsa).intersection(data.metrics):

            data_metric = data.filtBy(metric=metrics_persist_hsa)
            self.saveData(data_metric, 'hsa', overwrite=True)

    @contextmanager
    def _taskState(self):
        """ Provides initialization and teardown of task environment. """
        start_time = datetime.now()

        self.log.prod(f'Running {self.name} task')
        for param, value in self.config.asDict().items():
            self.log.debug(f'\t{param}: {value}')

        try:
            self._initTaskState()
            yield
        finally:
            self._exitTaskState()

            elapsed_time = datetime.now() - start_time
            self.log.prod(f'{self.name} task completed in {elapsed_time}')

            self.saveData(Datum(
                metric = TIME_EXECUTION_TASK.tag(self.name),
                units  = 'seconds',
                value  = int(elapsed_time.total_seconds()),
                source = self.name))
