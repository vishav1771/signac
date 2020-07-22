# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Implement the SyncedCollection class.

SyncedCollection encapsulates the synchronization and different data-structures.
These features are implemented in different subclasses which enable us to use a
backend with different data-structures or vice-versa. It declares as abstract
methods the methods that must be implemented by any subclass to match the API.
"""
import os
import errno
import sys
import logging
from contextlib import contextmanager
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Collection

from .errors import Error

try:
    import numpy
    NUMPY = True
except ImportError:
    NUMPY = False


logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 32 * 2**20    # 32 MB

_BUFFERED_MODE = 0
_BUFFERED_MODE_FORCE_WRITE = None
_BUFFER_SIZE = None
_BUFFER = dict()
_SYNCED_DATA = dict()
_FILEMETA = dict()


# TODO: uuid hash
class BufferException(Error):
    """An exception occured in buffered mode."""

    pass


class BufferedFileError(BufferException):
    """Raised when an error occured while flushing one or more buffered files.

    .. attribute:: files

        A dictionary of files that caused issues during the flush operation,
        mapped to a possible reason for the issue or None in case that it
        cannot be determined.
    """

    def __init__(self, files):
        self.files = files

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.files)


def _get_filemetadata(filename):
    try:
        return os.path.getsize(filename), os.path.getmtime(filename)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise


def _store_in_buffer(filename, blob, synced_data=False):
    assert _BUFFERED_MODE > 0
    blob_size = sys.getsizeof(blob)
    buffer_load = get_buffer_load()
    if _BUFFER_SIZE > 0:
        if blob_size > _BUFFER_SIZE:
            return False
        elif blob_size + buffer_load > _BUFFER_SIZE:
            logger.debug("Buffer overflow, flushing...")
            flush_all()
    _BUFFER[filename] = blob
    _SYNCED_DATA[filename] = synced_data
    if synced_data:
        if not _BUFFERED_MODE_FORCE_WRITE:
            _FILEMETA[filename] = _get_filemetadata(filename)
    return True


def flush_all():
    """Execute all deferred JSONDict write operations."""
    logger.debug("Flushing buffer...")
    issues = dict()
    while _BUFFER:
        filename, blob = _BUFFER.popitem()
        if not _BUFFERED_MODE_FORCE_WRITE:
            meta = _FILEMETA.pop(filename)
            if _get_filemetadata(filename) != meta:
                issues[filename] = 'File appears to have been externally modified.'
                continue
        if not _SYNCED_DATA.pop(filename):
            try:
                SyncedCollection.from_base(filename=filename, data=blob).sync()
            except OSError as error:
                logger.error(str(error))
                issues[filename] = error
    if issues:
        raise BufferedFileError(issues)


def get_buffer_size():
    """Return the current maximum size of the read/write buffer."""
    return _BUFFER_SIZE


def get_buffer_load():
    """Return the current actual size of the read/write buffer."""
    return sum((sys.getsizeof(x) for x in _BUFFER.values()))


def in_buffered_mode():
    """Return true if in buffered read/write mode."""
    return _BUFFERED_MODE > 0


@contextmanager
def buffer_reads_writes(buffer_size=DEFAULT_BUFFER_SIZE, force_write=False):
    """Enter a global buffer mode for all JSONDict instances.

    All future write operations are written to the buffer, read
    operations are performed from the buffer whenever possible.

    All write operations are deferred until the flush_all() function
    is called, the buffer overflows, or upon exiting the buffer mode.

    This context may be entered multiple times, however the buffer size
    can only be set *once*. Any subsequent specifications of the buffer
    size are ignored.

    :param buffer_size:
        Specify the maximum size of the read/write buffer. Defaults
        to DEFAULT_BUFFER_SIZE. A negative number indicates to not
        restrict the buffer size.
    :type buffer_size:
        int
    """
    global _BUFFERED_MODE
    global _BUFFERED_MODE_FORCE_WRITE
    global _BUFFER_SIZE
    assert _BUFFERED_MODE >= 0

    # Basic type check (to prevent common user error)
    if not isinstance(buffer_size, int) or \
            buffer_size is True or buffer_size is False:    # explicit check against boolean
        raise TypeError("The buffer size must be an integer!")

    # Can't enter force write mode, if already in non-force write mode:
    if _BUFFERED_MODE_FORCE_WRITE is not None and (force_write and not _BUFFERED_MODE_FORCE_WRITE):
        raise BufferException(
            "Unable to enter buffered mode with force write enabled, because "
            "we are already in buffered mode with force write disabled.")

    # Check whether we can adjust the buffer size and warn otherwise:
    if _BUFFER_SIZE is not None and _BUFFER_SIZE != buffer_size:
        raise BufferException("Buffer size already set, unable to change its size!")

    _BUFFER_SIZE = buffer_size
    _BUFFERED_MODE_FORCE_WRITE = force_write
    _BUFFERED_MODE += 1
    try:
        yield
    finally:
        _BUFFERED_MODE -= 1
        if _BUFFERED_MODE == 0:
            try:
                flush_all()
            finally:
                assert not _BUFFER
                assert not _SYNCED_DATA
                assert not _FILEMETA
                _BUFFER_SIZE = None
                _BUFFERED_MODE_FORCE_WRITE = None


class SyncedCollection(Collection):
    """The base synced collection represents a collection that is synced with a backend.

    The class is intended for use as an ABC. The SyncedCollection is a
    :class:`~collections.abc.Collection` where all data is stored persistently in
    the underlying backend.
    """

    backend = None

    def __init__(self, parent=None):
        self._data = None
        self._parent = parent
        self._suspend_sync_ = 0

    @classmethod
    def register(cls, *args):
        """Register the synced data structures.

        Registry is used when recursively converting synced data structures to determine
        what to convert their children into.

        Parameters
        ----------
        *args
            Classes to register
        """
        if not hasattr(cls, 'registry'):
            cls.registry = defaultdict(list)
        for _cls in args:
            cls.registry[_cls.backend].append(_cls)

    @classmethod
    def from_base(cls, data, backend=None, **kwargs):
        """Dynamically resolve the type of object to the corresponding synced collection.

        Parameters
        ----------
        data : any
            Data to be converted from base class.
        backend: str
            Name of backend for synchronization. Default to backend of class.
        **kwargs:
            Kwargs passed to instance of synced collection.

        Returns
        -------
        data : object
            Synced object of corresponding base type.
        """
        backend = cls.backend if backend is None else backend
        if backend is None:
            raise ValueError("No backend found!!")
        for _cls in cls.registry[backend]:
            if _cls.is_base_type(data):
                return _cls(data=data, **kwargs)
        if NUMPY:
            if isinstance(data, numpy.number):
                return data.item()
        return data

    @abstractmethod
    def to_base(self):
        """Dynamically resolve the synced collection to the corresponding base type."""
        pass

    @contextmanager
    def _suspend_sync(self):
        """Prepare context where load and sync are suspended."""
        self._suspend_sync_ += 1
        yield
        self._suspend_sync_ -= 1

    @classmethod
    @abstractmethod
    def is_base_type(cls, data):
        """Check whether data is of the same base type (such as list or dict) as this class."""
        pass

    @abstractmethod
    def _load(self):
        """Load data from file."""
        pass

    @abstractmethod
    def _sync(self):
        """Write data to file."""
        pass

    def sync(self):
        """Synchronize the data with the underlying backend."""
        if self._suspend_sync_ <= 0:
            if self._filename is not None:
                data = self.to_base()
                if _BUFFERED_MODE > 0:  # Storing in buffer
                    _store_in_buffer(self._filename, data)
                else:   # Saving to disk:
                    self._sync(data)
            else:
                self._parent.sync()

    def load(self):
        """Load the data from the underlying backend."""
        if self._suspend_sync_ <= 0:
            if self._filename is not None:
                if _BUFFERED_MODE > 0:
                    if self._filename in _BUFFER:
                        # Load from buffer:
                        blob = _BUFFER[self._filename]
                    else:
                        # Load from disk and store in buffer
                        blob = self._load()
                        _store_in_buffer(self._filename, blob, synced_data=True)
                else:
                    # Just load from disk
                    blob = self._load()
                    # Reset the instance
                    self._update(blob)
            else:
                self._parent.load()

    # defining common methods
    def __getitem__(self, key):
        self.load()
        return self._data[key]

    def __delitem__(self, item):
        del self._data[item]
        self.sync()

    def __iter__(self):
        self.load()
        return iter(self._data)

    def __len__(self):
        self.load()
        return len(self._data)

    def __call__(self):
        self.load()
        return self.to_base()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self() == other()
        else:
            return self() == other

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)
