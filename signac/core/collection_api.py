# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import os
import json
import errno
import uuid
import sys
import logging
from copy import copy
from contextlib import contextmanager
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from collections.abc import MutableSequence
from .errors import Error

try:
    import numpy
    NUMPY = True
except ImportError:
    NUMPY = False

try:
    from collections.abc import Collection
except ImportError:
    # Collection does not exist in Python 3.5, only Python 3.6 or newer.

    from collections.abc import Sized, Iterable, Container

    def _check_methods(C, *methods):
        mro = C.__mro__
        for method in methods:
            for B in mro:
                if method in B.__dict__:
                    if B.__dict__[method] is None:
                        return NotImplemented
                    break
            else:
                return NotImplemented
        return True

    class Collection(Sized, Iterable, Container):  # type: ignore
        @classmethod
        def __subclasshook__(cls, C):
            if cls is Collection:
                return _check_methods(C,  "__len__", "__iter__", "__contains__")
            return NotImplemented

        @classmethod
        def __instancecheck__(cls, instance):
            for parent in cls.__mro__:
                if not isinstance(instance, parent):
                    return False
            return True


logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 32 * 2**20    # 32 MB

_BUFFERED_MODE = 0
_BUFFERED_MODE_FORCE_WRITE = None
_BUFFER_SIZE = None
_BUFFER = dict()
_SYNCED_DATA = dict()
_FILEMETA = dict()


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

    def __init__(self, parent=None):
        self._data = None
        self._suspend_sync_ = 0
        self._parent = parent

    def __instancecheck__(self, instance):
        if not isinstance(instance, Collection):
            return False
        else:
            return all(
                [hasattr(instance, attr) for attr in
                 ['sync', 'load', 'to_base', 'from_base']])

    # TODO add back-end
    @classmethod
    def from_base(self, data, filename=None, parent=None):
        if isinstance(data, Mapping):
            return JSONDict(filename=filename, data=data, parent=parent)
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return JSONList(filename=filename, data=data, parent=parent)
        elif NUMPY:
            if isinstance(data, numpy.number):
                return data.item()
            elif isinstance(data, numpy.ndarray):
                return JSONList(filename=filename, data=data.tolist(), parent=parent)
        return data

    @abstractmethod
    def to_base(self):
        pass

    @contextmanager
    def _suspend_sync(self):
        self._suspend_sync_ += 1
        yield
        self._suspend_sync_ -= 1

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _sync(self):
        pass

    @contextmanager
    def _safe_sync(self):
        backup = self._data
        try:
            yield
        except BaseException:
            self._data = backup
            raise

    def sync(self):
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
                with self._suspend_sync():
                    self.reset(blob)
            else:
                self._parent.load()

    @contextmanager
    def buffered(self):
        buffered_collection = self.from_base(data=self, parent=BufferedSyncedCollection())
        yield buffered_collection
        self.reset(data=buffered_collection)


class BufferedSyncedCollection:
    def load(self):
        pass

    def sync(self):
        pass


class _SyncedDict(SyncedCollection, MutableMapping):

    _PROTECTED_KEYS = ('_data', '_suspend_sync_', '_load', '_sync', '_parent')

    VALID_KEY_TYPES = (str, int, bool, type(None))

    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        super().__init__(**kwargs)
        if data is None:
            self._data = {}
        else:
            self._data = {
                self._validate_key(key): self.from_base(data=value, parent=self)
                for key, value in data.items()
            }

    def to_base(self):
        converted = {}
        for key, value in self._data.items():
            if isinstance(value, SyncedCollection):
                converted[key] = value.to_base()
            else:
                converted[key] = value
        return converted

    def reset(self, data=None):
        if data is None:
            data = {}
        backup = self._data
        if isinstance(data, Mapping):
            try:
                with self._suspend_sync():
                    for key in data:
                        if key in self._data:
                            if data[key] == self._data[key]:
                                continue
                            try:
                                self._data[key].reset(key)
                                continue
                            except (ValueError, AttributeError):
                                pass
                        self._data[key] = self.from_base(data=data[key], parent=self)
                    remove = set()
                    for key in self._data:
                        if key not in data:
                            remove.add(key)
                    for key in remove:
                        del self._data[key]
                self.sync()
            except BaseException:  # rollback
                self._data = backup
                raise
        else:
            raise ValueError("The data must be a mapping or None not {}.".format(type(data)))

    @staticmethod
    def _validate_key(key):
        "Emit a warning or raise an exception if key is invalid. Returns key."
        if isinstance(key, _SyncedDict.VALID_KEY_TYPES):
            str_key = str(key)
            if '.' in str_key:
                from ..errors import InvalidKeyError
                raise InvalidKeyError(
                    "keys may not contain dots ('.'): {}".format(key))
            else:
                return key
        else:
            from ..errors import KeyTypeError
            raise KeyTypeError(
                "keys must be str, int, bool or None, not {}".format(type(key).__name__))

    def __delitem__(self, item):
        self.load()
        with self._safe_sync():
            del self._data[item]
            self.sync()

    def __setitem__(self, key, value):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data[self._validate_key(key)] = self.from_base(data=value, parent=self)
            self.sync()

    def __getitem__(self, key):
        self.load()
        return self._data[key]

    def __iter__(self):
        self.load()
        return iter(self._data)

    def __call__(self):
        self.load()
        return self.to_base()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self() == other()
        else:
            return self() == other

    def __len__(self):
        self.load()
        return len(self._data)

    def __repr__(self):
        return repr(self())

    def __str__(self):
        return str(self())

    def keys(self):
        self.load()
        return self._data.keys()

    def values(self):
        self.load()
        return self.to_base().values()

    def items(self):
        self.load()
        return self.to_base().items()

    def get(self, key, default=None):
        self.load()
        return self._data.get(key, default)

    def pop(self, key, default=None):
        self.load()
        with self._safe_sync():
            ret = self._data.pop(key, default)
            self.sync()
        return ret

    def popitem(self, key, default=None):
        self.load()
        with self._safe_sync():
            ret = self._data.pop(key, default)
            self.sync()
        return ret

    def clear(self):
        self.load()
        with self._safe_sync():
            self._data = {}
            self.sync()

    def update(self, mapping):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                for key, value in mapping.items():
                    self[key] = self.from_base(data=value, parent=self)
            self.sync()

    def setdefault(self, key, default=None):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                ret = self._data.setdefault(key, self.from_base(data=default, parent=self))
            self.sync()
        return ret


class SyncedAttrDict(_SyncedDict):

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name.startswith('__'):
                raise
            try:
                return self.__getitem__(name)
            except KeyError as e:
                raise AttributeError(e)

    def __setattr__(self, key, value):
        try:
            super().__getattribute__('_data')
        except AttributeError:
            super().__setattr__(key, value)
        else:
            if key.startswith('__') or key in self._PROTECTED_KEYS:
                super().__setattr__(key, value)
            else:
                self.__setitem__(key, value)

    def __delattr__(self, key):
        if key.startswith('__') or key in self._PROTECTED_KEYS:
            super().__delattr__(key)
        else:
            self.__delitem__(key)


class SyncedList(SyncedCollection, MutableSequence):

    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        super().__init__(**kwargs)
        if data is None:
            self._data = []
        else:
            self._data = [self.from_base(data=value, parent=self) for value in data]

    def to_base(self):
        converted = list()
        for value in self._data:
            if isinstance(value, SyncedCollection):
                converted.append(value.to_base())
            else:
                converted.append(value)
        return converted

    def reset(self, data=None):
        if data is None:
            data = []
        if isinstance(data, Sequence) and not isinstance(data, str):
            with self._suspend_sync():
                backup = copy(self._data)
                try:
                    for i in range(min(len(self), len(data))):
                        if data[i] == self._data[i]:
                            continue
                        try:
                            self._data[i].reset(data[i])
                            continue
                        except (ValueError, AttributeError):
                            pass
                        self._data[i] = self.from_base(data=data[i], parent=self)
                    if len(self._data) > len(data):
                        self._data[:len(data)]
                    else:
                        self.extend(data[len(self):])
                    self.sync()
                except BaseException:  # rollback
                    self._data = backup
                    raise
        else:
            raise ValueError("The data must be a non-string sequence or None.")

    def __delitem__(self, item):
        self.load()
        with self._safe_sync():
            del self._data[item]
            self.sync()

    def __setitem__(self, key, value):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data[key] = self.from_base(data=value, parent=self)
            self.sync()

    def __getitem__(self, key):
        self.load()
        return self._data[key]

    def __iter__(self):
        self.load()
        return iter(self._data)

    def __len__(self):
        self.load()
        return len(self._data)

    def __reversed__(self):
        self.load()
        return reversed(self._data)

    def __iadd__(self, iterable):
        self.load()
        with self._safe_sync():
            self._data += [self.from_base(data=value, parent=self) for value in iterable]
            self.sync()

    def __call__(self):
        self.load()
        return self.to_base()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self() == other()
        else:
            return self() == other

    def __repr__(self):
        return repr(self())

    def __str__(self):
        return str(self())

    def insert(self, index, item):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data.insert(index, self.from_base(data=item, parent=self))
            self.sync()

    def append(self, item):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data.append(self.from_base(data=item, parent=self))
            self.sync()

    def extend(self, iterable):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data.extend([self.from_base(data=value, parent=self) for value in iterable])
            self.sync()

    def remove(self, item):
        self.load()
        with self._safe_sync():
            with self._suspend_sync():
                self._data.remove(self.from_base(data=item, parent=self))
            self.sync()

    def clear(self):
        with self._safe_sync():
            self._data = []
            self.sync()


class JSONCollection(SyncedCollection):

    def __init__(self, **kwargs):
        filename = kwargs.pop('filename', None)
        self._filename = None if filename is None else os.path.realpath(filename)
        self._write_concern = kwargs.pop('write_concern', True)
        super().__init__(**kwargs)

    def _load(self):
        try:
            with open(self._filename, 'rb') as file:
                blob = file.read()
                return json.loads(blob.decode())
        except IOError as error:
            if error.errno == errno.ENOENT:
                return None

    def _sync(self):
        data = self.to_base()
        # Serialize data:
        blob = json.dumps(data).encode()

        if self._write_concern:
            dirname, filename = os.path.split(self._filename)
            fn_tmp = os.path.join(dirname, '._{uid}_{fn}'.format(
                uid=uuid.uuid4(), fn=filename))
            with open(fn_tmp, 'wb') as tmpfile:
                tmpfile.write(blob)
            os.replace(fn_tmp, self._filename)
        else:
            with open(self._filename, 'wb') as file:
                file.write(blob)


class JSONDict(JSONCollection, SyncedAttrDict):
    def __init__(self, filename=None, data=None, parent=None, write_concern=False):
        if (filename is None) == (parent is None):
            raise ValueError(
                "Illegal argument combination, one of the two arguments, "
                "parent or filename must be None, but not both.")
        super().__init__(filename=filename, data=data, parent=parent, write_concern=write_concern)


class JSONList(JSONCollection, SyncedList):
    def __init__(self, filename=None, data=None, parent=None, write_concern=False):
        if (filename is None) == (parent is None):
            raise ValueError(
                "Illegal argument combination, one of the two arguments, "
                "parent or filename must be None, but not both.")
        super().__init__(filename=filename, data=data, parent=parent, write_concern=write_concern)
