# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Implements JSON-backend.

This implements the JSON-backend for SyncedCollection API by
implementing sync and load methods.
"""

import os
import json
import errno
import uuid

from .synced_collection import SyncedCollection
from .buffered_collection import BufferedSyncedCollection
from .syncedattrdict import SyncedAttrDict
from .synced_list import SyncedList
from .buffered_collection import _get_filemetadata


def get_namespace(class_name):
    """Generate namespace for classname"""
    return uuid.uuid5(uuid.NAMESPACE_URL, 'signac::'+class_name)


class JSONCollection(BufferedSyncedCollection):
    """Implement sync and load using a JSON back end."""

    backend = 'JSON'  # type: ignore

    def __init__(self, filename=None, data=None, write_concern=False, no_sync=False, **kwargs):
        kwargs['data'] = data
        super().__init__(**kwargs)
        if (filename is None) == (self._parent is None):
            raise ValueError(
                "Illegal argument combination, one of the two arguments, "
                "parent or filename must be None, but not both.")
        self.backend_kwargs['filename'] = None if filename is None else os.path.realpath(filename)
        self.backend_kwargs['write_concern'] = write_concern
        self.backend_kwargs['backend'] = self.backend
        self._id = uuid.uuid5(get_namespace(type(self).__name__), self.backend_kwargs['filename'])
        if not no_sync and data is not None:
            self.sync()

    def _load(self):
        """Load the data from a JSON-file."""
        try:
            with open(self.backend_kwargs['filename'], 'rb') as file:
                blob = file.read()
                return json.loads(blob)
        except IOError as error:
            if error.errno == errno.ENOENT:
                return None

    def _sync(self, data=None):
        """Write the data to json file."""
        if data is None:
            data = self.to_base()
        _filename = self.backend_kwargs['filename']
        # Serialize data:
        blob = json.dumps(data).encode()
        # When write_concern flag is set, we write the data into dummy file and then
        # replace that file with original file.
        if self.backend_kwargs['write_concern']:
            dirname, filename = os.path.split(_filename)
            fn_tmp = os.path.join(dirname, '._{uid}_{fn}'.format(
                uid=uuid.uuid4(), fn=filename))
            with open(fn_tmp, 'wb') as tmpfile:
                tmpfile.write(blob)
            os.replace(fn_tmp, _filename)
        else:
            with open(_filename, 'wb') as file:
                file.write(blob)

    def _get_metadata(self):
        return _get_filemetadata(self.backend_kwargs['filename'])


class JSONDict(JSONCollection, SyncedAttrDict):
    """A dict-like mapping interface to a persistent JSON file.

    The JSONDict inherits from :class:`~core.collection_api.SyncedCollection`
    and :class:`~core.syncedattrdict.SyncedAttrDict`.

    .. code-block:: python

        doc = JSONDict('data.json', write_concern=True)
        doc['foo'] = "bar"
        assert doc.foo == doc['foo'] == "bar"
        assert 'foo' in doc
        del doc['foo']

    .. code-block:: python

        >>> doc['foo'] = dict(bar=True)
        >>> doc
        {'foo': {'bar': True}}
        >>> doc.foo.bar = False
        {'foo': {'bar': False}}

    .. warning::

        While the JSONDict object behaves like a dictionary, there are
        important distinctions to remember. In particular, because operations
        are reflected as changes to an underlying file, copying (even deep
        copying) a JSONDict instance may exhibit unexpected behavior. If a
        true copy is required, you should use the `to_base()` method to get a
        dictionary representation, and if necessary construct a new JSONDict
        instance: `new_dict = JSONDict(old_dict.to_base())`.

    Parameters
    ----------
    filename: str, optional
        The filename of the associated JSON file on disk (Default value = None).
    data: mapping, optional
        The intial data pass to JSONDict. Defaults to `list()`
    parent: object, optional
        A parent instance of JSONDict or None (Default value = None).
    write_concern: bool, optional
        Ensure file consistency by writing changes back to a temporary file
        first, before replacing the original file (Default value = None).
    """

    pass


class JSONList(JSONCollection, SyncedList):
    """A non-string sequence interface to a persistent JSON file.

    The JSONDict inherits from :class:`~core.collection_api.SyncedCollection`
    and :class:`~core.syncedlist.SyncedList`.

    .. code-block:: python

        doc = JSONList('data.json', write_concern=True)
        doc.append("bar")
        assert doc[0] == "bar"
        assert len(doc) == 1
        del doc[0]

    .. warning::

        While the JSONList object behaves like a list, there are
        important distinctions to remember. In particular, because operations
        are reflected as changes to an underlying file, copying (even deep
        copying) a JSONList instance may exhibit unexpected behavior. If a
        true copy is required, you should use the `to_base()` method to get a
        dictionary representation, and if necessary construct a new JSONList
        instance: `new_list = JSONList(old_list.to_base())`.

    Parameters
    ----------
    filename: str
        The filename of the associated JSON file on disk (Default value = None).
    data: non-str Sequence
        The intial data pass to JSONDict
    parent: object
        A parent instance of JSONDict or None (Default value = None).
    write_concern: bool
        Ensure file consistency by writing changes back to a temporary file
        first, before replacing the original file (Default value = None).
    """

    pass


SyncedCollection.register(JSONDict, JSONList)
