# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"""Implements Redis-backend.

This implements the Redis-backend for SyncedCollection API by
implementing sync and load methods.
"""
import zarr
import uuid
import numcodecs

from .synced_collection import SyncedCollection
from .syncedattrdict import SyncedAttrDict
from .synced_list import SyncedList


def get_namespace(class_name):
    """Generate Namespace for a class."""
    return uuid.uuid5(uuid.NAMESPACE_URL, 'signac::'+class_name)


class ZarrCollection(SyncedCollection):
    """Implement sync and load using a Redis backend."""

    backend = __name__  # type: ignore

    def __init__(self, name=None, store=None, redis_kwargs=None, **kwargs):
        self._root = zarr.group(store=store)
        self._name = name
        self._id = None if name is None else uuid.uuid5(get_namespace(type(self).__name__), name)
        super().__init__(**kwargs)
        if (name is None) == (self._parent is None):
            raise ValueError(
                "Illegal argument combination, one of the two arguments, "
                "parent or name must be None, but not both.")

    def _load(self):
        """Load the data from a Radis-database."""
        try:
            dataset = self._root[self._name]
            data = dataset[0]
        except KeyError:
            data = None
        return data

    def _sync(self):
        """Write the data from Radis-database."""
        data = self.to_base()
        # Serialize data:
        dataset = self._root.require_dataset(
            self._name, overwrite=True, shape=1, dtype='object', object_codec=numcodecs.JSON())
        dataset[0] = data


class ZarrDict(ZarrCollection, SyncedAttrDict):
    """A dict-like mapping interface to a persistent Redis-database.

    The RedisDict inherits from :class:`~core.collection_api.RedisCollection`
    and :class:`~core.syncedattrdict.SyncedAttrDict`.

    .. code-block:: python

        doc = RedisDict('data.json', write_concern=True)
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

        While the RedisDict object behaves like a dictionary, there are
        important distinctions to remember. In particular, because operations
        are reflected as changes to an underlying database, copying (even deep
        copying) a RedisDict instance may exhibit unexpected behavior. If a
        true copy is required, you should use the `to_base()` method to get a
        dictionary representation, and if necessary construct a new JSONDict
        instance: `new_dict = RedisDict(old_dict.to_base())`.

    Parameters
    ----------
    name: str, optional
        The name of the  (Default value = None).
    data: mapping, optional
        The intial data pass to JSONDict. Defaults to `list()`
    parent: object, optional
        A parent instance of JSONDict or None (Default value = None).
    write_concern: bool, optional
        Ensure file consistency by writing changes back to a temporary file
        first, before replacing the original file (Default value = None).
    """

    pass


class ZarrList(ZarrCollection, SyncedList):
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
        are reflected as changes to an underlying database, copying (even deep
        copying) a JSONList instance may exhibit unexpected behavior. If a
        true copy is required, you should use the `to_base()` method to get a
        dictionary representation, and if necessary construct a new RedisList
        instance: `new_list = RedisList(old_list.to_base())`.

    Parameters
    ----------
    filename: str
        The filename of the associated JSON file on disk (Default value = None).
    data: non-str Sequence
        The intial data pass to ZarrDict
    parent: object
        A parent instance of ZarrDict or None (Default value = None).
    write_concern: bool
        Ensure file consistency by writing changes back to a temporary file
        first, before replacing the original file (Default value = None).
    """

    pass


SyncedCollection.register(ZarrDict, ZarrList)
