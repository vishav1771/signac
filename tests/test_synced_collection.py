# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import pytest
import uuid
import os
import json
from tempfile import TemporaryDirectory
from collections.abc import MutableMapping
from collections.abc import MutableSequence

from signac.core.synced_collection import SyncedCollection
from signac.core.jsoncollection import JSONDict
from signac.core.jsoncollection import JSONList
from signac.errors import InvalidKeyError
from signac.errors import KeyTypeError

try:
    import numpy
    NUMPY = True
except ImportError:
    NUMPY = False

FN_JSON = 'test.json'


@pytest.fixture
def testdata():
    return str(uuid.uuid4())


class TestSyncedCollectionBase():

    _type = None
    _write_concern = False

    @pytest.fixture
    def synced_collection(self):
        self._tmp_dir = TemporaryDirectory(prefix='jsondict_')
        self._fn_ = os.path.join(self._tmp_dir.name, FN_JSON)
        if self._type is not None:
            yield self._type(filename=self._fn_, write_concern=self._write_concern)
        else:
            yield
        self._tmp_dir.cleanup()

    def test_init(self, synced_collection):
        if self._type is not None:
            assert len(synced_collection) == 0

    def test_invalid_kwargs(self):
        if self._type is not None:
            with pytest.raises(ValueError):
                return self._type()

    def test_from_base(self, synced_collection):
        sd = SyncedCollection.from_base(filename=self._fn_, data={'a': 0}, backend='JSON')
        assert isinstance(sd, JSONDict)
        assert 'a' in sd
        assert sd['a'] == 0

        # invalid input
        with pytest.raises(ValueError):
            SyncedCollection.from_base(data={'a': 0}, filename=self._fn_)

    def test_repr(self, synced_collection):
        if self._type is not None:
            repr(synced_collection)
            p = eval(repr(synced_collection))
            assert repr(p) == repr(synced_collection)
            assert p == synced_collection

    def test_str(self, synced_collection):
        if self._type is not None:
            str(synced_collection) == str(synced_collection.to_base())


class TestJSONDict(TestSyncedCollectionBase):

    _type = JSONDict

    def test_isinstance(self, synced_collection):
        sd = synced_collection
        assert isinstance(sd, SyncedCollection)
        assert isinstance(sd, MutableMapping)
        assert isinstance(sd, JSONDict)

    def test_set_get(self, synced_collection, testdata):
        sd = synced_collection
        key = 'setget'
        d = testdata
        sd.clear()
        assert not bool(sd)
        assert len(sd) == 0
        assert key not in sd
        sd[key] = d
        assert bool(sd)
        assert len(sd) == 1
        assert key in sd
        assert sd[key] == d
        assert sd.get(key) == d

    def test_set_get_explicit_nested(self, synced_collection, testdata):
        sd = synced_collection
        key = 'setgetexplicitnested'
        d = testdata
        sd.setdefault('a', dict())
        child1 = sd['a']
        child2 = sd['a']
        assert child1 == child2
        assert isinstance(child1, type(child2))
        assert id(child1) == id(child2)
        assert not child1
        assert not child2
        child1[key] = d
        assert child1
        assert child2
        assert key in child1
        assert key in child2
        assert child1 == child2
        assert child1[key] == d
        assert child2[key] == d

    def test_copy_value(self, synced_collection, testdata):
        sd = synced_collection
        key = 'copy_value'
        key2 = 'copy_value2'
        d = testdata
        assert key not in sd
        assert key2 not in sd
        sd[key] = d
        assert key in sd
        assert sd[key] == d
        assert key2 not in sd
        sd[key2] = sd[key]
        assert key in sd
        assert sd[key] == d
        assert key2 in sd
        assert sd[key2] == d

    def test_iter(self, synced_collection, testdata):
        sd = synced_collection
        key1 = 'iter1'
        key2 = 'iter2'
        d1 = testdata
        d2 = testdata
        d = {key1: d1, key2: d2}
        sd.update(d)
        assert key1 in sd
        assert key2 in sd
        for i, key in enumerate(sd):
            assert key in d
            assert d[key] == sd[key]
        assert i == 1

    def test_delete(self, synced_collection, testdata):
        sd = synced_collection
        key = 'delete'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        del sd[key]
        assert len(sd) == 0
        with pytest.raises(KeyError):
            sd[key]
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        del sd['delete']
        assert len(sd) == 0
        with pytest.raises(KeyError):
            sd[key]

    def test_update(self, synced_collection, testdata):
        sd = synced_collection
        key = 'update'
        d = {key: testdata}
        sd.update(d)
        assert len(sd) == 1
        assert sd[key] == d[key]

    def test_pop(self, synced_collection, testdata):
        sd = synced_collection
        key = 'pop'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        d1 = sd.pop(key)
        assert len(sd) == 0
        assert d == d1
        with pytest.raises(KeyError):
            sd[key]
        d2 = sd.pop(key, 'default')
        assert len(sd) == 0
        assert d2 == 'default'

    def test_popitem(self, synced_collection, testdata):
        sd = synced_collection
        key = 'pop'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        key1, d1 = sd.popitem()
        assert len(sd) == 0
        assert key == key1
        assert d == d1
        with pytest.raises(KeyError):
            sd[key]

    def test_values(self, synced_collection, testdata):
        sd = synced_collection
        data = {'value1': testdata, 'value_nested': {'value2': testdata}}
        sd.reset(data)
        assert 'value1' in sd
        assert 'value_nested' in sd
        for val in sd.values():
            assert not isinstance(val, self._type)
            assert val in data.values()

    def test_items(self, synced_collection, testdata):
        sd = synced_collection
        data = {'item1': testdata, 'item_nested': {'item2': testdata}}
        sd.reset(data)
        assert 'item1' in sd
        assert 'item_nested' in sd
        for key, val in sd.items():
            assert sd[key] == data[key]
            assert not isinstance(val, self._type)
            assert (key, val) in data.items()

    def test_reset(self, synced_collection, testdata):
        sd = synced_collection
        key = 'reset'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        sd.reset()
        assert len(sd) == 0
        d1 = testdata
        sd.reset({'reset': d1})
        assert len(sd) == 1
        assert sd[key] == d1

        # invalid input
        with pytest.raises(ValueError):
            sd.reset([0, 1])

    def test_attr_dict(self, synced_collection, testdata):
        sd = synced_collection
        key = 'test'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert key in sd
        assert sd[key] == d
        assert sd.get(key) == d
        assert sd.test == d
        del sd.test
        assert len(sd) == 0
        assert key not in sd
        d1 = testdata
        key = 'test2'
        sd.test2 = d1
        assert len(sd) == 1
        assert key in sd
        assert sd[key] == d1
        assert sd.get(key) == d1
        assert sd.test2 == d1
        with pytest.raises(AttributeError):
            sd.not_exist

        # deleting a protected attribute
        sd.load()
        del sd._parent
        # deleting _parent will lead to recursion as _parent is treated as key
        # load() will check for _parent and __getattr__ will call __getitem__ which calls load()
        with pytest.raises(RecursionError):
            sd.load()

    def test_clear(self, synced_collection, testdata):
        sd = synced_collection
        key = 'clear'
        d = testdata
        sd[key] = d
        assert len(sd) == 1
        assert sd[key] == d
        sd.clear()
        assert len(sd) == 0

    def test_reopen(self, synced_collection, testdata):
        jsd = synced_collection
        key = 'reopen'
        d = testdata
        jsd[key] = d
        jsd.sync()
        del jsd  # possibly unsafe
        jsd2 = synced_collection
        jsd2.load()
        assert len(jsd2) == 1
        assert jsd2[key] == d

    def test_update_recursive(self, synced_collection, testdata):
        sd = synced_collection
        sd.a = {'a': 1}
        sd.b = 'test'
        sd.c = [0, 1, 2]
        assert 'a' in sd
        assert 'b' in sd
        assert 'c' in sd
        data = {'a': 1, 'c': [0, 1, 2], 'd': 1}
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        assert sd == data

        # invalid data
        data = [1, 2, 3]
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        with pytest.raises(ValueError):
            sd.load()

    def test_copy_as_dict(self, synced_collection, testdata):
        sd = synced_collection
        key = 'copy'
        d = testdata
        sd[key] = d
        copy = dict(sd)
        del sd
        assert key in copy
        assert copy[key] == d

    def test_nested_dict(self, synced_collection):
        self._type = JSONDict
        sd = synced_collection
        sd['a'] = dict(a=dict())
        child1 = sd['a']
        child2 = sd['a']['a']
        assert isinstance(child1, type(sd))
        assert isinstance(child1, type(child2))

    def test_nested_dict_with_list(self, synced_collection):
        self._type = JSONDict
        sd = synced_collection
        sd['a'] = [1, 2, 3]
        child1 = sd['a']
        sd['a'].append(dict(a=[1, 2, 3]))
        child2 = sd['a'][3]
        child3 = sd['a'][3]['a']
        assert isinstance(child2, type(sd))
        assert isinstance(child1, type(child3))
        assert isinstance(child1, JSONList)
        assert isinstance(child3, JSONList)

    def test_write_invalid_type(self, synced_collection, testdata):
        class Foo(object):
            pass

        jsd = synced_collection
        key = 'write_invalid_type'
        d = testdata
        jsd[key] = d
        assert len(jsd) == 1
        assert jsd[key] == d
        d2 = Foo()
        with pytest.raises(TypeError):
            jsd[key + '2'] = d2
        assert len(jsd) == 1
        assert jsd[key] == d

    def test_keys_with_dots(self, synced_collection):
        sd = synced_collection
        with pytest.raises(InvalidKeyError):
            sd['a.b'] = None

    def test_keys_valid_type(self, synced_collection, testdata):
        jsd = synced_collection

        class MyStr(str):
            pass
        for key in ('key', MyStr('key'), 0, None, True):
            d = jsd[key] = testdata
            assert str(key) in jsd
            assert jsd[str(key)] == d

    def test_keys_invalid_type(self, synced_collection, testdata):
        sd = synced_collection

        class A:
            pass
        for key in (0.0, A(), (1, 2, 3)):
            with pytest.raises(KeyTypeError):
                sd[key] = testdata
        for key in ([], {}, dict()):
            with pytest.raises(TypeError):
                sd[key] = testdata


class TestJSONList(TestSyncedCollectionBase):

    _type = JSONList

    def test_isinstance(self, synced_collection):
        sl = synced_collection
        assert isinstance(sl, JSONList)
        assert isinstance(sl, MutableSequence)
        assert isinstance(sl, SyncedCollection)

    def test_set_get(self, synced_collection, testdata):
        sl = synced_collection
        d = testdata
        sl.clear()
        assert not bool(sl)
        assert len(sl) == 0
        sl.append(d)
        assert bool(sl)
        assert len(sl) == 1
        assert sl[0] == d
        d1 = testdata
        sl[0] = d1
        assert bool(sl)
        assert len(sl) == 1
        assert sl[0] == d1

    @pytest.mark.skipif(not NUMPY, reason='test requires the numpy package')
    def test_set_get_numpy_data(self, synced_collection):
        sl = synced_collection
        data = numpy.random.rand(3, 4)
        data_as_list = data.tolist()
        sl.reset(data)
        assert len(sl) == len(data_as_list)
        assert sl == data_as_list
        data2 = numpy.random.rand(3, 4)
        sl.append(data2)
        assert len(sl) == len(data_as_list) + 1
        assert sl[len(data_as_list)] == data2.tolist()
        data3 = numpy.float(3.14)
        sl.append(data3)
        assert len(sl) == len(data_as_list) + 2
        assert sl[len(data_as_list) + 1] == data3

    def test_iter(self, synced_collection, testdata):
        sd = synced_collection
        d1 = testdata
        d2 = testdata
        d = [d1, d2]
        sd.extend(d)
        for i in range(len(sd)):
            assert d[i] == sd[i]
        assert i == 1

    def test_delete(self, synced_collection, testdata):
        sd = synced_collection
        d = testdata
        sd.append(d)
        assert len(sd) == 1
        assert sd[0] == d
        del sd[0]
        assert len(sd) == 0
        with pytest.raises(IndexError):
            sd[0]

    def test_extend(self, synced_collection, testdata):
        sl = synced_collection
        d = [testdata]
        sl.extend(d)
        assert len(sl) == 1
        assert sl[0] == d[0]
        d1 = testdata
        sl += [d1]
        assert len(sl) == 2
        assert sl[0] == d[0]
        assert sl[1] == d1

    def test_clear(self, synced_collection, testdata):
        sd = synced_collection
        d = testdata
        sd.append(d)
        assert len(sd) == 1
        assert sd[0] == d
        sd.clear()
        assert len(sd) == 0

    def test_reset(self, synced_collection):
        sl = synced_collection
        sl.reset([1, 2, 3])
        assert len(sl) == 3
        assert sl == [1, 2, 3]
        sl.reset()
        assert len(sl) == 0
        sl.reset([3, 4])
        assert len(sl) == 2
        assert sl == [3, 4]

        # invalid inputs
        with pytest.raises(ValueError):
            sl.reset({'a': 1})

        with pytest.raises(ValueError):
            sl.reset(1)

    def test_insert(self, synced_collection, testdata):
        jsl = synced_collection
        jsl.reset([1, 2])
        assert len(jsl) == 2
        d = testdata
        jsl.insert(1, d)
        assert len(jsl) == 3
        assert jsl[1] == d

    def test_reversed(self,  synced_collection):
        sl = synced_collection
        data = [1, 2, 3]
        sl.reset([1, 2, 3])
        assert len(sl) == 3
        assert sl == data
        for i, j in zip(reversed(sl), reversed(data)):
            assert i == j

    def test_remove(self, synced_collection):
        jsl = synced_collection
        jsl.reset([1, 2])
        assert len(jsl) == 2
        jsl.remove(1)
        assert len(jsl) == 1
        assert jsl[0] == 2
        jsl.reset([1, 2, 1])
        jsl.remove(1)
        assert len(jsl) == 2
        assert jsl[0] == 2
        assert jsl[1] == 1

    def test_update_recursive(self, synced_collection, testdata):
        sl = synced_collection
        sl.reset([{'a': 1}, 'b', {'c': 1}])
        assert sl == [{'a': 1}, 'b', {'c': 1}]
        data = ['a', 'b', {'c': 1}, 'd']
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        assert sl == data
        data1 = ['a', 'b']
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data1).encode())
        assert sl == data1

        # inavlid data in file
        data2 = {'a': 1}
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data2).encode())
        with pytest.raises(ValueError):
            sl.load()

    def test_reopen(self, synced_collection, testdata):
        jsl = synced_collection
        d = testdata
        jsl.append(d)
        jsl.sync()
        del jsl  # possibly unsafe
        jsl2 = synced_collection
        jsl2.load()
        assert len(jsl2) == 1
        assert jsl2[0] == d

    def test_copy_as_list(self, synced_collection, testdata):
        sl = synced_collection
        d = testdata
        sl.append(d)
        assert sl[0] == d
        copy = list(sl)
        del sl
        assert copy[0] == d

    def test_nested_list(self, synced_collection):
        sl = synced_collection
        sl.reset([1, 2, 3])
        sl.append([2, 4])
        child1 = sl[3]
        child2 = sl[3]
        assert child1 == child2
        assert isinstance(child1, type(child2))
        assert isinstance(child1, self._type)
        assert id(child1) == id(child2)
        child1.append(1)
        assert child2[2] == child1[2]
        assert child1 == child2
        assert len(sl) == 4
        assert isinstance(child1, type(child2))
        assert isinstance(child1, self._type)
        assert id(child1) == id(child2)
        del child1[0]
        assert child1 == child2
        assert len(sl) == 4
        assert isinstance(child1, type(child2))
        assert isinstance(child1, self._type)
        assert id(child1) == id(child2)

    def test_nested_list_with_dict(self, synced_collection):
        sl = synced_collection
        sl.reset([{'a': [1, 2, 3, 4]}])
        child1 = sl[0]
        child2 = sl[0]['a']
        assert isinstance(child2, JSONList)
        assert isinstance(child1, JSONDict)


class TestJSONListWriteConcern(TestJSONList):

    _write_concern = True


class TestJSONDictWriteConcern(TestJSONDict):

    _write_concern = True