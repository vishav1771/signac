# Copyright (c) 2020 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import pytest
import os
import json
from tempfile import TemporaryDirectory
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from hypothesis import given, strategies as st, assume
from string import printable

from signac.core.synced_list import SyncedCollection
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

PRINTABLE_NO_DOTS = printable.replace('.', ' ')

JSON_Data = st.recursive(
    st.none() | st.booleans() | st.floats(allow_nan=False) | st.text(printable),
    lambda children: st.lists(children, 2) | st.dictionaries(
        st.text(PRINTABLE_NO_DOTS), children, min_size=1),  max_leaves=10)

Dict_keys = st.text(PRINTABLE_NO_DOTS)


class TestSyncedCollectionBase():

    @pytest.fixture(autouse=True)
    def synced_collection(self):
        self._tmp_dir = TemporaryDirectory(prefix='jsondict_')
        self._fn_ = os.path.join(self._tmp_dir.name, FN_JSON)
        yield
        self._tmp_dir.cleanup()

    def test_from_base(self):
        sd = SyncedCollection.from_base(filename=self._fn_,
                                        data={'a': 0}, backend='signac.core.jsoncollection')
        assert isinstance(sd, JSONDict)
        assert 'a' in sd
        assert sd['a'] == 0

        # invalid input
        with pytest.raises(ValueError):
            SyncedCollection.from_base(data={'a': 0}, filename=self._fn_)


class TestJSONDict():

    _write_concern = False

    @pytest.fixture
    def synced_dict(self):
        self._tmp_dir = TemporaryDirectory(prefix='jsondict_')
        self._fn_ = os.path.join(self._tmp_dir.name, FN_JSON)
        self._cls = JSONDict
        self._backend_kwargs = {'filename': self._fn_, 'write_concern': self._write_concern}
        yield JSONDict(**self._backend_kwargs)
        self._tmp_dir.cleanup()

    def store(self, data):
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())

    def test_init(self, synced_dict):
        assert len(synced_dict) == 0

    def test_invalid_kwargs(self, synced_dict):
        with pytest.raises(ValueError):
            return type(synced_dict)()

    def test_isinstance(self, synced_dict):
        assert isinstance(synced_dict, SyncedCollection)
        assert isinstance(synced_dict, MutableMapping)
        assert isinstance(synced_dict, self._cls)

    @given(key=Dict_keys, d=JSON_Data)
    def test_set_get(self, synced_dict, key, d):
        synced_dict.clear()
        assert not bool(synced_dict)
        assert len(synced_dict) == 0
        assert key not in synced_dict
        synced_dict[key] = d
        assert bool(synced_dict)
        assert len(synced_dict) == 1
        assert key in synced_dict
        assert synced_dict[key] == d
        assert synced_dict.get(key) == d

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_set_get_explicit_nested(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict.setdefault('a', dict())
        child1 = synced_dict['a']
        child2 = synced_dict['a']
        assert child1 == child2
        assert isinstance(child1, type(child2))
        assert id(child1) == id(child2)
        assert not child1
        assert not child2
        child1[key] = testdata
        assert child1
        assert child2
        assert key in child1
        assert key in child2
        assert child1 == child2
        assert child1[key] == testdata
        assert child2[key] == testdata

    @given(key=Dict_keys, key2=Dict_keys, testdata=JSON_Data)
    def test_copy_value(self, synced_dict, key, key2, testdata):
        synced_dict.clear()
        assert key not in synced_dict
        assert key2 not in synced_dict
        synced_dict[key] = testdata
        assert key in synced_dict
        assert synced_dict[key] == testdata
        synced_dict[key2] = synced_dict[key]
        assert key in synced_dict
        assert synced_dict[key] == testdata
        assert key2 in synced_dict
        assert synced_dict[key2] == testdata

    @given(key1=Dict_keys, key2=Dict_keys, testdata=JSON_Data)
    def test_iter(self, synced_dict, key1, key2, testdata):
        synced_dict.clear()
        assume(key1 != key2)
        d = {key1: testdata, key2: testdata}
        synced_dict.update(d)
        assert key1 in synced_dict
        assert key2 in synced_dict
        for i, key in enumerate(synced_dict):
            assert key in d
            assert d[key] == synced_dict[key]
        assert i == 1

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_delete(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        del synced_dict[key]
        assert len(synced_dict) == 0
        with pytest.raises(KeyError):
            synced_dict[key]
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        del synced_dict[key]
        assert len(synced_dict) == 0
        with pytest.raises(KeyError):
            synced_dict[key]

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_update(self, synced_dict, key, testdata):
        synced_dict.clear()
        d = {key: testdata}
        synced_dict.update(d)
        assert len(synced_dict) == 1
        assert synced_dict[key] == d[key]

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_pop(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        d1 = synced_dict.pop(key)
        assert len(synced_dict) == 0
        assert testdata == d1
        with pytest.raises(KeyError):
            synced_dict[key]
        d2 = synced_dict.pop(key, 'default')
        assert len(synced_dict) == 0
        assert d2 == 'default'

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_popitem(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        key1, d1 = synced_dict.popitem()
        assert len(synced_dict) == 0
        assert key == key1
        assert testdata == d1
        with pytest.raises(KeyError):
            synced_dict[key]

    @given(key1=Dict_keys, key2=Dict_keys, testdata1=JSON_Data, testdata2=JSON_Data)
    def test_values(self, synced_dict, key1, key2, testdata1, testdata2):
        data = {key1: testdata1, key2: {'value_nested': testdata2}}
        synced_dict.reset(data)
        assert key1 in synced_dict
        assert key2 in synced_dict
        for val in synced_dict.values():
            assert not isinstance(val, SyncedCollection)
            assert val in data.values()

    @given(key1=Dict_keys, key2=Dict_keys, testdata1=JSON_Data, testdata2=JSON_Data)
    def test_items(self, synced_dict, key1, key2, testdata1, testdata2):
        data = {key1: testdata1, key2: {'value_nested': testdata2}}
        synced_dict.reset(data)
        assert key1 in synced_dict
        assert key2 in synced_dict
        for key, val in synced_dict.items():
            assert synced_dict[key] == data[key]
            assert not isinstance(val, self._cls)
            assert (key, val) in data.items()

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_reset(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        synced_dict.reset()
        assert len(synced_dict) == 0
        synced_dict.reset({key: 'abc'})
        assert len(synced_dict) == 1
        assert synced_dict[key] == 'abc'

        # invalid input
        with pytest.raises(ValueError):
            synced_dict.reset([0, 1])

    @given(testdata=JSON_Data, testdata1=JSON_Data)
    def test_attr_dict(self, synced_dict, testdata, testdata1):
        synced_dict.clear()
        key = 'test'
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert key in synced_dict
        assert synced_dict[key] == testdata
        assert synced_dict.get(key) == testdata
        assert synced_dict.test == testdata
        del synced_dict.test
        assert len(synced_dict) == 0
        assert key not in synced_dict
        key = 'test2'
        synced_dict.test2 = testdata1
        assert len(synced_dict) == 1
        assert key in synced_dict
        assert synced_dict[key] == testdata1
        assert synced_dict.get(key) == testdata1
        assert synced_dict.test2 == testdata1
        with pytest.raises(AttributeError):
            synced_dict.not_exist

    def test_delete_protected_attr(self, synced_dict):
        # deleting a protected attribute
        synced_dict.load()
        del synced_dict._parent
        # deleting _parent will lead to recursion as _parent is treated as key
        # load() will check for _parent and __getattr__ will call __getitem__ which calls load()
        with pytest.raises(RecursionError):
            synced_dict.load()

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_clear(self, synced_dict, key,  testdata):
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        synced_dict.clear()
        assert len(synced_dict) == 0

    def test_repr(self, synced_dict):
        repr(synced_dict)
        p = eval(repr(synced_dict))
        assert repr(p) == repr(synced_dict)
        assert p == synced_dict

    def test_str(self, synced_dict):
        str(synced_dict) == str(synced_dict.to_base())

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_call(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        assert isinstance(synced_dict(), dict)
        assert synced_dict() == synced_dict.to_base()

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_reopen(self, synced_dict, key, testdata):
        synced_dict.reset({key: testdata})
        del synced_dict  # possibly unsafe
        synced_dict2 = self._cls(**self._backend_kwargs)
        assert len(synced_dict2) == 1
        assert synced_dict2[key] == testdata

    def test_update_recursive(self, synced_dict):
        synced_dict.a = {'a': 1}
        synced_dict.b = 'test'
        synced_dict.c = [0, 1, 2]
        assert 'a' in synced_dict
        assert 'b' in synced_dict
        assert 'c' in synced_dict
        data = {'a': 1, 'c': [0, 1, 3], 'd': 1}
        self.store(data)
        assert synced_dict == data

        # invalid data
        data = [1, 2, 3]
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        with pytest.raises(ValueError):
            synced_dict.load()

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_copy_as_dict(self, synced_dict, key, testdata):
        synced_dict.clear()
        synced_dict[key] = testdata
        copy = dict(synced_dict)
        del synced_dict
        assert key in copy
        assert copy[key] == testdata

    def test_nested_dict(self, synced_dict):
        synced_dict['a'] = dict(a=dict())
        child1 = synced_dict['a']
        child2 = synced_dict['a']['a']
        assert isinstance(child1, type(synced_dict))
        assert isinstance(child1, type(child2))

    def test_nested_dict_with_list(self, synced_dict):
        synced_dict['a'] = [1, 2, 3]
        child1 = synced_dict['a']
        synced_dict['a'].append(dict(a=[1, 2, 3]))
        child2 = synced_dict['a'][3]
        child3 = synced_dict['a'][3]['a']
        assert isinstance(child2, type(synced_dict))
        assert isinstance(child1, type(child3))
        assert isinstance(child1, SyncedCollection)
        assert isinstance(child3, SyncedCollection)

    @given(key=Dict_keys, testdata=JSON_Data)
    def test_write_invalid_type(self, synced_dict, key, testdata):
        synced_dict.clear()

        class Foo(object):
            pass

        synced_dict[key] = testdata
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata
        d2 = Foo()
        with pytest.raises(TypeError):
            synced_dict[key + '2'] = d2
        assert len(synced_dict) == 1
        assert synced_dict[key] == testdata

    def test_keys_with_dots(self, synced_dict):
        with pytest.raises(InvalidKeyError):
            synced_dict['a.b'] = None

    @given(key=st.none() | st.booleans() | st.integers() | st.text(PRINTABLE_NO_DOTS),
           testdata=JSON_Data)
    def test_keys_valid_type(self, synced_dict, key, testdata):

        class MyStr(str):
            pass

        synced_dict.clear()
        synced_dict[key] = testdata
        assert str(key) in synced_dict
        assert synced_dict[str(key)] == testdata
        key2 = MyStr(key)
        synced_dict[key2] = testdata
        assert synced_dict[key2] == testdata

    @given(key1=st.tuples(st.integers(), st.integers()) | st.floats(allow_nan=False),
           key2=st.lists(st.integers()) | st.dictionaries(keys=st.text(), values=st.text()),
           testdata=JSON_Data)
    def test_keys_invalid_type(self, synced_dict, key1, key2, testdata):

        class A:
            pass

        with pytest.raises(KeyTypeError):
            synced_dict[A()] = testdata
        with pytest.raises(KeyTypeError):
            synced_dict[key1] = testdata
        with pytest.raises(TypeError):
            synced_dict[key2] = testdata


class TestJSONList:

    _write_concern = False

    @pytest.fixture
    def synced_list(self):
        self._tmp_dir = TemporaryDirectory(prefix='jsondict_')
        self._fn_ = os.path.join(self._tmp_dir.name, FN_JSON)
        self._cls = JSONList
        self._backend_kwargs = {'filename': self._fn_, 'write_concern': self._write_concern}
        yield self._cls(**self._backend_kwargs)
        self._tmp_dir.cleanup()

    def store(self, data):
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())

    def test_init(self, synced_list):
        assert len(synced_list) == 0

    def test_invalid_kwargs(self, synced_list):
        with pytest.raises(ValueError):
            type(synced_list)()

    def test_isinstance(self, synced_list):
        assert isinstance(synced_list, MutableSequence)
        assert isinstance(synced_list, SyncedCollection)
        assert isinstance(synced_list, self._cls)

    @given(JSON_Data)
    def test_set_get(self, synced_list, testdata):
        synced_list.clear()
        assert not bool(synced_list)
        assert len(synced_list) == 0
        synced_list.append(testdata)
        assert bool(synced_list)
        assert len(synced_list) == 1
        assert synced_list[0] == testdata
        synced_list[0] = 1
        assert bool(synced_list)
        assert len(synced_list) == 1
        assert synced_list[0] == 1

    @pytest.mark.skipif(not NUMPY, reason='test requires the numpy package')
    def test_set_get_numpy_data(self, synced_list):
        data = numpy.random.rand(3, 4)
        data_as_list = data.tolist()
        synced_list.reset(data)
        assert len(synced_list) == len(data_as_list)
        assert synced_list == data_as_list
        data2 = numpy.random.rand(3, 4)
        synced_list.append(data2)
        assert len(synced_list) == len(data_as_list) + 1
        assert synced_list[len(data_as_list)] == data2.tolist()
        data3 = numpy.float_(3.14)
        synced_list.append(data3)
        assert len(synced_list) == len(data_as_list) + 2
        assert synced_list[len(data_as_list) + 1] == data3

    @given(st.lists(JSON_Data))
    def test_iter(self, synced_list, testdata):
        synced_list.reset(testdata)
        for i in range(len(synced_list)):
            assert testdata[i] == synced_list[i]

    @given(JSON_Data)
    def test_delete(self, synced_list, testdata1):
        synced_list.clear()
        synced_list.append(testdata1)
        assert len(synced_list) == 1
        assert synced_list[0] == testdata1
        del synced_list[0]
        assert len(synced_list) == 0
        with pytest.raises(IndexError):
            synced_list[0]

    @given(st.lists(JSON_Data, max_size=5), JSON_Data)
    def test_extend(self, synced_list, testdata1, testdata2):
        synced_list.clear()
        synced_list.extend(testdata1)
        assert len(synced_list) == len(testdata1)
        assert synced_list == testdata1
        synced_list += [testdata2]
        assert len(synced_list) == len(testdata1) + 1
        assert synced_list[len(testdata1)] == testdata2

    @given(st.lists(JSON_Data, max_size=5))
    def test_clear(self, synced_list, testdata):
        synced_list.reset(testdata)
        assert len(synced_list) == len(testdata)
        assert synced_list == testdata
        synced_list.clear()
        assert len(synced_list) == 0

    @given(st.lists(JSON_Data, max_size=5), st.lists(JSON_Data, max_size=5))
    def test_reset(self, synced_list, testdata, testdata1):
        synced_list.reset(testdata)
        assert len(synced_list) == len(testdata)
        assert synced_list == testdata
        synced_list.reset()
        assert len(synced_list) == 0
        synced_list.reset(testdata1)
        assert len(synced_list) == len(testdata1)
        assert synced_list == testdata1

        # invalid inputs
        with pytest.raises(ValueError):
            synced_list.reset({'a': 1})

        with pytest.raises(ValueError):
            synced_list.reset(1)

    @given(JSON_Data)
    def test_insert(self, synced_list, testdata):
        synced_list.reset([1, 2])
        assert len(synced_list) == 2
        synced_list.insert(1, testdata)
        assert len(synced_list) == 3
        assert synced_list[1] == testdata

    @given(st.lists(JSON_Data))
    def test_reversed(self,  synced_list, data):
        synced_list.reset(data)
        assert len(synced_list) == len(data)
        assert synced_list == data
        for i, j in zip(reversed(synced_list), reversed(data)):
            assert i == j

    @given(d1=JSON_Data, d2=JSON_Data)
    def test_remove(self, synced_list, d1, d2):
        synced_list.reset([d1, d2])
        assert len(synced_list) == 2
        synced_list.remove(d1)
        assert len(synced_list) == 1
        assert synced_list[0] == d2
        synced_list.reset([d1, d2, d1])
        synced_list.remove(d1)
        assert len(synced_list) == 2
        assert synced_list[0] == d2
        assert synced_list[1] == d1

    @given(st.lists(JSON_Data))
    def test_call(self, synced_list, testdata):
        synced_list.reset(testdata)
        assert len(synced_list) == len(testdata)
        assert isinstance(synced_list(), list)
        assert not isinstance(synced_list(), SyncedCollection)
        assert synced_list() == testdata

    def test_update_recursive(self, synced_list):
        synced_list.reset([{'a': 1}, 'b', [1, 2, 3]])
        assert synced_list == [{'a': 1}, 'b', [1, 2, 3]]
        data = ['a', 'b', [1, 2, 4], 'd']
        self.store(data)
        assert synced_list == data
        data1 = ['a', 'b']
        self.store(data1)
        assert synced_list == data1

        # inavlid data in file
        data2 = {'a': 1}
        self.store(data2)
        with pytest.raises(ValueError):
            synced_list.load()

    @given(JSON_Data)
    def test_reopen(self, synced_list, testdata):
        synced_list.clear()
        synced_list.append(testdata)
        synced_list.sync()
        del synced_list  # possibly unsafe
        synced_list2 = self._cls(**self._backend_kwargs)
        synced_list2.load()
        assert len(synced_list2) == 1
        assert synced_list2[0] == testdata

    @given(testdata=JSON_Data)
    def test_copy_as_list(self, synced_list, testdata):
        synced_list.clear()
        synced_list.append(testdata)
        assert synced_list[0] == testdata
        copy = list(synced_list)
        del synced_list
        assert copy[0] == testdata

    def test_repr(self, synced_list):
        repr(synced_list)
        p = eval(repr(synced_list))
        assert repr(p) == repr(synced_list)
        assert p == synced_list

    def test_str(self, synced_list):
        str(synced_list) == str(synced_list.to_base())

    @given(st.lists(JSON_Data, max_size=5), JSON_Data)
    def test_nested_list(self, synced_list, testdata_list, testdata2):
        synced_list.reset([testdata_list])
        child1 = synced_list[0]
        child2 = synced_list[0]
        assert child1 == child2
        assert isinstance(child1, type(child2))
        assert isinstance(child1, type(synced_list))
        assert id(child1) == id(child2)
        child1.append(testdata2)
        assert child1 == child2
        assert len(synced_list) == 1
        assert len(child1) == len(testdata_list) + 1
        assert len(child2) == len(testdata_list) + 1
        assert isinstance(child1, type(child2))
        assert isinstance(child1, type(synced_list))
        assert id(child1) == id(child2)
        del child1[0]
        assert child1 == child2
        assert len(synced_list) == 1
        assert isinstance(child1, type(child2))
        assert isinstance(child1, type(synced_list))
        assert id(child1) == id(child2)

    @given(key=Dict_keys, testdata=st.lists(JSON_Data, max_size=5))
    def test_nested_list_with_dict(self, synced_list, key, testdata):
        synced_list.reset([{key: testdata}])
        child1 = synced_list[0]
        child2 = synced_list[0][key]
        assert isinstance(child2, SyncedCollection)
        assert isinstance(child1, SyncedCollection)


class TestJSONListWriteConcern(TestJSONList):

    _write_concern = True


class TestJSONDictWriteConcern(TestJSONDict):

    _write_concern = True
