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
from hypothesis import given, strategies as st
from string import printable

from signac.core.synced_collection import SyncedCollection
from signac.core.jsoncollection import JSONDict
from signac.core.jsoncollection import JSONList
from signac.errors import InvalidKeyError
from signac.errors import KeyTypeError


@pytest.fixture
def testdata():
    return str(uuid.uuid4())


# https://github.com/glotzerlab/signac/compare/feature/hypothesis
# https://github.com/glotzerlab/coxeter


PRINTABLE_NO_DOTS = printable.replace('.', ' ')

JSON_Data = st.recursive(
    st.none() | st.booleans() | st.floats(allow_nan=False) | st.text(printable),
    lambda children: st.lists(children, 2) | st.dictionaries(
        st.text(PRINTABLE_NO_DOTS), children, min_size=1),  max_leaves=20)

JSON_List = st.iterables(JSON_Data)


class TestSyncedCollectionBase():

    _type = None
    _write_concern = False
    FN_JSON = 'test_json_dict.json'
    _tmp_dir = None

    @pytest.fixture(scope='class', autouse=True)
    def synced_collection(self):
        cls = type(self)
        cls._tmp_dir = TemporaryDirectory(prefix='jsondict_')
        cls._fn_ = os.path.join(cls._tmp_dir.name, cls.FN_JSON)
        print(cls.__dict__['_fn_'])
        if cls._type is not None:
            yield cls._type(filename=cls._fn_, write_concern=cls._write_concern)
        else:
            yield
        cls._tmp_dir.cleanup()

    def test_init(self, synced_collection):
        if synced_collection:
            assert len(synced_collection)
        print(self._tmp_dir)
        assert 0 == 1

    def test_invalid_kwargs(self):
        if self._type is not None:
            with pytest.raises(ValueError):
                return self._type()

    # def excute_example(self):


class TestJSONDict(TestSyncedCollectionBase):

    _type = JSONDict

    def test_isinstance(self, synced_collection):
        sd = synced_collection
        assert isinstance(sd, SyncedCollection)
        assert isinstance(sd, MutableMapping)
        assert isinstance(sd, JSONDict)

    @given(key=st.text(PRINTABLE_NO_DOTS), d=JSON_Data)
    def test_set_get(self, synced_collection, key, d):
        sd = synced_collection
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

    def test_dfs_update(self, synced_collection, testdata):
        sd = synced_collection
        sd.a = {'a': 1}
        sd.b = 'test'
        assert 'a' in sd
        assert 'b' in sd
        data = {'a': 1, 'c': 1}
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        assert sd == data

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
    FN_JSON = 'test_json_list.json'

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

    def test_insert(self, synced_collection, testdata):
        jsl = synced_collection
        jsl.reset([1, 2])
        assert len(jsl) == 2
        d = testdata
        jsl.insert(1, d)
        assert len(jsl) == 3
        assert jsl[1] == d

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

    def test_dfs_update(self, synced_collection, testdata):
        sl = synced_collection
        sl.reset([{'a': 1}, 'b'])
        assert sl == [{'a': 1}, 'b']
        data = ['a', 'b', 'c']
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data).encode())
        assert sl == data
        data1 = ['a', 'b']
        with open(self._fn_, 'wb') as file:
            file.write(json.dumps(data1).encode())
        assert sl == data1

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
