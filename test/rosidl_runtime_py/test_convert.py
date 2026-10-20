# Copyright 2017-2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
from collections import OrderedDict

import numpy
import pytest
import rosidl_parser.definition

from rosidl_runtime_py import message_to_csv
from rosidl_runtime_py import message_to_ordereddict
from rosidl_runtime_py import message_to_yaml
from rosidl_runtime_py.convert import _convert_value

from test_msgs import message_fixtures
from test_msgs.msg import Arrays
import yaml


@pytest.mark.parametrize('value', [
    array.array('B', [0, 42, 255]),
    array.array('i', [-1, 0, 42]),
    array.array('f', [-1.5, 0.0, 42.5]),
    numpy.array([-1.5, 0.0, 42.5], dtype=numpy.float32),
    numpy.array([-1.5, 0.0, 42.5], dtype=numpy.float64),
    numpy.array([0, 42, 255], dtype=numpy.uint8),
    array.array('B'),
    numpy.array([], dtype=numpy.float32),
])
def test_convert_numeric_arrays(value):
    expected = value.tolist()
    result = _convert_value(value)
    assert result == expected
    assert all(type(actual) is type(wanted) for actual, wanted in zip(result, expected))
    assert _convert_value({'nested': [value]}) == {'nested': [expected]}
    for limit in (0, 2, 10):
        truncated = expected[:limit] + ['...'] if len(expected) > limit else expected
        assert _convert_value(value, truncate_length=limit) == truncated
    field_type = rosidl_parser.definition.UnboundedSequence(
        rosidl_parser.definition.BasicType('uint8'))
    assert _convert_value(value, field_type=field_type, no_arr=True) == (
        '<sequence type: uint8, length: {}>'.format(len(value)))
    assert value.tolist() == expected
    assert _convert_value(value, no_str=True) == expected


def test_convert_non_numeric_arrays():
    values = [array.array('u', 'abc'), numpy.array(['a', 'b', 'c'])]
    for value in values:
        assert _convert_value(value) == ['a', 'b', 'c']
        assert _convert_value(value, no_str=True) == ['<string length: <1>>'] * 3
    assert _convert_value(numpy.array(['abc', 42], dtype=object), no_str=True) == [
        '<string length: <3>>', 42]


def test_array_message_formatters():
    msg = Arrays(float32_values=numpy.array([-1.5, 0.0, 42.5], dtype=numpy.float32))
    expected = [-1.5, 0.0, 42.5]
    result = message_to_ordereddict(msg)
    assert result['float32_values'] == expected
    assert all(type(value) is float for value in result['float32_values'])
    assert yaml.safe_load(message_to_yaml(msg))['float32_values'] == expected


def test_primitives():
    # Smoke-test the formatters on a bunch of messages
    msgs = []
    msgs.extend(message_fixtures.get_msg_arrays())
    msgs.extend(message_fixtures.get_msg_basic_types())
    msgs.extend(message_fixtures.get_msg_bounded_sequences())
    msgs.extend(message_fixtures.get_msg_builtins())
    msgs.extend(message_fixtures.get_msg_constants())
    msgs.extend(message_fixtures.get_msg_defaults())
    msgs.extend(message_fixtures.get_msg_empty())
    msgs.extend(message_fixtures.get_msg_multi_nested())
    msgs.extend(message_fixtures.get_msg_nested())
    msgs.extend(message_fixtures.get_msg_strings())
    msgs.extend(message_fixtures.get_msg_unbounded_sequences())
    for m in msgs:
        message_to_csv(m, truncate_length=100)
        message_to_csv(m, truncate_length=None)
        message_to_ordereddict(m, truncate_length=100)
        message_to_ordereddict(m, truncate_length=None)
        message_to_yaml(m, truncate_length=100)
        message_to_yaml(m, truncate_length=None)


def test_convert_primitives():
    assert 5 == _convert_value(5)
    assert 5 == _convert_value(5, truncate_length=0)
    assert 5 == _convert_value(5, truncate_length=1)
    assert 5 == _convert_value(5, truncate_length=10000)
    assert 42.0 == _convert_value(42.0)
    assert 42.0 == _convert_value(42.0, truncate_length=0)
    assert 42.0 == _convert_value(42.0, truncate_length=1)
    assert 42.0 == _convert_value(42.0, truncate_length=10000)
    assert True is _convert_value(True)
    assert True is _convert_value(True, truncate_length=0)
    assert True is _convert_value(True, truncate_length=1)
    assert True is _convert_value(True, truncate_length=10000)
    assert False is _convert_value(False)
    assert False is _convert_value(False, truncate_length=0)
    assert False is _convert_value(False, truncate_length=1)
    assert False is _convert_value(False, truncate_length=10000)


def test_convert_tuple():
    assert (1, 2, 3) == _convert_value((1, 2, 3))
    assert ('...',) == _convert_value((1, 2, 3), truncate_length=0)
    assert (1, 2, '...') == _convert_value((1, 2, 3), truncate_length=2)
    assert ('123', '456', '789') == _convert_value(('123', '456', '789'))
    assert ('12...', '45...', '...') == _convert_value(('123', '456', '789'), truncate_length=2)
    assert ('123', '456', '789') == _convert_value(('123', '456', '789'), truncate_length=5)


def test_convert_list():
    assert [1, 2, 3] == _convert_value([1, 2, 3])
    assert ['...'] == _convert_value([1, 2, 3], truncate_length=0)
    assert [1, 2, '...'] == _convert_value([1, 2, 3], truncate_length=2)
    assert ['123', '456', '789'] == _convert_value(['123', '456', '789'])
    assert ['12...', '45...', '...'] == _convert_value(['123', '456', '789'], truncate_length=2)
    assert ['123', '456', '789'] == _convert_value(['123', '456', '789'], truncate_length=5)


def test_convert_str():
    assert 'hello world' == _convert_value('hello world')
    assert 'hello...' == _convert_value('hello world', truncate_length=5)
    assert 'hello world' == _convert_value('hello world', truncate_length=1000)


def test_convert_bytes():
    assert 'hello world' == _convert_value(b'hello world')
    assert 'hello...' == _convert_value(b'hello world', truncate_length=5)
    assert 'hello world' == _convert_value(b'hello world', truncate_length=1000)


def test_convert_ordered_dict():
    assert OrderedDict([(1, 'a'), ('2', 'b')]) == _convert_value(
        OrderedDict([(1, 'a'), ('2', 'b')]))
    assert OrderedDict([(1, 'a'), ('2', 'b')]) == _convert_value(
        OrderedDict([(1, 'a'), ('2', 'b')]), truncate_length=1)
    assert OrderedDict([(1, 'a'), ('2', 'b')]) == _convert_value(
        OrderedDict([(1, 'a'), ('2', 'b')]), truncate_length=1000)
    assert OrderedDict([(1, 'a...'), ('234', 'b...')]) == _convert_value(
        OrderedDict([(1, 'abc'), ('234', 'bcd')]), truncate_length=1)


def test_convert_dict():
    assert {1: 'a', '2': 'b'} == _convert_value({1: 'a', '2': 'b'})
    assert {1: 'a', '2': 'b'} == _convert_value({1: 'a', '2': 'b'}, truncate_length=1)
    assert {1: 'a', '2': 'b'} == _convert_value({1: 'a', '2': 'b'}, truncate_length=1000)
    assert {1: 'a...', '234': 'b...'} == _convert_value(
        {1: 'abc', '234': 'bcd'}, truncate_length=1)
