# Copyright (c) 2022 Google LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Tests for NamedArray support in jaxtyping."""

import collections

import jax.numpy as jnp
import jax.random as jr
import pytest
from penzai.core.named_axes import NamedArray

from jaxtyping import Float32, Shaped

from .helpers import ParamError


def test_basic_named_axes(jaxtyp, typecheck, getkey):
    """Test annotation with only named axes."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    # Create a valid NamedArray
    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=jr.normal(getkey(), (4, 8)),
    )
    f(arr)


def test_wrong_axis_names(jaxtyp, typecheck, getkey):
    """Test that wrong axis names fail."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    # Wrong names
    arr = NamedArray(
        named_axes=collections.OrderedDict([("foo", 4), ("bar", 8)]),
        data_array=jr.normal(getkey(), (4, 8)),
    )
    with pytest.raises(ParamError):
        f(arr)


def test_wrong_axis_order(jaxtyp, typecheck, getkey):
    """Test that wrong axis order fails."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    # Right names but wrong order
    arr = NamedArray(
        named_axes=collections.OrderedDict([("seq", 8), ("batch", 4)]),
        data_array=jr.normal(getkey(), (8, 4)),
    )
    with pytest.raises(ParamError):
        f(arr)


def test_wrong_axis_count(jaxtyp, typecheck, getkey):
    """Test that wrong number of axes fails."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq heads"]):
        pass

    # Too few axes
    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=jr.normal(getkey(), (4, 8)),
    )
    with pytest.raises(ParamError):
        f(arr)


def test_dimension_binding(jaxtyp, typecheck, getkey):
    """Test that dimension sizes are bound across arguments."""

    @jaxtyp(typecheck)
    def f(
        x: Float32[NamedArray, "batch seq"],
        y: Float32[NamedArray, "batch hidden"],
    ):
        pass

    arr1 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=jr.normal(getkey(), (4, 8)),
    )
    arr2 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("hidden", 16)]),
        data_array=jr.normal(getkey(), (4, 16)),
    )
    f(arr1, arr2)

    # Mismatched batch size should fail
    arr3 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 5), ("hidden", 16)]),
        data_array=jr.normal(getkey(), (5, 16)),
    )
    with pytest.raises(ParamError):
        f(arr1, arr3)


def test_named_and_positional(jaxtyp, typecheck, getkey):
    """Test annotation with both named and positional axes."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq | height width"]):
        pass

    # Create NamedArray with positional prefix
    data = jr.normal(getkey(), (32, 32, 4, 8))  # pos: (32, 32), named: (4, 8)
    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=data,
    )
    f(arr)


def test_only_positional_with_pipe(jaxtyp, typecheck, getkey):
    """Test annotation with only positional axes via | syntax."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "| height width"]):
        pass

    data = jr.normal(getkey(), (32, 32))
    arr = NamedArray(
        named_axes=collections.OrderedDict(),
        data_array=data,
    )
    f(arr)


def test_unexpected_positional_axes(jaxtyp, typecheck, getkey):
    """Test that positional axes fail when not expected."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    # Array has positional axes but annotation doesn't expect them
    data = jr.normal(getkey(), (32, 4, 8))  # pos: (32,), named: (4, 8)
    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=data,
    )
    with pytest.raises(ParamError):
        f(arr)


def test_view_rejected(jaxtyp, typecheck, getkey):
    """Test that NamedArrayView fails type check."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4), ("seq", 8)]),
        data_array=jr.normal(getkey(), (4, 8)),
    )
    view = arr.as_namedarrayview()

    with pytest.raises(ParamError):
        f(view)


def test_correct_dtype(jaxtyp, typecheck, getkey):
    """Test that correct dtype passes."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch"]):
        pass

    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jnp.ones(4, dtype=jnp.float32),
    )
    f(arr)


def test_wrong_dtype(jaxtyp, typecheck, getkey):
    """Test that wrong dtype fails."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch"]):
        pass

    arr = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jnp.ones(4, dtype=jnp.int32),
    )
    with pytest.raises(ParamError):
        f(arr)


def test_any_dtype_with_shaped(jaxtyp, typecheck, getkey):
    """Test Shaped allows any dtype."""

    @jaxtyp(typecheck)
    def f(x: Shaped[NamedArray, "batch"]):
        pass

    # Float should work
    arr1 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jnp.ones(4, dtype=jnp.float32),
    )
    f(arr1)

    # Int should also work
    arr2 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jnp.ones(4, dtype=jnp.int32),
    )
    f(arr2)


def test_not_namedarray(jaxtyp, typecheck, getkey):
    """Test that regular arrays fail NamedArray annotation."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, "batch seq"]):
        pass

    # Plain JAX array should fail
    arr = jr.normal(getkey(), (4, 8))
    with pytest.raises(ParamError):
        f(arr)


def test_empty_named_axes(jaxtyp, typecheck, getkey):
    """Test annotation with empty named axes string."""

    @jaxtyp(typecheck)
    def f(x: Float32[NamedArray, ""]):
        pass

    # Scalar-like NamedArray (no named axes, no positional axes)
    arr = NamedArray(
        named_axes=collections.OrderedDict(),
        data_array=jnp.array(1.0),
    )
    f(arr)


def test_positional_dimension_binding(jaxtyp, typecheck, getkey):
    """Test that positional dimension sizes are bound across arguments."""

    @jaxtyp(typecheck)
    def f(
        x: Float32[NamedArray, "batch | height width"],
        y: Float32[NamedArray, "batch | height channels"],
    ):
        pass

    arr1 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jr.normal(getkey(), (32, 32, 4)),
    )
    arr2 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jr.normal(getkey(), (32, 3, 4)),
    )
    f(arr1, arr2)

    # Mismatched height should fail
    arr3 = NamedArray(
        named_axes=collections.OrderedDict([("batch", 4)]),
        data_array=jr.normal(getkey(), (64, 3, 4)),  # height=64, not 32
    )
    with pytest.raises(ParamError):
        f(arr1, arr3)
