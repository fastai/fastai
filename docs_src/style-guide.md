# fastai docment Sprint Style Guide

## Introduction

This is an extreme TL;DR of the style guide, providing you with direction on how to properly "docment" a function. For the full style guide, you should follow [the style documentation](https://docs.fast.ai/dev/style.html).

## Typing

If typing is an explicit known, then it should be used. If it is more generalized, we avoid typing and instead opt for
a clear `docment` (more on this later)

For example, any `Transform` should always have types since they rely off `TypeDispatch`, such as:

```python
class MyTransform(Transform):
    def encodes(self, x:(PILImage, Image.Image)):
        ...
```

A return type is typically undocumented unless what went in is radically different to what comes out. Instead opting towards
having `docments` to explain what is being returned, if a type wouldn't make sense.

**Why is `typing` not included?**

Rather than using the `typing` library, standard conventions to specify *and* or *or* are used instead such as:

- `(TypeA, TypeB)` denoting `TypeA` **or** `TypeB`.

This is a practical design choice, aiming for less key-strokes while still being readable.

## Documenting functions and classes

Typically you want to aim for one-line docstrings unless it is *absolutely* necessary.

Along with this, you combine more explainability into a function through [docments](https://fastcore.fast.ai/docments). These are
comments next to each input variable allowing for us to describe how each variable should be written.

As an example, here is a "well documented" function definition:

```python
def addition(
    a:(int, float), # The first number to be added
    b:(int, float), # The second number to be added
) -> (int, float):
    "Adds two numbers together"
    return a+b
```
Very clearly you can see that even though the docstring is quite small, when combined with the docments we get a very clear return.

Combine this with the `show_doc` functionality in `nbdev` (which is how you should peruse the documentation), and you are presented
with a clear documentation snippet:

```python
>>> from nbdev.showdoc import show_doc

>>> show_doc(addition)
```

<h4 id="addition" class="doc_header"><code>addition</code><a href="__main__.py#L1" class="source_link" style="float:right">[source]</a></h4>

> <code>addition</code>(**`a`**:`(<class 'int'>, <class 'float'>)`, **`b`**:`(<class 'int'>, <class 'float'>)`)

Adds two numbers together

||Type|Default|Details|
|---|---|---|---|
|**`a`**|`(int, float)`||The first number to be added|
|**`b`**|`(int, float)`||The second number to be added|
|**Returns**|`(int, float)`|||

## Additional Style Examples

When docmenting a class method with `self` or `cls`, keep self on the method definition. Types are placed before the default values, if applicable:

```python
class Math:
    ...

    def addition(self,
        a:(int, float)=4, # The first number to be added
        b:(int, float)=2, # The second number to be added
    ) -> (int, float):
        "Adds two numbers together"
        return a+b
```

If a method returns itself, such as a transform patch, there is no need to add the return type:

```python
@patch
def dihedral(x:TensorImage,
    k:int, # Dihedral transformation to apply
):
    if k in [1,3,4,7]: x = x.flip(-1)
    if k in [2,4,5,7]: x = x.flip(-2)
    if k in [3,5,6,7]: x = x.transpose(-1,-2)
    return x
```

## Notes

* Don't use `typing` library.
* Arguments should have one indent relative to method definition.
* Only `cls` and `self` should be on same line of method name and first argument of `@patch` like `x` on `dihedral` sample.
* Closing parenthesis should be aligned with `def`.
* Keyword arguments (`**kwags`) should not be documented.
