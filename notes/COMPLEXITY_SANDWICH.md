# Complexity-sandwich derivation

Let `K_L(f)` be minimum formula gate count in language `L`, let `U(f)` be
minimum deterministic decision-tree leaf count, and let

```text
B(f) = s_0(f) s_1(f) = |E_01(f)|^2 / (|f^-1(0)| |f^-1(1)|),
```

with `B = 0` for a constant function. Here `s_b` is the average number of
bichromatic hypercube edges incident to an input on output side `b`.

## General lower bound

For NAND and NOR, replace every gate by its De Morgan representation and push
negations to literals. This preserves the formula tree's leaves. For
AND/OR/NOT, push every unary NOT to the literals; this also preserves leaves.
Khrapchenko's bound gives

```text
B(f) <= De Morgan leaf size.
```

If the original formula has `b` binary gates and any number of unary gates,
its number of leaves is `b + 1`. Since `b <= K_L(f)`, all three languages obey

```text
B(f) <= K_L(f) + 1.
```

## General constructive decision-tree bounds

At an internal node querying `x`, Shannon expansion is

```text
f = ((not x) and f_0) or (x and f_1).
```

It adds four AND/OR/NOT gates to the two recursively compiled children. The
same multiplexer uses four NAND gates, and its dual uses four NOR gates.

With no free constants, a constant costs at most two AND/OR/NOT gates and at
most five NAND or NOR gates. A full binary decision tree with `U` leaves has
`U - 1` internal nodes. Consequently,

```text
AND/OR/NOT: K(f) + 1 <= 3 + 6(U(f) - 1),
NAND/NOR:   K(f) + 1 <= 6 + 9(U(f) - 1).
```

These bounds hold for every input dimension. They are intentionally separated
from the sharper exhaustive four-input envelopes

```text
NAND/NOR:   K(f) + 1 <= 6 + (9/4)(U(f) - 1),
AND/OR/NOT: K(f) + 1 <= min(3 + (13/9)(U(f) - 1), Q_min(f)),
```

whose affine slopes are empirical extremal constants on the four-input
universe.

## Exact prime-cover construction

For a DNF cube with `l` literals and `z` negated literals, its internal cost is
`l - 1 + z`. Combining `m` cubes costs another `m - 1` OR gates, so the full
DNF has

```text
K + 1 = sum_cubes (l + z).
```

The corresponding CNF weight uses the polarity after complementing a
zero-cube. The implementation solves the exact weighted set-cover problem over
prime cubes for DNF, CNF, `NOT-CNF(not f)`, and `NOT-DNF(not f)`, then takes the
least cost. Positive cube weights ensure a minimum cover can be chosen from
prime cubes. Constant functions are handled separately with `K + 1 = 3`.
