# SpacemiT X60 vector instruction throughput

This table records only stable steady-state throughput values that converge to
a clear hardware rate. Measurements use a 256-bit VLEN, CPU 0 on
`spacemit-k1`, independent registers where needed, hot L1 data for memory
operations, and sufficiently long unrolling to make loop/front-end overhead
negligible. `Cycles/inst` is throughput, not dependency latency.

Keep one configuration per row and sort first by mnemonic, then by vector
configuration. New results should normally require adding only one table row.

| Mnemonic | Vector configuration | Cycles/inst | Inst/cycle |
|---|---|---:|---:|
| `smt.vmadot` | `e8,m1`; M2 accumulator | 1 | 1 |
| `vadd.vi` | `e8,m1` | 1 | 1 |
| `vand.vi` | `e8,m1` | 2 | 0.5 |
| `vfcvt.f.x.v` | `e32,m2` | 4 | 0.25 |
| `vfmacc.vv` | `e32,m2` | 2 | 0.5 |
| `vfmul.vf` | `e32,m2` | 2 | 0.5 |
| `vfwcvt.f.f.v` | `e16,m1` to `e32,m2` | 4 | 0.25 |
| `vl1re16.v` | One 256-bit register | 2 | 0.5 |
| `vl2re32.v` | Two registers / 512 bits | 4 | 0.25 |
| `vle8.v` | `e8,m1` | 2 | 0.5 |
| `vmv2r.v` | Two registers / 512 bits | 4 | 0.25 |
| `vse32.v`, masked | `e32,m1`; arbitrary mask | 3 | 0.333333 |
| `vsrl.vi` | `e8,m1` | 2 | 0.5 |

## Exclusion rule

Do not add a result merely because repeated runs are numerically consistent.
Record it only when different unroll lengths and relevant operand/address
variants converge to a clear hardware rate. In particular, the current
measurements do not establish a context-independent rate for unmasked
`vse32.v e32,mf2`, unmasked `vse32.v e32,m1`, or multi-address `vs2r.v`.