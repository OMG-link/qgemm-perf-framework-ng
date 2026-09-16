# K1 `perf cpu-clock` PC semantics calibration

## Validity and artifacts

This is an independent `machine-benchmarks/perf-pc-semantics` ELF; no existing kernel was changed. It ran on `spacemit-k1`, CPU0, performance governor at 1.6 GHz. Final ELF SHA-256 is `c445067c4c720abf34fc029bdf2d8c43e46bddebe6a9f13b1d28c0c1e2e53049`. The final linked assembly was audited with `llvm-objdump`; relaxation did not merge or compress marker instructions (`.option norvc`, 4-byte unique PCs, 64/256-byte block alignment).

Final evidence is in `build/machine-benchmarks/pc-final/` (exact ELF, environment, five `perf.data`, `perf script`, direct outputs per case). External single-event `perf stat -e cycles:u --no-scale` evidence is in `build/machine-benchmarks/pc-calibration/*/stat-valid-*`. The complete per-marker, per-run absolute-PC counts (including explicit zeros) are retained in the version-controlled [`2026-09-04-perf-cpu-clock-pc-semantics-pc-counts.json`](2026-09-04-perf-cpu-clock-pc-semantics-pc-counts.json). Every stat run reports 100.00% running; the initial invalid 33% run, caused by competing with the benchmark's internal pinned cycle counter, was rejected and rerun with `--no-counter`. Direct and profile runs use one internal pinned `cycles:u` event and report 100% running.

Sampling started at `perf record -e cpu-clock:u -c 200000` with no call graph. Eight cases met the <5% median hardware-cycle perturbation requirement at that period. `dyn_cold` measured 5.54% at period 400000 and was rejected; its accepted rerun used period 800000, retained 2258–2372 local samples per run, and measured 2.35% perturbation. Samples are parsed by absolute non-PIE PC, never by source line or symbol percentage.

## Final marker map

| Marker | Absolute PC/range |
|---|---:|
| executed corridor branch / post-corridor control | `0x12e44` / `0x12f44` |
| never-executed corridor | `0x12e80..0x12f3c` (192 B) |
| RAW vmadot 1..16 | `0x13040,0x13048,...,0x130b8` |
| unique sentinel 1..16 | `0x13044,0x1304c,...,0x130bc` |
| nop control slots / sentinels | `0x13140,0x13148,...,0x131b8` / `+4` |
| shared pointer load / immediate control | `0x13240` / `0x13244,0x13260` |
| pointer independent gap | `0x1324c,0x13250,0x13254,0x13258` |
| Dyn B load | `0x13354` |
| Dyn A loads | `0x1335c,0x13364,0x1336c,0x13374` |
| Dyn vmadot 1..16 | `0x1337c..0x133b8` |

The pointer benchmark uses exactly the same static dependent `ld a1,0(a1)` at `0x13240` for all four variants. Hot is a prebuilt 16 KiB cycle. Cold is a 64 MiB, cache-line-granular Fisher–Yates random Hamiltonian cycle, greater than the 512 KiB shared L2/LLC; each next address is data-dependent, so address prefetch cannot run ahead. Smoke cycles/load were about 13 hot versus 494 cold. Dyn hot wraps 64 iterations (8 KiB A + 16 KiB B); cold wraps 32768 iterations (4 MiB A + 8 MiB B), with a 96 MiB cache thrash before measurement.

## Raw results

Each cell is `direct internal cycles / external stat cycles / profile internal cycles / local PC samples`. External stat is whole-process and is reported independently; perturbation compares matched direct/profile internal timed-body medians.

| case | run 1 | run 2 | run 3 | run 4 | run 5 | median profile perturbation |
|---|---:|---:|---:|---:|---:|---:|
| corridor | 1934623353/2902425896/1955473289/6109 | 1932348277/2895789425/1953212284/6104 | 1937814804/2906565252/1953705145/6104 | 1934608696/2896425037/1951120419/6096 | 1935162360/2904290800/1954332483/6105 | 0.99% |
| vmadot_raw | 2888384224/3947544550/2916643633/9110 | 2888031604/3861218058/2915854920/9112 | 2888861296/3867558579/2918555617/9115 | 2888546525/3850181349/2917489283/9112 | 2889932547/3915189844/2917143654/9113 | 0.99% |
| vmadot_nop | 2707734786/3671664002/2733677589/8538 | 2707850696/3667212950/2733731426/8542 | 2705353396/3669847802/2735562124/8540 | 2709755840/3682118631/2732942732/8540 | 2710603121/3670475364/2733617065/8540 | 0.95% |
| ptr_hot_immediate | 905653529/1920636476/915783070/2862 | 902666692/1992711872/919266560/2873 | 903291774/1953673352/913662014/2855 | 906305047/1942254574/915306322/2860 | 902679859/1954497410/954456375/2983 | 1.38% |
| ptr_hot_gap | 1390328065/2323001640/1382015573/4319 | 1353754393/2318794523/1371803787/4282 | 1358191984/2328000493/1368491290/4277 | 1354400410/2321733166/1369487425/4277 | 1353835955/2317031883/1370616617/4281 | 1.20% |
| ptr_cold_immediate | 2648210965/3667959133/2687175096/8393 | 2619910534/3607728741/2689160201/8397 | 2635544803/3589517390/2691468992/8408 | 2729322659/3629700961/2688604776/8398 | 2648469700/3631836646/2791614536/8722 | 1.55% |
| ptr_cold_gap | 2615175604/3752918062/2676095132/8360 | 2757227993/3624778414/2789790236/8700 | 2613774434/3610718088/2718311069/8486 | 2687474409/3641998438/2758218366/8555 | 2625660334/3758251435/2701649337/8438 | 3.53% |
| dyn_hot | 2793402287/3763559288/2840649810/8877 | 2789601132/3755851236/2848700235/8902 | 2790445056/3752469762/2845145362/8890 | 2808672813/3760584854/2839900787/8874 | 2790126355/3753033643/2842960121/8882 | 1.88% |
| dyn_cold (period 800000) | 3079887393/3893305956/3039156120/2372 | 2846212581/3831782568/2913461858/2273 | 2844142432/4042050367/2998033423/2340 | 2855302521/3836004102/2893189169/2258 | 2926416322/3793661952/2922357493/2283 | 2.35% |

Sample-count repeat spread is small: ≤0.21% for corridor, ≤0.18% for RAW vmadot, ≤0.05% for nop, about 0.9% for hot-gap and Dyn hot; cold cases have expected cache/system noise (up to about 4.8% local-count spread, and 5.0% for accepted Dyn cold).

## Absolute-PC distributions and answers

1. **Are never-executed addresses sampled?** No. Across 30,518 corridor-local samples, `0x12e80..0x12f3c` received **0**. Samples split between the taken jump at `0x12e44` (15,227) and post-corridor branch at `0x12f44` (15,291). Thus observed skid did not cross even the 60-byte pre-corridor padding, much less the 192-byte corridor; later interpretations do not need a quantified nonzero corridor-skid correction.

2. **Where does RAW vmadot stall land?** Almost entirely on the following sentinel, not the vmadot PC. RAW totals: vmadot PCs **435**, sentinels **45,127** (98.99% of these two groups). The highest individual PCs are sentinels such as `0x130b4` 2,970, `0x130ac` 2,938, `0x1309c` 2,903. Nop control totals are nop slots **2,436**, following scalar instructions **40,264**, so this machine's `cpu-clock` attribution generally advances to a subsequent instruction; RAW adds a stronger and very uniform sentinel concentration. It is therefore unsafe to label a hot sentinel as that scalar instruction's own cost.

3. **Where does a load miss land?** On the immediately following control PC `0x13244`, not on the load. Hot-immediate: load/control/backedge = **3,996/7,212/3,222**. Cold-immediate: **130/42,146/42**. The cold-minus-hot new mass is overwhelmingly at `0x13244`. This is a direct demonstration that a dependent load miss is charged to the next control instruction in this layout.

4. **How does an independent gap move PC?** Hot gap spreads samples across load `0x13240` 3,496, control `0x13244` 3,656, and independent PCs `0x1324c/50/54/58` = **3,511/3,611/3,554/3,607**. For cold gap, however, the miss remains at the first post-load control PC: `0x13244` **42,251**, while load is 77 and gap PCs total only 180. Independent instructions after that control do not drag the outstanding miss attribution forward; they only distribute hot execution samples.

5. **Where do vector hot/cold added samples land?** Because accepted cold uses twice the period, compare shares rather than raw totals. Hot (44,425 local): B+A load PCs **2,870** (6.46%), vmadot PCs **16,334** (36.77%). Cold (11,526 local): B+A load PCs **373** (3.24%), vmadot PCs **2,100** (18.22%). The dominant PCs are instead the scalar address immediately after the first A load, `0x13360`: hot **12,704** (28.60%), cold **5,101** (44.26%); other post-load address PCs `0x13368/70/78` also rise collectively from 13.21% hot to 22.69% cold. Thus the controlled cold working set adds relative sample mass chiefly to instructions following vector loads, especially `0x13360`, not to the vector-load PCs themselves. The exact sequence is one `vl8re8` (`objdump` canonical `vl8r.v`), four `vle8.v`, and sixteen `smt.vmadot` with the same fixed register topology as the DPU inner loop.

These results calibrate PC attribution only; `cpu-clock` samples are not hardware-cycle counts. Hardware cycles are reported separately above.
