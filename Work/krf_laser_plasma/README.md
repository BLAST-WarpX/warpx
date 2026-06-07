# KrF Laser - Plasma Interaction

## Physics

KrF エキシマレーザー（λ=248 nm）はTi:Sa（800 nm）に比べて臨界密度が約10倍高い：

```
nc(KrF,  248nm) = 1.814e28 m⁻³
nc(TiSa, 800nm) = 1.742e27 m⁻³
nc(KrF) / nc(TiSa) = (800/248)^2 ≈ 10.4
```

**KrFの利点：** 固体密度に近いターゲット（ICFなど）でも nc に近い near-critical regime で相互作用できる。同じ固体水素（n ~ 4.2×10²⁸ m⁻³）に対して：
- 800nm 側：n/nc ≈ 24 → 深く overdense
- 248nm 側：n/nc ≈ 2.3 → near-critical！

このシミュレーションは 2×nc(KrF) のプラズマへのKrFレーザー照射を示す。

| パラメータ | 値 |
|---|---|
| レーザー波長 | 248 nm (KrF excimer) |
| 規格化振幅 a0 | ~2 (e_max = 2.6×10¹³ V/m) |
| nc(KrF) | 1.814×10²⁸ m⁻³ |
| プラズマ密度 | 最大 2×nc(KrF) = 3.63×10²⁸ m⁻³ |
| グリッド | 400×800（セルサイズ 25 nm = λ/10） |
| ドメイン | 10 μm × 20 μm |

**NOTE:** セルサイズ 25 nm は λ=248nm を解像するために必要。ドメインはTiSaのシミュレーションより小さいが、物理は等価。

## 観測できる現象

- レーザーの反射と吸収（Ey フィールド）
- 密度ランプへのレーザー浸透
- 透過条件（a0 > sqrt(2)×n/nc = 2.83）付近での挙動変化
- KrFとTiSaの結果比較（同じ密度でどう変わるか）

## 実行方法

```bash
cd Work/krf_laser_plasma
mpirun -n 1 ../../build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d_krf_laser_plasma
```

## 可視化

```python
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt

ts = OpenPMDTimeSeries('./diag1/')
Ey, info = ts.get_field('Ey', iteration=ts.iterations[-1])
rho, _   = ts.get_field('rho', iteration=ts.iterations[-1])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.imshow(Ey.T, aspect='auto', origin='lower', cmap='RdBu')
ax1.set_title('Ey (laser field)')
ax2.imshow(rho.T, aspect='auto', origin='lower', cmap='viridis')
ax2.set_title('Charge density')
plt.tight_layout(); plt.show()
```

## 参考

- Yano et al. (2019), arXiv:1904.09057
- Nuckolls et al. (1972), Nature — KrF in ICF context
