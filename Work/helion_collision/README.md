# Helion-like Plasma-Plasma Collision

## Physics

Helion Energy は2つの Field-Reversed Configuration（FRC）プラズモイドを対向衝突させ、D-He3 核融合を目指している。このシミュレーションはその簡略版：2つのガウス型プラズマ塊（D+ + He3²⁺ + 電子）が対向衝突し、圧縮・加熱される過程を追う。

| パラメータ | 値 |
|---|---|
| プラズモイド密度（peak） | 1×10²¹ m⁻³ (= 10¹⁵ cm⁻³) |
| デバイ長 λ_D | ~16 μm（T=5keV で計算） |
| セルサイズ | 10 μm < λ_D ✓ |
| 衝突速度 | ±0.05c |
| プラズモイド1 | z = -1.5 mm、+z方向に 0.05c |
| プラズモイド2 | z = +1.5 mm、-z方向に 0.05c |
| 衝突時刻 | t ~ 0.1 ns（step ~ 3000） |
| グリッド | 200×800（ドメイン: 2×8 mm） |
| 種別 | e1, d1, he3_1（左）+ e2, d2, he3_2（右） |

## 実際の Helion との違い

| 項目 | このシミュレーション | 実際の Helion |
|---|---|---|
| 磁場 | なし | FRC 構造（azimuthal B） |
| 密度プロファイル | ガウス分布 | 自己整合 FRC |
| 核融合反応 | なし（オプション） | D-He3 → He4 + p |
| スケール | μm-mm | cm オーダー |

## 観測できる現象

- step ~3000 付近でのプラズマ塊の衝突・圧縮
- 衝突後の密度増大と温度上昇
- `KE`（運動エネルギー）と `FieldEnergy`（電場エネルギー）の時間発展
- 圧縮率の見積もり

## D-D融合反応の追加（オプション）

inputs ファイルのコメントアウト部分を有効にすることで D-D 融合を追加できる（WarpX が D-He3 の断面積を持つかは要確認）：

```
collisions.collision_names = dd_fusion
dd_fusion.type = nuclearfusion
dd_fusion.species1 = d1
dd_fusion.species2 = d2
```

## 実行方法

```bash
cd Work/helion_collision
mpirun -n 1 ../../build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d_helion_collision
```

衝突を見るには ~3000 ステップ必要。GPU で数分程度の見込み。

## 可視化

```python
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
import numpy as np

ts = OpenPMDTimeSeries('./diag1/')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, it in enumerate(ts.iterations[::len(ts.iterations)//3]):
    rho, info = ts.get_field('rho', iteration=it)
    axes[i].imshow(rho.T, aspect='auto', origin='lower',
                   extent=[info.x[0]*1e3, info.x[-1]*1e3,
                           info.z[0]*1e3, info.z[-1]*1e3])
    axes[i].set_title(f't = {ts.t[ts.iterations==it][0]*1e9:.2f} ns')
    axes[i].set_xlabel('x (mm)'); axes[i].set_ylabel('z (mm)')
plt.tight_layout(); plt.show()
```

エネルギー時間発展（Reduced Diagnostics）:
```python
import numpy as np
import matplotlib.pyplot as plt
ke = np.loadtxt('diags/reducedfiles/ParticleEnergy.txt', skiprows=1)
plt.plot(ke[:,1]*1e9, ke[:,2], label='KE')
plt.xlabel('Time (ns)'); plt.ylabel('Energy (J)')
plt.title('Kinetic energy vs time'); plt.show()
```

## 参考

- Helion Energy: https://www.helionenergy.com/
- Steinhauer (2011), Phys. Plasmas 18, 070501 — FRC review
