# Codex 工程規格：以條件式 Level-Set PINN 預測半導體 EPI 沉積與蝕刻製程

- 文件版本：3.0
- 開發階段：初步可行性驗證（Proof of Concept）
- 主要語言：Python 3.11+
- 深度學習框架：PyTorch
- 輸入格式：XLSX
- 標準影像尺寸：`H = 350`、`W = 200`
- 標準陣列形狀：`(H, W) = (350, 200)`
- 模型數量：
  - `DepositionPINN`
  - `EtchPINN`

---

## 0. 給 Codex 的最高優先指令

請依照本規格建立一個可直接安裝與執行的 Python repository，用於初步驗證條件式 Level-Set PINN 是否能預測半導體 EPI 製程中的沉積與蝕刻結果。

本階段只要求完成：

1. XLSX level-set 資料讀取與前處理。
2. 20 點初始零輪廓擷取。
3. 沉積與蝕刻兩個獨立 PINN 模型。
4. 已知平均製程速率的物理約束。
5. 從 `2E` 遞迴預測至 `5E`。
6. 使用真實 `5M`、`5E` 做最終比較。
7. 輸出預測陣列、指標與圖形。

本階段不要求：

- pytest。
- unittest。
- `tests/` 目錄。
- synthetic test code。
- CI/CD。
- 自動化超參數搜尋。
- production API。
- 分散式訓練。
- 完整部署程式。

核心程式不得留下未完成的 `TODO`、空函式或僅有 pseudo-code 的主要流程。

---

## 1. 問題定義

完整製程序列為：

```text
init -> 1M -> 1E -> 2M -> 2E -> 3M -> 3E -> 4M -> 4E -> 5M -> 5E
```

其中：

- `M`：Deposition，沉積。
- `E`：Etch，蝕刻。
- 每個狀態是一張 scalar level-set。
- `phi = 0` 表示材料與空隙的介面。
- 模型以前一個製程狀態預測下一個製程狀態。

已知真實狀態：

```text
init
1M
1E
2M
2E
5M
5E
```

需要模型遞迴產生：

```text
pred_3M
pred_3E
pred_4M
pred_4E
pred_5M
pred_5E
```

最終評估：

```text
pred_5M vs true_5M
pred_5E vs true_5E
```

---

## 2. 製程時間與訓練配對

### 2.1 製程時間

| Cycle | 沉積時間 M | 蝕刻時間 E |
|---:|---:|---:|
| 1 | 9000 s | 50 s |
| 2 | 8000 s | 50 s |
| 3 | 7000 s | 50 s |
| 4 | 6000 s | 50 s |
| 5 | 5000 s | 50 s |

### 2.2 沉積模型訓練資料

| ID | 初始狀態 | 目標狀態 | 時間 |
|---|---|---|---:|
| M1 | `init` | `1M` | 9000 s |
| M2 | `1E` | `2M` | 8000 s |

### 2.3 蝕刻模型訓練資料

| ID | 初始狀態 | 目標狀態 | 時間 |
|---|---|---|---:|
| E1 | `1M` | `1E` | 50 s |
| E2 | `2M` | `2E` | 50 s |

### 2.4 Holdout 使用規則

- `5M` 只能用來評估 `pred_5M`。
- `5E` 只能用來評估 `pred_5E`。
- `5M`、`5E` 不得參與模型訓練。
- `5M`、`5E` 不得參與 normalization statistics。
- `5M`、`5E` 不得用於選擇 checkpoint 或調整 loss weight。
- `pred_5E` 的輸入必須是 `pred_5M`，不可在主要結果中使用真實 `5M` 作為輸入。

---

## 3. 影像尺寸與座標定義

### 3.1 標準尺寸

本專案統一使用：

```text
H = 350
W = 200
```

NumPy 與 PyTorch 的 level-set 陣列排列為：

```text
phi[y, x]
shape = (H, W) = (350, 200)
```

座標範圍：

```text
x index: 0 ... 199
y index: 0 ... 349
```

不可將 `350 x 200` 解讀為 `(W, H)`。

程式中必須定義：

```python
EXPECTED_HEIGHT = 350
EXPECTED_WIDTH = 200
EXPECTED_SHAPE = (350, 200)
```

### 3.2 空間方向

預設影像座標：

```text
x 向右增加
y 向下增加
```

若有實際 TEM pixel size，使用：

```yaml
spatial:
  pixel_size_x: 1.0
  pixel_size_y: 1.0
  unit: pixel
  y_axis_direction: down
```

若已知每個 pixel 對應的實際尺寸，例如 nm，則將 `unit` 設為 `nm` 並填入真實的 `pixel_size_x`、`pixel_size_y`。

---

## 4. XLSX 輸入格式

### 4.1 建議資料配置

預設支援兩個 workbook。

#### 沉積 workbook

```text
deposition.xlsx
├── init
├── 1
├── 2
└── 5
```

對應：

```text
init -> init
1    -> 1M
2    -> 2M
5    -> 5M
```

#### 蝕刻 workbook

```text
etch.xlsx
├── 1
├── 2
└── 5
```

對應：

```text
1 -> 1E
2 -> 2E
5 -> 5E
```

也必須支援單一 workbook，但所有 state 與 sheet 的對應必須在 YAML 中明確設定，不可依名稱模糊猜測。

### 4.2 設定範例

```yaml
data:
  workbooks:
    deposition: data/raw/deposition.xlsx
    etch: data/raw/etch.xlsx

  state_sources:
    init:
      workbook: deposition
      sheet: init

    1M:
      workbook: deposition
      sheet: "1"

    1E:
      workbook: etch
      sheet: "1"

    2M:
      workbook: deposition
      sheet: "2"

    2E:
      workbook: etch
      sheet: "2"

    5M:
      workbook: deposition
      sheet: "5"

    5E:
      workbook: etch
      sheet: "5"
```

### 4.3 XLSX 讀取

使用：

```python
pandas.read_excel(
    workbook_path,
    sheet_name=sheet_name,
    header=None,
    engine="openpyxl",
)
```

每個工作表預設只包含數值矩陣，不包含 header 與 index。

### 4.4 前處理步驟

對每個指定工作表依序執行：

1. 讀取指定 sheet。
2. 移除最外圍整列皆空白的 rows。
3. 移除最外圍整欄皆空白的 columns。
4. 將所有 cell 轉為 numeric。
5. 轉成 `numpy.float64`。
6. 檢查 NaN。
7. 檢查 Inf。
8. 檢查 shape。
9. 必要時 transpose。
10. 轉為 C-contiguous NumPy array。
11. 保存處理後陣列與 metadata。

遇到以下情況直接報錯：

- workbook 不存在。
- sheet 不存在。
- 內部存在非數值資料。
- 內部存在 NaN 或 Inf。
- shape 不符合規則。
- 資料矩陣內部存在無法解釋的空白。
- 公式 cell 無 cached numeric value。

### 4.5 Shape 規則

```text
raw shape = (350, 200)
=> 直接接受

raw shape = (200, 350) 且 allow_transpose = true
=> transpose 成 (350, 200)

其他 shape
=> 報錯
```

不得使用 image resize 將錯誤尺寸強制縮放到 `(350, 200)`。

---

## 5. Level-set 前處理

### 5.1 符號定義

預設：

```text
phi < 0：材料內部
phi = 0：材料與空隙介面
phi > 0：材料外部或空隙
```

設定：

```yaml
level_set:
  sign_convention: negative_inside
```

### 5.2 輸入型態

支援：

```yaml
level_set:
  input_kind: signed_distance
```

或：

```yaml
level_set:
  input_kind: level_set
```

若輸入只是一般 level-set 而不是 Signed Distance Function，建議以前處理方式依 `phi = 0` 的介面重建 signed distance field，因為：

1. Level-Set PDE 中的速率單位需要與距離一致。
2. Eikonal constraint 假設 `|grad(phi)|` 接近 1。
3. 已知平均速率才能正確換算成介面位移。

若輸入其實是 binary mask，則以 distance transform 建立 SDF：

```python
phi = distance_outside - distance_inside
```

使材料內部為負值。

### 5.3 Level-set clipping 與標準化

先裁切：

```math
phi_clip =
clip(phi_sdf, -d_clip, d_clip)
```

再標準化：

```math
tilde_phi =
phi_clip / d_clip
```

因此：

```text
tilde_phi 約落在 [-1, 1]
```

預設：

```yaml
level_set:
  phi_clip_distance: 32.0
```

`phi_clip_distance` 必須由設定檔提供，不得從 `5M` 或 `5E` 計算。

---

## 6. 初始零輪廓的 20 點條件

### 6.1 輪廓定義

每次預測除了輸入完整初始 level-set，也要輸入本次初始介面的 20 個輪廓點。

令：

```text
K = 20
W = 200
H = 350
```

在 x 方向等距取點：

```math
x_k =
k(W-1)/(K-1),
\qquad k = 0,1,...,19
```

即：

```math
x_k =
199k/19
```

對每個 `x_k` 尋找：

```math
phi_0(x_k, y_k) = 0
```

形成：

```text
[(x_0, y_0), ..., (x_19, y_19)]
```

此處 `y_k` 是 level-set 等於 0 的 y 座標，不是把 y 座標固定為 0。

### 6.2 Zero-crossing 擷取

對每個 `x_k`：

1. 若 `x_k` 不是整數 column，先在 x 方向做線性插值。
2. 取得垂直 profile `phi_0(x_k, y)`。
3. 找出相鄰 y 位置的 sign change。
4. 使用線性插值求 zero crossing：

```math
y^* =
y_j -
phi_j
(y_{j+1}-y_j) /
(phi_{j+1}-phi_j+\epsilon)
```

### 6.3 多個 zero crossings

同一個 x 可能有多個 `phi = 0` 位置。

預設策略：

```yaml
contour:
  crossing_policy: closest_to_previous
  first_crossing_policy: topmost
```

規則：

- 第一個有效 x 使用最上方的 zero crossing。
- 後續 x 選擇最接近前一個有效 `y_k` 的 crossing。
- 保存每個 x 的候選 crossing 數量，便於檢查輪廓是否可由單值函數 `y(x)` 表示。

### 6.4 找不到 crossing

若某個 x 找不到 zero crossing：

```text
valid_mask[k] = 0
```

固定長度輸入中的 `y_k` 可由鄰近有效點插值，但模型同時必須收到 `valid_mask`，避免把插值值當作真實觀測。

若有效點少於設定門檻，直接報錯：

```yaml
contour:
  min_valid_points: 10
```

### 6.5 輪廓標準化

x 座標：

```math
xi_k =
2x_k/(W-1)-1 =
2x_k/199-1
```

y 座標：

```math
eta_k =
2y_k/(H-1)-1 =
2y_k/349-1
```

模型輪廓輸入：

```text
contour_points.shape = (20, 3)
```

每個點包含：

```text
[xi_k, eta_k, valid_mask_k]
```

### 6.6 預測介面

必須提供：

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContourCondition:
    points_xy: np.ndarray
    valid_mask: np.ndarray


def predict_next_levelset(
    phi_initial: np.ndarray,
    duration_s: float,
    average_rate: float,
    initial_contour: ContourCondition | None = None,
) -> np.ndarray:
    ...
```

規則：

- 若 `initial_contour=None`，由 `phi_initial` 自動抽取 20 點。
- 若使用者明確傳入 `initial_contour`，模型使用該輪廓作為條件。
- 同時從 `phi_initial` 自動抽取輪廓，檢查兩者是否一致。
- 若平均差異超過設定門檻，輸出 warning 或報錯。

### 6.7 Rollout 中的輪廓更新

每次預測後，下一步都必須從最新預測結果重新抽取 20 點：

```text
true_2E
-> extract contour
-> pred_3M

pred_3M
-> extract contour
-> pred_3E

pred_3E
-> extract contour
-> pred_4M

pred_4M
-> extract contour
-> pred_4E

pred_4E
-> extract contour
-> pred_5M

pred_5M
-> extract contour
-> pred_5E
```

不可在所有步驟重複使用 `2E` 的初始輪廓。

---

## 7. x、y、t 與條件變數的標準化

### 7.1 x 標準化

實體 x 座標：

```math
x = j dx,
\qquad j = 0,...,199
```

```math
L_x = (W-1)dx = 199dx
```

輸入模型前轉換到 `[-1, 1]`：

```math
xi =
2x/L_x - 1
```

因此：

```text
影像最左側 -> xi = -1
影像最右側 -> xi = +1
```

### 7.2 y 標準化

實體 y 座標：

```math
y = i dy,
\qquad i = 0,...,349
```

```math
L_y = (H-1)dy = 349dy
```

輸入模型前轉換到 `[-1, 1]`：

```math
eta =
2y/L_y - 1
```

因此在影像座標 `y_axis_direction=down` 時：

```text
影像最上方 -> eta = -1
影像最下方 -> eta = +1
```

### 7.3 t 標準化

對每一個單次製程 transition：

```text
0 <= t <= Delta_t
```

標準化為：

```math
tau =
t / Delta_t
```

因此：

```text
製程開始 -> tau = 0
製程結束 -> tau = 1
```

不同 cycle 的實際製程時間不同，因此必須另外將 `Delta_t` 當作模型條件輸入，不能只輸入 `tau`。

### 7.4 Duration 標準化

沉積模型：

```math
d_M =
Delta_t / 9000
```

蝕刻模型：

```math
d_E =
Delta_t / 50
```

也可使用訓練資料的固定 reference，但 reference 必須保存在 config 中。

### 7.5 Average rate 標準化

```math
r =
average_rate / rate_reference
```

`rate_reference` 可使用該 process 已知平均速率的固定參考值，不得使用 holdout 統計。

### 7.6 Nominal displacement 條件

```math
q =
average_rate * Delta_t / d_clip
```

模型至少接收：

```text
xi
eta
tau
duration_normalized
rate_normalized
nominal_displacement_normalized
```

### 7.7 Automatic differentiation 的 Chain Rule

網路輸出標準化 level-set `tilde_phi`，實際 level-set：

```math
phi =
s_phi tilde_phi
```

其中：

```math
s_phi = d_clip
```

物理導數必須轉回實際座標：

```math
phi_x =
s_phi (2/L_x)
partial(tilde_phi)/partial(xi)
```

```math
phi_y =
s_phi (2/L_y)
partial(tilde_phi)/partial(eta)
```

```math
phi_t =
s_phi (1/Delta_t)
partial(tilde_phi)/partial(tau)
```

PINN 的 PDE residual 必須使用上述實體尺度導數，不可直接把 normalized derivative 當成物理導數。

---

## 8. 已知平均速率

### 8.1 輸入設定

平均速率使用正的 magnitude：

```text
deposition average rate > 0
etch average rate > 0
```

製程方向由模型內的 sign 決定：

```math
s_M = +1
```

```math
s_E = -1
```

支援：

1. 每個 process 一個固定平均速率。
2. 每個 cycle 各自指定平均速率。

設定優先順序：

```text
cycle-specific rate
> process default rate
```

### 8.2 單位

平均速率必須與 level-set 的距離單位一致，例如：

```text
phi unit = pixel
rate unit = pixel/s
```

或：

```text
phi unit = nm
rate unit = nm/s
```

若輸入速率是 `nm/min`，必須在資料前處理時轉換為 `nm/s`。

### 8.3 速率物理意義

本初步驗證預設：

```yaml
rate_definition: normal_interface_speed
```

也就是平均速率直接代表介面的平均 normal velocity magnitude。

若目前已知的速率其實是平坦區域沿 y 方向量測的厚度速率，需在報告中註明它不一定等於斜面上的 normal velocity。

---

## 9. Level-Set PINN 方程

### 9.1 介面

```math
Gamma(t) =
{(x,y) | phi(x,y,t)=0}
```

### 9.2 法向量

```math
n =
grad(phi) /
sqrt(phi_x^2 + phi_y^2 + epsilon_n^2)
```

### 9.3 Level-Set PDE

```math
partial(phi)/partial(t)
+
V_n ||grad(phi)||_2
=
0
```

PDE residual：

```math
r_pde =
phi_t
+
V_n sqrt(phi_x^2 + phi_y^2 + epsilon_n^2)
```

### 9.4 製程方向

在 `negative_inside` 定義下：

```math
V_n^M > 0
```

```math
V_n^E < 0
```

沉積使材料區域增加，蝕刻使材料區域減少。

---

## 10. Jacobian 與幾何特徵

### 10.1 Level-set 空間 Jacobian

```math
J_xy(phi) =
[phi_x, phi_y]
```

其 norm：

```math
||J_xy(phi)||_2 =
sqrt(phi_x^2 + phi_y^2 + epsilon_n^2)
```

### 10.2 時空 Jacobian

```math
J_xyt(phi) =
[phi_x, phi_y, phi_t]
```

### 10.3 速度場 Jacobian

```math
J_xy(V_n) =
[V_n,x, V_n,y]
```

可使用下列正則化避免速度場過度震盪：

```math
L_JV =
mean(V_n,x^2 + V_n,y^2)
```

### 10.4 Hessian 與曲率

```math
H(phi) =
[[phi_xx, phi_xy],
 [phi_xy, phi_yy]]
```

曲率：

```math
kappa =
(
phi_xx phi_y^2
- 2 phi_xy phi_x phi_y
+ phi_yy phi_x^2
)
/
(phi_x^2 + phi_y^2 + epsilon_n^2)^(3/2)
```

曲率可作為 VelocityNet 的輸入特徵，但初步驗證可由 config 開關控制。

---

## 11. 模型輸入與架構

### 11.1 兩個獨立模型

必須建立：

```python
DepositionPINN
EtchPINN
```

兩者可使用相同 class definition，但必須有獨立：

- parameters
- optimizer
- checkpoint
- training log

### 11.2 單次預測的輸入

每次預測至少包含：

1. 完整初始 level-set `phi_initial`。
2. Query point 的 `xi`、`eta`、`tau`。
3. 本次製程時間。
4. 本次已知平均速率。
5. nominal displacement。
6. 初始零輪廓 20 點。
7. 20 點 validity mask。
8. Query point 對應的初始 local level-set。
9. 可選 local gradient、normal、curvature。

### 11.3 初始場 local sampler

對完整 `phi_initial` 預先計算：

```text
phi0
phi0_x
phi0_y
normal0_x
normal0_y
optional kappa0
```

使用 bilinear interpolation 在任意 query coordinate 取得 local features。

### 11.4 ContourEncoder

輸入：

```text
shape = (20, 3)
```

展平成 60 維後，使用小型 MLP：

```text
Linear(60, 128)
Tanh
Linear(128, 64)
Tanh
```

輸出：

```text
z_contour.shape = (64,)
```

必須保留 20 點的 x 順序，不可只做 unordered mean pooling。

### 11.5 SolutionNet

建議輸入：

```text
xi
eta
tau
duration_normalized
rate_normalized
nominal_displacement_normalized
sampled_phi0
sampled_phi0_x
sampled_phi0_y
sampled_normal0_x
sampled_normal0_y
optional sampled_kappa0
interpolated_contour_y_at_x
relative_y_to_contour
z_contour
```

建議架構：

```text
6 層 MLP
hidden dimension = 128
activation = Tanh
```

PINN 主網路不可使用 ReLU，因為需要平滑的一階與二階導數。

### 11.6 Hard initial condition

若 `phi_initial` 是 SDF，使用已知平均速率建立 nominal solution：

```math
V_ref =
s_process * average_rate
```

```math
phi_nominal(x,y,t) =
phi_0(x,y) - t V_ref
```

網路只學習 correction：

```math
phi_theta =
phi_nominal
+
tau * s_phi * correction_scale * tanh(g_theta)
```

因此：

```math
phi_theta(x,y,0) =
phi_0(x,y)
```

精確成立。

### 11.7 VelocityNet

使用已知平均速率加上 bounded local correction：

```math
V_n =
s_process * average_rate
*
[1 + alpha_v tanh(v_psi)]
```

其中：

```text
0 <= alpha_v < 1
```

預設：

```yaml
model:
  velocity_residual_fraction: 0.5
```

此設計可：

- 保證沉積與蝕刻方向。
- 讓已知平均速率成為主要物理先驗。
- 只讓模型學習局部速度修正。

---

## 12. Loss Functions

### 12.1 Endpoint level-set loss

在 `tau = 1`：

```math
L_sdf =
mean(
w *
SmoothL1(phi_pred - phi_target)
)
```

其中介面附近給較高權重：

```math
w =
1 + alpha exp(-|phi_target|/sigma)
```

### 12.2 Material mask Dice loss

使用 soft mask：

```math
m(phi) =
sigmoid(-phi / epsilon_H)
```

```math
L_dice =
1 -
(
2 sum(m_pred m_gt) + epsilon
)
/
(
sum(m_pred^2) + sum(m_gt^2) + epsilon
)
```

### 12.3 PDE loss

```math
L_pde =
mean(r_pde^2)
```

實作時應對 residual 做適當尺度 normalization，避免沉積與蝕刻的時間尺度差異造成數值不平衡。

### 12.4 Eikonal loss

在介面附近：

```math
L_eikonal =
mean(
(||grad(phi)||_2 - 1)^2
)
```

### 12.5 Average-rate loss

使用介面附近的 smooth delta 權重計算平均速度：

```math
mean_interface_speed =
sum(delta(phi) * |V_n|)
/
(sum(delta(phi)) + epsilon)
```

```math
L_rate =
(
mean_interface_speed - average_rate
)^2
/
(rate_reference^2 + epsilon)
```

### 12.6 Velocity Jacobian loss

```math
L_JV =
mean(
V_n,x^2 + V_n,y^2
)
```

### 12.7 製程方向 loss

沉積：

```math
L_sign_M =
mean(ReLU(phi_t)^2)
```

蝕刻：

```math
L_sign_E =
mean(ReLU(-phi_t)^2)
```

### 12.8 面積單調性

材料面積：

```math
A(tau) =
sum(sigmoid(-phi/epsilon_H))
```

沉積要求：

```text
A(tau_b) >= A(tau_a), tau_b > tau_a
```

蝕刻要求：

```text
A(tau_b) <= A(tau_a), tau_b > tau_a
```

### 12.9 初始輪廓 consistency loss

在有效的 20 個輪廓點：

```math
L_contour =
sum(
mask_k *
|phi_theta(x_k, y_k, 0)|
)
/
(sum(mask_k) + epsilon)
```

### 12.10 總 Loss

```math
L_total =
lambda_sdf L_sdf
+ lambda_dice L_dice
+ lambda_pde L_pde
+ lambda_eikonal L_eikonal
+ lambda_rate L_rate
+ lambda_JV L_JV
+ lambda_sign L_sign
+ lambda_area L_area
+ lambda_contour L_contour
```

初始建議值：

```yaml
loss:
  sdf: 1.0
  dice: 0.5
  pde: 1.0
  eikonal: 0.02
  average_rate: 0.5
  velocity_jacobian: 1.0e-4
  sign: 0.05
  area_monotonic: 0.1
  contour_initial_condition: 0.1
```

所有 loss weight 必須可由 YAML 修改。

---

## 13. Sampling

### 13.1 Endpoint data points

每個 transition：

```text
70%：從 target 的 phi = 0 附近 narrow band 取樣
30%：從完整影像均勻取樣
```

### 13.2 PDE collocation points

建議：

```text
60%：從初始介面附近取樣
20%：從 20 點輪廓附近取樣
20%：完整空間均勻取樣
```

時間：

```text
tau 隨機取樣於 (0, 1)
```

並可加強接近 `tau = 0`、`tau = 1` 的取樣。

### 13.3 Transition 平衡

沉積的 M1 與 M2 必須等權取樣。

蝕刻的 E1 與 E2 必須等權取樣。

不可因某張影像的有效 narrow-band pixel 較多，就讓該 transition 主導訓練。

---

## 14. 訓練流程

### 14.1 初步驗證流程

對沉積與蝕刻模型分別執行：

1. 載入 training states。
2. 建立 SDF 與 20 點初始輪廓。
3. 建立 endpoint samples。
4. 建立 PDE collocation samples。
5. 使用 Adam 訓練。
6. 保存最低 training proxy loss 的 checkpoint。
7. 輸出 loss curve。
8. 凍結 checkpoint。
9. 執行 rollout。
10. 最後才載入 `5M`、`5E` 評估。

### 14.2 建議設定

```yaml
training:
  seed: 42
  device: auto
  dtype: float64

  adam_steps: 10000
  adam_lr: 1.0e-3

  endpoint_batch_size: 4096
  collocation_batch_size: 4096

  grad_clip_norm: 10.0

  checkpoint_every: 500
  log_every: 100

  use_lbfgs: false
```

初步驗證先使用單一 seed。

### 14.3 Checkpoint selection

由於只有兩個 transition，沒有獨立 validation set。

checkpoint 僅依下列 training proxy 選擇：

- total training loss
- endpoint loss
- PDE residual
- average-rate consistency
- numerical stability

報告中不得將這個 checkpoint selection 稱為 unbiased validation。

---

## 15. 遞迴預測流程

從真實 `2E` 開始：

```python
phi = true_2E

pred_3M = deposition_model.predict_next_levelset(
    phi_initial=phi,
    duration_s=7000,
    average_rate=rate_M3,
    initial_contour=extract_contour20(phi),
)

pred_3E = etch_model.predict_next_levelset(
    phi_initial=pred_3M,
    duration_s=50,
    average_rate=rate_E3,
    initial_contour=extract_contour20(pred_3M),
)

pred_4M = deposition_model.predict_next_levelset(
    phi_initial=pred_3E,
    duration_s=6000,
    average_rate=rate_M4,
    initial_contour=extract_contour20(pred_3E),
)

pred_4E = etch_model.predict_next_levelset(
    phi_initial=pred_4M,
    duration_s=50,
    average_rate=rate_E4,
    initial_contour=extract_contour20(pred_4M),
)

pred_5M = deposition_model.predict_next_levelset(
    phi_initial=pred_4E,
    duration_s=5000,
    average_rate=rate_M5,
    initial_contour=extract_contour20(pred_4E),
)

pred_5E = etch_model.predict_next_levelset(
    phi_initial=pred_5M,
    duration_s=50,
    average_rate=rate_E5,
    initial_contour=extract_contour20(pred_5M),
)
```

每個輸出 shape 必須是：

```text
(350, 200)
```

每一步保存：

- predicted level-set。
- predicted material mask。
- zero contour。
- 20 點 contour condition。
- validity mask。
- duration。
- average rate。
- nominal displacement。
- optional velocity field。

---

## 16. 初步評估指標

對 `5M`、`5E` 至少計算：

1. Full-field level-set MAE。
2. Narrow-band level-set MAE。
3. Material mask Dice。
4. Material mask IoU。
5. Material area percentage error。
6. Zero-contour symmetric Chamfer distance。
7. 20 點輪廓 y-coordinate MAE。
8. Mean Eikonal error。
9. Predicted mean interface speed error。

20 點輪廓誤差只使用 prediction 與 ground truth 都有效的點：

```math
MAE_20 =
sum(
shared_mask_k *
|y_pred_k - y_gt_k|
)
/
(sum(shared_mask_k) + epsilon)
```

### 16.1 必須輸出的圖形

- `5M` ground truth level-set。
- `5M` predicted level-set。
- `5M` absolute error。
- `5M` GT/pred zero-contour overlay。
- `5E` ground truth level-set。
- `5E` predicted level-set。
- `5E` absolute error。
- `5E` GT/pred zero-contour overlay。
- 每個 rollout step 的 20 點輪廓。
- training loss curves。

---

## 17. Baseline

為判斷 PINN 是否真的優於只使用平均速率，必須實作簡單 baseline：

```math
phi_next_baseline =
phi_initial
-
s_process *
average_rate *
Delta_t
```

此 baseline 假設：

- `phi_initial` 是 SDF。
- 介面各處使用固定 normal speed。
- 不考慮局部幾何修正。

最終報告比較：

```text
PINN vs known-average-rate baseline
```

---

## 18. 預設 YAML

建立：

```text
configs/default.yaml
```

內容至少包含：

```yaml
project:
  name: epi_levelset_pinn_poc
  output_dir: artifacts

data:
  expected_height: 350
  expected_width: 200
  array_order: HW
  allow_transpose: true

  workbooks:
    deposition: data/raw/deposition.xlsx
    etch: data/raw/etch.xlsx

  state_sources:
    init: {workbook: deposition, sheet: init}
    1M: {workbook: deposition, sheet: "1"}
    1E: {workbook: etch, sheet: "1"}
    2M: {workbook: deposition, sheet: "2"}
    2E: {workbook: etch, sheet: "2"}
    5M: {workbook: deposition, sheet: "5"}
    5E: {workbook: etch, sheet: "5"}

spatial:
  pixel_size_x: 1.0
  pixel_size_y: 1.0
  unit: pixel
  y_axis_direction: down

level_set:
  input_kind: signed_distance
  sign_convention: negative_inside
  value_unit: pixel
  rebuild_sdf: false
  phi_clip_distance: 32.0
  narrow_band_distance: 8.0

contour:
  num_points: 20
  x_sampling: uniform
  crossing_policy: closest_to_previous
  first_crossing_policy: topmost
  min_valid_points: 10
  consistency_tolerance_px: 2.0

schedule:
  deposition_seconds:
    1: 9000
    2: 8000
    3: 7000
    4: 6000
    5: 5000

  etch_seconds:
    1: 50
    2: 50
    3: 50
    4: 50
    5: 50

processes:
  deposition:
    sign: 1.0
    rate_unit: pixel/s
    rate_definition: normal_interface_speed
    average_rate_default: null
    average_rate_by_cycle:
      1: null
      2: null
      3: null
      4: null
      5: null
    duration_reference_s: 9000.0
    rate_reference: null

  etch:
    sign: -1.0
    rate_unit: pixel/s
    rate_definition: normal_interface_speed
    average_rate_default: null
    average_rate_by_cycle:
      1: null
      2: null
      3: null
      4: null
      5: null
    duration_reference_s: 50.0
    rate_reference: null

transitions:
  deposition_train:
    - {id: M1, cycle: 1, input_state: init, target_state: 1M}
    - {id: M2, cycle: 2, input_state: 1E, target_state: 2M}

  etch_train:
    - {id: E1, cycle: 1, input_state: 1M, target_state: 1E}
    - {id: E2, cycle: 2, input_state: 2M, target_state: 2E}

normalization:
  x_range: [-1.0, 1.0]
  y_range: [-1.0, 1.0]
  tau_range: [0.0, 1.0]
  normalize_duration: true
  normalize_rate: true
  normalize_nominal_displacement: true

model:
  activation: tanh

  solution_hidden_dim: 128
  solution_depth: 6

  velocity_hidden_dim: 64
  velocity_depth: 4

  contour_embedding_dim: 64

  use_nominal_rate_solution: true
  hard_initial_condition: true
  correction_scale: 0.5
  velocity_residual_fraction: 0.5
  use_curvature_feature: true

loss:
  sdf: 1.0
  dice: 0.5
  pde: 1.0
  eikonal: 0.02
  average_rate: 0.5
  velocity_jacobian: 1.0e-4
  sign: 0.05
  area_monotonic: 0.1
  contour_initial_condition: 0.1

sampling:
  endpoint_interface_fraction: 0.70
  collocation_interface_fraction: 0.60
  collocation_contour_fraction: 0.20
  collocation_global_fraction: 0.20

training:
  seed: 42
  device: auto
  dtype: float64

  adam_steps: 10000
  adam_lr: 1.0e-3

  endpoint_batch_size: 4096
  collocation_batch_size: 4096

  grad_clip_norm: 10.0

  checkpoint_every: 500
  log_every: 100

  use_lbfgs: false

rollout:
  start_state: 2E
  save_intermediate: true

evaluation:
  holdout_deposition_state: 5M
  holdout_etch_state: 5E
  compare_known_rate_baseline: true
  export_prediction_workbook: true
```

所有平均速率的 `null` 必須由使用者填入，程式不得自行捏造數值。

若每個 process 只有一個平均速率，可填入 `average_rate_default`，並保留 `average_rate_by_cycle` 為 `null`。

---

## 19. Repository 結構

```text
epi-levelset-pinn/
├── README.md
├── SPEC.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── deposition.xlsx
│   │   └── etch.xlsx
│   └── processed/
├── src/
│   └── epi_pinn/
│       ├── __init__.py
│       ├── config.py
│       ├── units.py
│       ├── excel_io.py
│       ├── preprocess.py
│       ├── sdf.py
│       ├── contour.py
│       ├── geometry.py
│       ├── sampling.py
│       ├── losses.py
│       ├── baseline.py
│       ├── train.py
│       ├── rollout.py
│       ├── evaluate.py
│       ├── visualize.py
│       ├── pipeline.py
│       └── models/
│           ├── __init__.py
│           ├── contour_encoder.py
│           └── conditional_levelset_pinn.py
└── scripts/
    ├── inspect_xlsx.py
    ├── preprocess_data.py
    ├── train_deposition.py
    ├── train_etch.py
    ├── run_rollout.py
    └── evaluate_holdout.py
```

本階段不要建立：

```text
tests/
pytest.ini
CI workflow
```

---

## 20. Dependencies

`pyproject.toml` 至少包含：

```text
numpy
scipy
pandas
openpyxl
pyyaml
pydantic
torch
scikit-image
matplotlib
```

---

## 21. CLI

### 21.1 檢查 XLSX

```bash
python scripts/inspect_xlsx.py \
  --config configs/default.yaml
```

輸出：

- workbook 路徑。
- sheet names。
- 每個 state 的原始 shape。
- 是否 transpose。
- NaN/Inf 數量。
- level-set min/max。
- zero-crossing coverage。

### 21.2 前處理

```bash
python scripts/preprocess_data.py \
  --config configs/default.yaml \
  --split train
```

訓練完成並凍結 checkpoint 後：

```bash
python scripts/preprocess_data.py \
  --config configs/default.yaml \
  --split holdout
```

### 21.3 訓練

```bash
python scripts/train_deposition.py \
  --config configs/default.yaml
```

```bash
python scripts/train_etch.py \
  --config configs/default.yaml
```

### 21.4 Rollout

```bash
python scripts/run_rollout.py \
  --config configs/default.yaml
```

### 21.5 評估

```bash
python scripts/evaluate_holdout.py \
  --config configs/default.yaml
```

每個 CLI 必須：

- 支援 `--help`。
- 失敗時回傳非零 exit code。
- 顯示清楚的錯誤訊息。
- 不得靜默忽略資料格式錯誤。

---

## 22. 輸出

```text
artifacts/
├── resolved_config.yaml
├── preprocess/
│   ├── data_summary.json
│   └── contour_summary.csv
├── checkpoints/
│   ├── deposition_best.pt
│   └── etch_best.pt
├── logs/
│   ├── deposition_training.csv
│   └── etch_training.csv
├── predictions/
│   ├── 3M.npy
│   ├── 3E.npy
│   ├── 4M.npy
│   ├── 4E.npy
│   ├── 5M.npy
│   ├── 5E.npy
│   └── predictions.xlsx
├── contours/
│   ├── 3M_contour20.csv
│   ├── 3E_contour20.csv
│   ├── 4M_contour20.csv
│   ├── 4E_contour20.csv
│   ├── 5M_contour20.csv
│   └── 5E_contour20.csv
├── metrics/
│   ├── 5M_metrics.json
│   ├── 5E_metrics.json
│   └── summary.csv
├── figures/
└── report.md
```

所有預測 level-set：

```text
shape = (350, 200)
```

`predictions.xlsx` 每個 sheet：

```text
3M
3E
4M
4E
5M
5E
```

每個 sheet 都是純 `(350, 200)` 數值矩陣，不加入 header 或 index。

---

## 23. 執行時資料檢查

雖然本階段不需要 unit tests，程式仍必須在執行時檢查：

1. XLSX workbook 與 sheet 是否存在。
2. 矩陣是否為 numeric。
3. 是否存在 NaN 或 Inf。
4. 前處理後 shape 是否為 `(350, 200)`。
5. `phi_initial` 與 `phi_target` shape 是否相同。
6. 20 點輪廓有效點是否達到門檻。
7. duration 是否大於 0。
8. average rate 是否大於 0。
9. rate unit 與 level-set unit 是否相容。
10. 所有 model output 是否為 finite。
11. rollout 的輸入順序是否正確。
12. `pred_5E` 是否以 `pred_5M` 為輸入。

---

## 24. 初步驗收條件

下列項目全部完成即視為本階段可行性驗證程式完成：

1. 可讀取 XLSX 指定 sheet。
2. 可將資料統一轉為 `(350, 200)`。
3. 可建立或讀取 SDF。
4. 可從每張初始 level-set 抽取 20 點零輪廓。
5. x、y、t 均依本規格標準化。
6. duration、rate 與 nominal displacement 均作條件標準化。
7. 可分別訓練沉積與蝕刻 PINN。
8. 已知平均速率有進入模型與 physics loss。
9. Jacobian 與 PDE residual 使用 PyTorch autograd。
10. 可從 `2E` 遞迴輸出 `3M` 到 `5E`。
11. 每一步重新擷取當前預測的 20 點輪廓。
12. 所有輸出 shape 為 `(350, 200)`。
13. 可比較 `pred_5M` 與真實 `5M`。
14. 可比較 `pred_5E` 與真實 `5E`。
15. 可輸出 baseline、metrics、圖形與報告。
16. 執行過程沒有 NaN 或 Inf。
17. repository 不包含 unit test 或 test code。

---

## 25. 必須揭露的限制

1. 每個 process 目前只有兩個完整 transition。
2. 影像中的 70,000 個 pixel 不等於 70,000 個獨立製程樣本。
3. 沉積模型只看過 9000 s 與 8000 s。
4. 預測 7000、6000、5000 s 屬於時間條件外插。
5. 蝕刻時間全部為 50 s，因此無法從目前資料識別 duration dependence。
6. 已知平均速率主要約束平均介面速度。
7. 局部速度場不一定是唯一可辨識的真實物理速度。
8. 20 點 `y(x)` 表示不適合完整描述：
   - overhang。
   - 封閉孔洞。
   - 同一 x 上多個主要介面。
   - 多材料介面。
9. 因此完整初始 level-set 仍必須作為輸入。
10. 若 cycle 間還有溫度、壓力、氣體流量、RF power 或材料差異，後續應加入 condition vector。
11. `5M` 或 `5E` 一旦用於調參，就不可再稱為 unbiased holdout。

---

## 26. Codex 最終交付內容

Codex 必須交付：

1. 完整 Python repository。
2. `README.md`。
3. 本 `SPEC.md`。
4. `pyproject.toml`。
5. `configs/default.yaml`。
6. XLSX inspection 程式。
7. 資料前處理程式。
8. 20 點輪廓擷取程式。
9. `DepositionPINN`。
10. `EtchPINN`。
11. 沉積訓練 CLI。
12. 蝕刻訓練 CLI。
13. 遞迴 rollout CLI。
14. holdout 評估 CLI。
15. known-average-rate baseline。
16. 預測 XLSX 匯出。
17. metrics、圖形與 `report.md`。

使用者必須在執行前填入：

- workbook path。
- state-to-sheet mapping。
- pixel size 或確認使用 pixel。
- level-set input kind。
- level-set unit。
- 沉積平均速率。
- 蝕刻平均速率。
- average rate unit。
- average rate definition。

程式不得自行捏造平均速率數值。
