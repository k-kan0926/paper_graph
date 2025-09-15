# paper_graph
***
## 0. 前提
$p_\Sigma$：和圧

$p_\Delta$：差圧

## 1. rosbagからcsv抽出

`/one_arm_modeling/analye_bag.py`

```
python analyze_bag.py   --bag 2025-09-12-19-27-02.bag   --sum-topic /exp_diff_grid_mpa/p_sum_MPa   --diff-topic /exp_diff_grid_mpa/p_diff_MPa   --joint-topic /kinikun1/joint_states   --joint-name arm1_joint   --cmd-topic /mpa_cmd   --cmd-scale 0.9/4096 --out-prefix out/diff_run
```

## 2. csvから3D可視化

`/one_arm_modeling/smooth_visualize_3d_theta_from_csv.py`

```
python smooth_visualize_3d_theta_form_csv.py   --csv out/diff_run1_data.csv   --out-prefix out/vis3d   --downsample 12000   --nbins 80   --smooth cubic
``` 

## 3. 静的面の同定　
$$
\Large \theta \approx f(p_\Sigma, p_\Delta)
$$

`/one_arm_modeling/fit_theta_static_surface.py`

```
python fit_theta_static_surface.py --csv out/diff_run1_data.csv --deg 3 --lambda 1e-3 --out-prefix out/theta_static/theta_static
```

## 4. 動的モデルの同定
$$
{\Large \displaystyle
\dot{\theta} = \frac{f(p_\Sigma, p_\Delta) - \theta}{\tau(p_\Sigma)},\quad
\tau(p_\Sigma)=\tau_0+\tau_1 p_\Sigma,\ \tau>0
}
$$

`/one_arm_modeling/fit_theta_dynamic_h1.py`

```
python fit_theta_dynamic_h1.py --csv out/diff_run1_data.csv --deg 3 --use-scipy --out-prefix out/theta_dyn/theta_dyn
```

## 5. 予測性能の可視化

`/one_arm_modeling/eval_theta_dyn_fit.py`

```
python eval_theta_dyn_fit.py --csv out/diff_run1_data.csv --predictor out/theta_dyn/theta_dyn_predictor.py --out-prefix out/theta_dyn_eval/theta_dyn_eval
```

