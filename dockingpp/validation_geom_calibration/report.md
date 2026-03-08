# Calibração do pocket_scan_score (geom_kdtree)

## 1) Ranking por caso: scan vs docking final

### case_01_dual_cluster_near
- ranking por pocket_scan_score: ['auto_grid_1', 'auto_grid_0', 'auto_grid_2']
- ranking por best_score_cheap final: ['auto_grid_0', 'auto_grid_1', 'auto_grid_2']

### case_02_dual_cluster_mid
- ranking por pocket_scan_score: ['auto_grid_2', 'auto_grid_0']
- ranking por best_score_cheap final: ['auto_grid_0', 'auto_grid_2']

### case_03_tri_cluster
- ranking por pocket_scan_score: ['auto_grid_3', 'auto_grid_1', 'auto_grid_0', 'auto_grid_2']
- ranking por best_score_cheap final: ['auto_grid_0', 'auto_grid_1', 'auto_grid_2', 'auto_grid_3']

### case_04_spread_chain
- ranking por pocket_scan_score: ['auto_grid_2', 'auto_grid_1', 'auto_grid_0']
- ranking por best_score_cheap final: ['auto_grid_0', 'auto_grid_1', 'auto_grid_2']

### case_05_compact_dense
- ranking por pocket_scan_score: ['auto_grid_2', 'auto_grid_0']
- ranking por best_score_cheap final: ['auto_grid_0', 'auto_grid_2']

## 2) Tabela por caso e por bolsão

| case_id | pocket_id | pocket_scan_score | feasible_fraction | severe_clash_fraction | mean_contacts | best_geom_energy | best_score_cheap_final |
|---|---|---:|---:|---:|---:|---:|---:|
| case_01_dual_cluster_near | auto_grid_0 | 5.306250 | 0.343750 | 0.656250 | 4.218750 | -1.400000 | 1117.000000 |
| case_01_dual_cluster_near | auto_grid_1 | 5.787615 | 0.359375 | 0.640625 | 4.671875 | -1.400000 | 413.000000 |
| case_01_dual_cluster_near | auto_grid_2 | 4.431250 | 0.281250 | 0.718750 | 3.468750 | -1.400000 | 281.000000 |
| case_02_dual_cluster_mid | auto_grid_0 | 4.728125 | 0.171875 | 0.828125 | 3.984375 | -1.400000 | 1222.000000 |
| case_02_dual_cluster_mid | auto_grid_2 | 6.884375 | 0.250000 | 0.750000 | 5.984375 | -1.400000 | 948.500000 |
| case_03_tri_cluster | auto_grid_1 | 5.853125 | 0.359375 | 0.640625 | 4.734375 | -1.400000 | 859.000000 |
| case_03_tri_cluster | auto_grid_3 | 8.077270 | 0.421875 | 0.578125 | 6.843750 | -1.400000 | 103.000000 |
| case_03_tri_cluster | auto_grid_2 | 5.140625 | 0.203125 | 0.796875 | 4.234375 | -1.500000 | 764.500000 |
| case_03_tri_cluster | auto_grid_0 | 5.687500 | 0.218750 | 0.781250 | 4.750000 | -1.500000 | 924.500000 |
| case_04_spread_chain | auto_grid_1 | 5.096875 | 0.265625 | 0.734375 | 4.265625 | -1.300000 | 639.000000 |
| case_04_spread_chain | auto_grid_2 | 6.778002 | 0.484375 | 0.515625 | 5.515625 | -1.300000 | 178.000000 |
| case_04_spread_chain | auto_grid_0 | 4.878125 | 0.234375 | 0.765625 | 4.109375 | -1.300000 | 672.000000 |
| case_05_compact_dense | auto_grid_0 | 2.325000 | 0.109375 | 0.890625 | 1.906250 | -1.200000 | 1484.000000 |
| case_05_compact_dense | auto_grid_2 | 6.153125 | 0.468750 | 0.531250 | 5.015625 | -1.200000 | 107.500000 |

## 3) Correlações

### Global (todos os bolsões)
- corr_pocket_scan_score_vs_best_final: -0.6266005266877358
- corr_feasible_fraction_vs_best_final: -0.791306633921319
- corr_severe_clash_fraction_vs_best_final: 0.7913066339213192
- corr_mean_contacts_vs_best_final: -0.5802611796940753
- corr_best_geom_energy_vs_best_final: -0.09280996195924557

### Por caso
- case_01_dual_cluster_near: {'corr_pocket_scan_score_vs_best_final': 0.30829240317533296, 'corr_feasible_fraction_vs_best_final': 0.46254016564831807, 'corr_severe_clash_fraction_vs_best_final': -0.46254016564831807, 'corr_mean_contacts_vs_best_final': 0.28489394576125643, 'corr_best_geom_energy_vs_best_final': None}
- case_02_dual_cluster_mid: {'corr_pocket_scan_score_vs_best_final': -1.0, 'corr_feasible_fraction_vs_best_final': -1.0, 'corr_severe_clash_fraction_vs_best_final': 1.0, 'corr_mean_contacts_vs_best_final': -1.0, 'corr_best_geom_energy_vs_best_final': None}
- case_03_tri_cluster: {'corr_pocket_scan_score_vs_best_final': -0.9246384724075682, 'corr_feasible_fraction_vs_best_final': -0.7208173722537202, 'corr_severe_clash_fraction_vs_best_final': 0.7208173722537202, 'corr_mean_contacts_vs_best_final': -0.9306152540740005, 'corr_best_geom_energy_vs_best_final': -0.5538814918351217}
- case_04_spread_chain: {'corr_pocket_scan_score_vs_best_final': -0.9989590143548782, 'corr_feasible_fraction_vs_best_final': -0.9984779004607657, 'corr_severe_clash_fraction_vs_best_final': 0.9984779004607657, 'corr_mean_contacts_vs_best_final': -0.9991283733559879, 'corr_best_geom_energy_vs_best_final': None}
- case_05_compact_dense: {'corr_pocket_scan_score_vs_best_final': -1.0, 'corr_feasible_fraction_vs_best_final': -1.0, 'corr_severe_clash_fraction_vs_best_final': 1.0, 'corr_mean_contacts_vs_best_final': -1.0, 'corr_best_geom_energy_vs_best_final': None}

## 4) Termos que ajudam/atrapalham (observado)

- No conjunto agregado, `pocket_scan_score` correlaciona negativamente com score final de docking (esperado seria positivo).
- `feasible_fraction` e `mean_contacts` também ficaram com correlação negativa com score final, indicando que pesos positivos atuais podem estar empurrando ranking para bolsões piores no desfecho final.
- `severe_clash_fraction` ficou positivamente correlacionado com score final (contrário ao esperado físico), sugerindo que a penalização atual pode estar excessiva para este regime de dados.
- `best_geom_energy` apresentou correlação fraca/instável no agregado.

## 5) Recomendação de nova calibração de pesos (alpha/beta/gamma/delta)

- proposta (a partir de ajuste linear padronizado): alpha=0.0, beta=0.0, gamma=0.0, delta=0.1
- interpretação prática: reduzir fortemente o efeito de feasible/contacts/clash no score composto; manter energia como termo central e reintroduzir penalidades gradualmente após nova rodada de calibração com dados reais adicionais.
- sugestão operacional imediata: score de triagem conservador `pocket_scan_score_cal = -best_geom_energy + 0.1*feasible_fraction + 0.1*mean_contacts - 0.05*severe_clash_fraction - 0.1*mean_exposure_penalty`, seguido de nova validação.

## 6) Aderência de ranking

- top1_rank_match_cases: 0
- n_cases: 5
- top1_rank_match_ratio: 0.0