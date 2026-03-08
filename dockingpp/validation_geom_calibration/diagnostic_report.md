# Diagnóstico de desalinhamento scan vs docking

## Associações globais

- corr_severe_clash_vs_mean_contacts: -0.6608732150132293
- corr_severe_clash_vs_best_score_cheap_final: 0.7913066339213192
- corr_feasible_fraction_vs_best_score_cheap_final: -0.791306633921319
- corr_mean_contacts_vs_best_score_cheap_final: -0.5802611796940753

## Teste de bolsões "folgados" vs "confinados"

- loose_definition: feasible_fraction >= p75 AND severe_clash_fraction <= p25
- tight_definition: feasible_fraction <= p25 AND severe_clash_fraction >= p75
- n_loose: 5
- n_tight: 4
- mean_best_score_loose: 332.1
- mean_best_score_tight: 1098.75

## Análise qualitativa por caso

### case_01_dual_cluster_near
- scan escolheu: auto_grid_1
- docking preferiu: auto_grid_0
- scan_top metrics: {'case_id': 'case_01_dual_cluster_near', 'pocket_id': 'auto_grid_1', 'pocket_scan_score': 5.787615048993214, 'feasible_fraction': 0.359375, 'severe_clash_fraction': 0.640625, 'mean_contacts': 4.671875, 'best_geom_energy': -1.4000000000000001, 'best_score_cheap_final': 413.0}
- docking_top metrics: {'case_id': 'case_01_dual_cluster_near', 'pocket_id': 'auto_grid_0', 'pocket_scan_score': 5.30625, 'feasible_fraction': 0.34375, 'severe_clash_fraction': 0.65625, 'mean_contacts': 4.21875, 'best_geom_energy': -1.4000000000000001, 'best_score_cheap_final': 1117.0}
- deltas (scan_top - docking_top): {'pocket_scan_score': 0.4813650489932133, 'feasible_fraction': 0.015625, 'severe_clash_fraction': -0.015625, 'mean_contacts': 0.453125, 'best_geom_energy': 0.0, 'best_score_cheap_final': -704.0}
- best_score_expensive_final disponível: False

### case_02_dual_cluster_mid
- scan escolheu: auto_grid_2
- docking preferiu: auto_grid_0
- scan_top metrics: {'case_id': 'case_02_dual_cluster_mid', 'pocket_id': 'auto_grid_2', 'pocket_scan_score': 6.884375, 'feasible_fraction': 0.25, 'severe_clash_fraction': 0.75, 'mean_contacts': 5.984375, 'best_geom_energy': -1.4000000000000001, 'best_score_cheap_final': 948.5}
- docking_top metrics: {'case_id': 'case_02_dual_cluster_mid', 'pocket_id': 'auto_grid_0', 'pocket_scan_score': 4.728125, 'feasible_fraction': 0.171875, 'severe_clash_fraction': 0.828125, 'mean_contacts': 3.984375, 'best_geom_energy': -1.4000000000000001, 'best_score_cheap_final': 1222.0}
- deltas (scan_top - docking_top): {'pocket_scan_score': 2.15625, 'feasible_fraction': 0.078125, 'severe_clash_fraction': -0.078125, 'mean_contacts': 2.0, 'best_geom_energy': 0.0, 'best_score_cheap_final': -273.5}
- best_score_expensive_final disponível: False

### case_03_tri_cluster
- scan escolheu: auto_grid_3
- docking preferiu: auto_grid_0
- scan_top metrics: {'case_id': 'case_03_tri_cluster', 'pocket_id': 'auto_grid_3', 'pocket_scan_score': 8.077269893537641, 'feasible_fraction': 0.421875, 'severe_clash_fraction': 0.578125, 'mean_contacts': 6.84375, 'best_geom_energy': -1.4000000000000001, 'best_score_cheap_final': 103.0}
- docking_top metrics: {'case_id': 'case_03_tri_cluster', 'pocket_id': 'auto_grid_0', 'pocket_scan_score': 5.6875, 'feasible_fraction': 0.21875, 'severe_clash_fraction': 0.78125, 'mean_contacts': 4.75, 'best_geom_energy': -1.5, 'best_score_cheap_final': 924.5}
- deltas (scan_top - docking_top): {'pocket_scan_score': 2.3897698935376415, 'feasible_fraction': 0.203125, 'severe_clash_fraction': -0.203125, 'mean_contacts': 2.09375, 'best_geom_energy': 0.09999999999999987, 'best_score_cheap_final': -821.5}
- best_score_expensive_final disponível: False

### case_04_spread_chain
- scan escolheu: auto_grid_2
- docking preferiu: auto_grid_0
- scan_top metrics: {'case_id': 'case_04_spread_chain', 'pocket_id': 'auto_grid_2', 'pocket_scan_score': 6.778002255306963, 'feasible_fraction': 0.484375, 'severe_clash_fraction': 0.515625, 'mean_contacts': 5.515625, 'best_geom_energy': -1.3, 'best_score_cheap_final': 178.0}
- docking_top metrics: {'case_id': 'case_04_spread_chain', 'pocket_id': 'auto_grid_0', 'pocket_scan_score': 4.878125, 'feasible_fraction': 0.234375, 'severe_clash_fraction': 0.765625, 'mean_contacts': 4.109375, 'best_geom_energy': -1.3, 'best_score_cheap_final': 672.0}
- deltas (scan_top - docking_top): {'pocket_scan_score': 1.8998772553069632, 'feasible_fraction': 0.25, 'severe_clash_fraction': -0.25, 'mean_contacts': 1.40625, 'best_geom_energy': 0.0, 'best_score_cheap_final': -494.0}
- best_score_expensive_final disponível: False

### case_05_compact_dense
- scan escolheu: auto_grid_2
- docking preferiu: auto_grid_0
- scan_top metrics: {'case_id': 'case_05_compact_dense', 'pocket_id': 'auto_grid_2', 'pocket_scan_score': 6.153125, 'feasible_fraction': 0.46875, 'severe_clash_fraction': 0.53125, 'mean_contacts': 5.015625, 'best_geom_energy': -1.2000000000000002, 'best_score_cheap_final': 107.5}
- docking_top metrics: {'case_id': 'case_05_compact_dense', 'pocket_id': 'auto_grid_0', 'pocket_scan_score': 2.325, 'feasible_fraction': 0.109375, 'severe_clash_fraction': 0.890625, 'mean_contacts': 1.90625, 'best_geom_energy': -1.2000000000000002, 'best_score_cheap_final': 1484.0}
- deltas (scan_top - docking_top): {'pocket_scan_score': 3.828125, 'feasible_fraction': 0.359375, 'severe_clash_fraction': -0.359375, 'mean_contacts': 3.109375, 'best_geom_energy': 0.0, 'best_score_cheap_final': -1376.5}
- best_score_expensive_final disponível: False

## Hipótese principal para sinal invertido

- O score geométrico está favorecendo bolsões com maior viabilidade local rígida (mais contatos e menor colisão severa no scan), mas estes bolsões parecem levar a ótimos piores no espaço de busca barato final.
- Em especial, casos com pocket_scan alto apresentaram best_score_cheap final menor que bolsões mais "apertados" (maior severe_clash), sugerindo mismatch entre objetivo local rígido e objetivo global do docking barato.
- A energia geométrica apresenta baixa variabilidade útil (muito próxima entre bolsões), então termos de viabilidade/contatos dominam o ranking.

## Recomendação conceitual (sem refatorar agora)

- Tratar `pocket_scan_score` como filtro de inviabilidade e não como ranker forte até recalibrar com alvo final de docking.
- Na próxima rodada, usar ranking híbrido: prior pocket rank base + scan geom como tie-breaker/penalizador (não termo dominante).
- Coletar features pós-docking curto (micro-budget por bolsão) para calibrar score com sinal consistente ao objetivo final.