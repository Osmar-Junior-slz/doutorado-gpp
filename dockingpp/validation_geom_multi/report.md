# Validação multi-casos: legacy vs geom_kdtree

## Tabela comparativa

| case_id | legacy_selected_count | geom_selected_count | legacy_fallback | geom_fallback | legacy_best_score | geom_best_score | legacy_runtime | geom_runtime | legacy_total_eval | geom_total_eval |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| case_01_dual_cluster_near | 0 | 2 | True | False | 1129.0 | 1137.0 | 0.191335 | 0.122899 | 64 | 64 |
| case_02_dual_cluster_mid | 0 | 2 | True | False | 1332.5 | 1222.0 | 0.137714 | 0.090930 | 64 | 64 |
| case_03_tri_cluster | 0 | 2 | True | False | 2578.5 | 859.0 | 0.234104 | 0.142634 | 64 | 64 |
| case_04_spread_chain | 0 | 2 | True | False | 1412.5 | 639.0 | 0.165721 | 0.106495 | 64 | 64 |
| case_05_compact_dense | 0 | 2 | True | False | 1644.5 | 1484.0 | 0.118310 | 0.082341 | 64 | 64 |

## Métricas detalhadas por caso

### case_01_dual_cluster_near
- receptor: `validation_geom_multi/inputs/case_01_dual_cluster_near_receptor.pdb`
- peptide: `validation_geom_multi/inputs/case_01_dual_cluster_near_peptide.pdb`
- **legacy**: selected_pockets=[], rejected_pockets=[{'pocket_id': 'auto_grid_2', 'reason': 'feasible_fraction<=0.0'}, {'pocket_id': 'auto_grid_1', 'reason': 'feasible_fraction<=0.0'}], fallback_to_full=True, n_pockets_used=0, best_pocket_id=global, best_over_pockets_cheap=1129.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.19133503599732649, full_total_n_eval=64, full_total_runtime_sec=0.04706337399693439, scan_eval_budget=192, docking_eval_budget=64, total_eval_budget_pipeline=256
- **geom_kdtree**: selected_pockets=['auto_grid_1', 'auto_grid_0'], rejected_pockets=[], fallback_to_full=False, n_pockets_used=2, best_pocket_id=auto_grid_0, best_over_pockets_cheap=1137.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.12289925899676746, full_total_n_eval=64, full_total_runtime_sec=0.04706337399693439, scan_eval_budget=192, docking_eval_budget=64, total_eval_budget_pipeline=256

### case_02_dual_cluster_mid
- receptor: `validation_geom_multi/inputs/case_02_dual_cluster_mid_receptor.pdb`
- peptide: `validation_geom_multi/inputs/case_02_dual_cluster_mid_peptide.pdb`
- **legacy**: selected_pockets=[], rejected_pockets=[{'pocket_id': 'auto_grid_0', 'reason': 'feasible_fraction<=0.0'}, {'pocket_id': 'auto_grid_2', 'reason': 'feasible_fraction<=0.0'}], fallback_to_full=True, n_pockets_used=0, best_pocket_id=global, best_over_pockets_cheap=1332.5, total_n_eval_reduced=64, total_runtime_sec_reduced=0.13771417600219138, full_total_n_eval=64, full_total_runtime_sec=0.05315437300305348, scan_eval_budget=128, docking_eval_budget=64, total_eval_budget_pipeline=192
- **geom_kdtree**: selected_pockets=['auto_grid_0', 'auto_grid_2'], rejected_pockets=[], fallback_to_full=False, n_pockets_used=2, best_pocket_id=auto_grid_0, best_over_pockets_cheap=1222.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.09092963099465123, full_total_n_eval=64, full_total_runtime_sec=0.05315437300305348, scan_eval_budget=128, docking_eval_budget=64, total_eval_budget_pipeline=192

### case_03_tri_cluster
- receptor: `validation_geom_multi/inputs/case_03_tri_cluster_receptor.pdb`
- peptide: `validation_geom_multi/inputs/case_03_tri_cluster_peptide.pdb`
- **legacy**: selected_pockets=[], rejected_pockets=[{'pocket_id': 'auto_grid_3', 'reason': 'feasible_fraction<=0.0'}, {'pocket_id': 'auto_grid_0', 'reason': 'feasible_fraction<=0.0'}], fallback_to_full=True, n_pockets_used=0, best_pocket_id=global, best_over_pockets_cheap=2578.5, total_n_eval_reduced=64, total_runtime_sec_reduced=0.2341042030020617, full_total_n_eval=64, full_total_runtime_sec=0.05046505199788953, scan_eval_budget=256, docking_eval_budget=64, total_eval_budget_pipeline=320
- **geom_kdtree**: selected_pockets=['auto_grid_3', 'auto_grid_1'], rejected_pockets=[], fallback_to_full=False, n_pockets_used=2, best_pocket_id=auto_grid_1, best_over_pockets_cheap=859.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.14263429100174108, full_total_n_eval=64, full_total_runtime_sec=0.05046505199788953, scan_eval_budget=256, docking_eval_budget=64, total_eval_budget_pipeline=320

### case_04_spread_chain
- receptor: `validation_geom_multi/inputs/case_04_spread_chain_receptor.pdb`
- peptide: `validation_geom_multi/inputs/case_04_spread_chain_peptide.pdb`
- **legacy**: selected_pockets=[], rejected_pockets=[{'pocket_id': 'auto_grid_2', 'reason': 'feasible_fraction<=0.0'}, {'pocket_id': 'auto_grid_0', 'reason': 'feasible_fraction<=0.0'}], fallback_to_full=True, n_pockets_used=0, best_pocket_id=global, best_over_pockets_cheap=1412.5, total_n_eval_reduced=64, total_runtime_sec_reduced=0.16572092799833626, full_total_n_eval=64, full_total_runtime_sec=0.04251246199783054, scan_eval_budget=192, docking_eval_budget=64, total_eval_budget_pipeline=256
- **geom_kdtree**: selected_pockets=['auto_grid_2', 'auto_grid_1'], rejected_pockets=[], fallback_to_full=False, n_pockets_used=2, best_pocket_id=auto_grid_1, best_over_pockets_cheap=639.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.10649462700530421, full_total_n_eval=64, full_total_runtime_sec=0.04251246199783054, scan_eval_budget=192, docking_eval_budget=64, total_eval_budget_pipeline=256

### case_05_compact_dense
- receptor: `validation_geom_multi/inputs/case_05_compact_dense_receptor.pdb`
- peptide: `validation_geom_multi/inputs/case_05_compact_dense_peptide.pdb`
- **legacy**: selected_pockets=[], rejected_pockets=[{'pocket_id': 'auto_grid_0', 'reason': 'feasible_fraction<=0.0'}, {'pocket_id': 'auto_grid_2', 'reason': 'feasible_fraction<=0.0'}], fallback_to_full=True, n_pockets_used=0, best_pocket_id=global, best_over_pockets_cheap=1644.5, total_n_eval_reduced=64, total_runtime_sec_reduced=0.11830973999894923, full_total_n_eval=64, full_total_runtime_sec=0.043159657001524465, scan_eval_budget=128, docking_eval_budget=64, total_eval_budget_pipeline=192
- **geom_kdtree**: selected_pockets=['auto_grid_0', 'auto_grid_2'], rejected_pockets=[], fallback_to_full=False, n_pockets_used=2, best_pocket_id=auto_grid_0, best_over_pockets_cheap=1484.0, total_n_eval_reduced=64, total_runtime_sec_reduced=0.08234095599254942, full_total_n_eval=64, full_total_runtime_sec=0.043159657001524465, scan_eval_budget=128, docking_eval_budget=64, total_eval_budget_pipeline=192

## Resumo final

- n_cases: 5
- fallback_reduced_in_cases: 5
- geom_selected_useful_pockets_in_cases: 5
- operational_improvement_cases: 5
- acceptable_cost_cases: 5
- note_eval_64: O valor total_n_eval=64 corresponde ao budget do docking/search (generations*pop_size), não ao scan; scan_eval_budget é separado como samples_per_pocket*n_pockets_scanned.