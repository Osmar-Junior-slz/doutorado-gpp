import json

from dockingpp.pipeline.run import Config, run_pipeline


def _ler_jsonl(path):
    return [json.loads(linha) for linha in path.read_text(encoding="utf-8").splitlines() if linha.strip()]


def test_reducao_condicionada_registra_logs_estruturados(tmp_path):
    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.usar_reducao_condicionada_ao_peptideo = True
    cfg.debug_log_enabled = True
    cfg.debug_enabled = True

    out_dir = tmp_path / "out_reducao_logs"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    debug_path = out_dir / "debug" / "debug.jsonl"
    trace_path = out_dir / "debug" / "trace.jsonl"

    debug_eventos = _ler_jsonl(debug_path)
    trace_eventos = _ler_jsonl(trace_path)

    tipos = {evento.get("type") for evento in debug_eventos}
    assert "reducao_perfil_peptideo" in tipos
    assert "reducao_bolsao_avaliado" in tipos
    assert "reducao_pre_afinidade" in tipos
    assert "reducao_ranking" in tipos
    assert "reducao_condicionada" in tipos

    event_types = {evento.get("event_type") for evento in trace_eventos}
    assert "reducao_condicionada_ativada" in event_types
    assert "reducao_condicionada_aplicada" in event_types


def test_flag_desligada_nao_registra_eventos_da_reducao(tmp_path):
    cfg = Config()
    cfg.search_space_mode = "reduced"
    cfg.full_search = False
    cfg.top_pockets = 3
    cfg.usar_reducao_condicionada_ao_peptideo = False
    cfg.debug_log_enabled = True

    out_dir = tmp_path / "out_sem_reducao_logs"
    run_pipeline(cfg, "__dummy__", "__dummy__", str(out_dir))

    debug_path = out_dir / "debug" / "debug.jsonl"
    debug_eventos = _ler_jsonl(debug_path)

    tipos = {evento.get("type") for evento in debug_eventos}
    assert "reducao_perfil_peptideo" not in tipos
    assert "reducao_bolsao_avaliado" not in tipos
    assert "reducao_pre_afinidade" not in tipos
    assert "reducao_ranking" not in tipos
    assert "reducao_condicionada" not in tipos
