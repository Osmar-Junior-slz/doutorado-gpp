"""Configuration editing page."""

from __future__ import annotations

import streamlit as st
import yaml

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.state import AppState, StateKeys


class ConfigPage(BasePage):
    id = "Configurações"
    title = "Configurações"

    def render(self, state: AppState) -> None:
        st.header("Configurações")
        config_data = state.cfg_dict or {}
        source_label = state.cfg_source or "Padrão (configs/default.yaml)"

        st.write(f"Configuração carregada: {source_label}")
        if config_data:
            st.code(yaml.safe_dump(config_data, sort_keys=False), language="yaml")
        else:
            st.warning("Nenhuma configuração carregada no momento.")

        st.subheader("Sobrescrever parâmetros")
        st.write("As alterações abaixo são aplicadas apenas na execução atual (em memória).")

        st.session_state.setdefault(StateKeys.OVERRIDE_SEED, config_data.get("seed", 7))
        st.session_state.setdefault(StateKeys.OVERRIDE_GENERATIONS, config_data.get("generations", 5))
        st.session_state.setdefault(StateKeys.OVERRIDE_POP_SIZE, config_data.get("pop_size", 20))
        st.session_state.setdefault(StateKeys.OVERRIDE_TOPK, config_data.get("topk", 5))
        st.session_state.setdefault(StateKeys.OVERRIDE_FULL_SEARCH, config_data.get("full_search", True))

        seed = st.number_input("seed", value=int(st.session_state[StateKeys.OVERRIDE_SEED]), step=1, key=StateKeys.OVERRIDE_SEED)
        generations = st.number_input(
            "generations",
            value=int(st.session_state[StateKeys.OVERRIDE_GENERATIONS]),
            step=1,
            key=StateKeys.OVERRIDE_GENERATIONS,
        )
        pop_size = st.number_input(
            "pop_size",
            value=int(st.session_state[StateKeys.OVERRIDE_POP_SIZE]),
            step=1,
            key=StateKeys.OVERRIDE_POP_SIZE,
        )
        topk = st.number_input("topk", value=int(st.session_state[StateKeys.OVERRIDE_TOPK]), step=1, key=StateKeys.OVERRIDE_TOPK)
        full_search = st.checkbox(
            "full_search",
            value=bool(st.session_state[StateKeys.OVERRIDE_FULL_SEARCH]),
            key=StateKeys.OVERRIDE_FULL_SEARCH,
        )

        top_pockets = None
        if not full_search:
            st.session_state.setdefault(StateKeys.OVERRIDE_TOP_POCKETS, config_data.get("top_pockets", 3))
            top_pockets = st.number_input(
                "top_pockets",
                value=int(st.session_state[StateKeys.OVERRIDE_TOP_POCKETS]),
                step=1,
                key=StateKeys.OVERRIDE_TOP_POCKETS,
            )

        st.session_state[StateKeys.CONFIG_OVERRIDES] = {
            "seed": int(seed),
            "generations": int(generations),
            "pop_size": int(pop_size),
            "topk": int(topk),
            "full_search": bool(full_search),
            "top_pockets": int(top_pockets) if top_pockets is not None else None,
        }

