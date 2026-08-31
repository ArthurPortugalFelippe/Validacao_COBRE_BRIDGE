"""Programa principal do validador de variáveis hidráulicas NEWAVE → COBRE.

Execute este arquivo pelo botão "Run Python File" do VS Code. Quando necessário,
ele instala as dependências e inicia o Streamlit automaticamente.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


DEPENDENCIAS = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "inewave": "inewave",
}


def garantir_dependencias() -> None:
    """Instala somente os pacotes ainda ausentes no ambiente Python."""
    ausentes = [
        pacote
        for modulo, pacote in DEPENDENCIAS.items()
        if importlib.util.find_spec(modulo) is None
    ]
    if ausentes:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *ausentes]
        )


def esta_dentro_do_streamlit() -> bool:
    """Informa se este arquivo já está sendo executado pelo Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


garantir_dependencias()


if __name__ == "__main__" and not esta_dentro_do_streamlit():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
        ],
        check=False,
    )
    raise SystemExit


import streamlit as st

from aba_01_usinas_hidraulicas import mostrar_aba as mostrar_aba_01
from aba_02_cascatas_hidraulicas import mostrar_aba as mostrar_aba_02
from aba_03_parametros_hidraulicos import mostrar_aba as mostrar_aba_03
from aba_04_produtividade_hidraulica import mostrar_aba as mostrar_aba_04


st.set_page_config(
    page_title="Validador hidráulico NEWAVE → COBRE",
    page_icon="💧",
    layout="wide",
)

st.title("Validador hidráulico NEWAVE → COBRE")
st.caption(
    "Ferramenta independente para verificar a conversão das variáveis "
    "relacionadas às usinas hidráulicas."
)


# ABAS DE VALIDAÇÃO HIDRÁULICA
aba_01, aba_02, aba_03, aba_04 = st.tabs(
    [
        "1. Presença e identificação das usinas",
        "2. Formação das cascatas hidráulicas",
        "3. Parâmetros hidráulicos",
        "4. Produtividade e modelo de produção",
    ]
)

with aba_01:
    mostrar_aba_01()

with aba_02:
    mostrar_aba_02()

with aba_03:
    mostrar_aba_03()

with aba_04:
    mostrar_aba_04()
