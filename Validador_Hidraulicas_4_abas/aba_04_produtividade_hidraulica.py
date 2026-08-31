"""Aba 4 — produtividade energética e modelo de produção hidráulica.

Reproduz as regras do cobre-bridge 0.15 diretamente a partir dos arquivos de
entrada do NEWAVE e compara os resultados com:

* system/hydro_energy_productivity.parquet;
* system/hydro_production_models.json;
* system/hydros.json;
* stages.json.

O módulo é independente das três abas anteriores e mantém as funções de
cálculo separadas da interface Streamlit para facilitar testes controlados.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MESES = [
    "JAN",
    "FEV",
    "MAR",
    "ABR",
    "MAI",
    "JUN",
    "JUL",
    "AGO",
    "SET",
    "OUT",
    "NOV",
    "DEZ",
]

STATUS_COINCIDENTE = "Coincidente"
STATUS_DIVERGENTE = "Divergente"
STATUS_NAO_VALIDADO = "Não validado"


class ErroAba04(Exception):
    """Erro de entrada ou de estrutura que impede a validação."""


@dataclass(frozen=True)
class ContextoNewave:
    """Dados NEWAVE já lidos e preparados para os cálculos."""

    dger: Any
    confhd: pd.DataFrame
    cadastro_original: pd.DataFrame
    cadastro: pd.DataFrame
    exph: pd.DataFrame | None
    modif: Any | None
    volref: dict[int, dict[int, float]]
    alteracoes_permanentes: pd.DataFrame
    alteracoes_temporais: dict[int, list[dict[str, Any]]]
    arquivos_presentes: frozenset[str]
    tratamento_fpha: bytes | None


@dataclass(frozen=True)
class ResultadoAba04:
    """Tabelas e indicadores produzidos pela Aba 4."""

    depara: pd.DataFrame
    produtividade: pd.DataFrame
    memoria_calculo: pd.DataFrame
    modelos: pd.DataFrame
    parametros_modelos: pd.DataFrame
    fitting_windows: pd.DataFrame
    volumes_referencia: pd.DataFrame
    inconsistencias: pd.DataFrame
    arquivos_utilizados: pd.DataFrame
    total_usinas_newave: int
    total_usinas_cobre: int
    total_linhas_parquet: int
    total_parametros_json: int
    total_coincidentes: int
    total_divergencias: int
    total_nao_validados: int
    aprovada: bool


def normalizar_nome(nome: Any) -> str:
    """Normaliza um nome somente para conferência visual do de-para."""
    texto = str(nome or "").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def gerar_csv(df: pd.DataFrame) -> bytes:
    """Gera CSV compatível com Excel em português."""
    return df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode(
        "utf-8-sig"
    )


def _nome_base(nome: str) -> str:
    return Path(nome.replace("\\", "/")).name.upper()


def expandir_arquivos(arquivos: dict[str, bytes]) -> dict[str, bytes]:
    """Expande ZIPs e devolve um mapa por nome-base, sem extrair no deck original."""
    resultado: dict[str, bytes] = {}
    for nome, conteudo in arquivos.items():
        if nome.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
                    for item in pacote.infolist():
                        if item.is_dir():
                            continue
                        resultado[_nome_base(item.filename)] = pacote.read(item)
            except zipfile.BadZipFile as exc:
                raise ErroAba04(f"O arquivo {nome} não é um ZIP válido.") from exc
        else:
            resultado[_nome_base(nome)] = conteudo
    return resultado


def arquivos_de_diretorio(diretorio: str | Path) -> dict[str, bytes]:
    """Carrega arquivos de um diretório para testes fora da interface."""
    raiz = Path(diretorio)
    if not raiz.exists():
        raise ErroAba04(f"Diretório não encontrado: {raiz}")
    return {
        caminho.name: caminho.read_bytes()
        for caminho in raiz.rglob("*")
        if caminho.is_file()
    }


def _localizar(
    arquivos: dict[str, bytes], *nomes: str, obrigatorio: bool = False
) -> tuple[str | None, bytes | None]:
    for nome in nomes:
        chave = _nome_base(nome)
        if chave in arquivos:
            return chave, arquivos[chave]
    if obrigatorio:
        esperados = ", ".join(nomes)
        raise ErroAba04(f"Arquivo obrigatório não fornecido: {esperados}.")
    return None, None


def _tratamento_fpha(arquivos: dict[str, bytes]) -> tuple[str | None, bytes | None]:
    for nome, conteudo in arquivos.items():
        token = nome.upper().replace("_", "-")
        if "FPHA" in token and ("TRAT" in token or "TREAT" in token):
            return nome, conteudo
    return None, None


def _gravar_leitor(raiz: Path, nome: str, conteudo: bytes) -> Path:
    caminho = raiz / nome
    caminho.write_bytes(conteudo)
    return caminho


def _ler_newave(arquivos_brutos: dict[str, bytes]) -> ContextoNewave:
    """Lê os arquivos NEWAVE usando os leitores oficiais do pacote inewave."""
    try:
        from inewave.newave import Confhd, Dger, Exph, Hidr, Modif, VolrefSaz
    except ImportError as exc:
        raise ErroAba04(
            "A dependência 'inewave' não está instalada no ambiente Python."
        ) from exc

    arquivos = expandir_arquivos(arquivos_brutos)
    nome_dger, bytes_dger = _localizar(arquivos, "DGER.DAT", obrigatorio=True)
    nome_confhd, bytes_confhd = _localizar(
        arquivos, "CONFHD.DAT", obrigatorio=True
    )
    nome_hidr, bytes_hidr = _localizar(arquivos, "HIDR.DAT", obrigatorio=True)
    nome_exph, bytes_exph = _localizar(arquivos, "EXPH.DAT")
    nome_modif, bytes_modif = _localizar(arquivos, "MODIF.DAT")
    nome_volref, bytes_volref = _localizar(
        arquivos, "VOLREF_SAZ.DAT", "VOLREFSAZ.DAT"
    )
    nome_tratamento, bytes_tratamento = _tratamento_fpha(arquivos)

    presentes = {
        nome
        for nome in (
            nome_dger,
            nome_confhd,
            nome_hidr,
            nome_exph,
            nome_modif,
            nome_volref,
            nome_tratamento,
        )
        if nome is not None
    }

    with tempfile.TemporaryDirectory(prefix="validador_aba04_newave_") as tmp:
        raiz = Path(tmp)
        try:
            dger = Dger.read(str(_gravar_leitor(raiz, "DGER.DAT", bytes_dger)))
            confhd_obj = Confhd.read(
                str(_gravar_leitor(raiz, "CONFHD.DAT", bytes_confhd))
            )
            hidr_obj = Hidr.read(
                str(_gravar_leitor(raiz, "HIDR.DAT", bytes_hidr))
            )
            exph_obj = (
                Exph.read(str(_gravar_leitor(raiz, "EXPH.DAT", bytes_exph)))
                if bytes_exph is not None
                else None
            )
            modif_obj = (
                Modif.read(str(_gravar_leitor(raiz, "MODIF.DAT", bytes_modif)))
                if bytes_modif is not None
                else None
            )
            volref_obj = (
                VolrefSaz.read(
                    str(_gravar_leitor(raiz, "VOLREF_SAZ.DAT", bytes_volref))
                )
                if bytes_volref is not None
                else None
            )
        except Exception as exc:
            raise ErroAba04(f"Falha ao ler os arquivos NEWAVE: {exc}") from exc

    cadastro_original = hidr_obj.cadastro.copy()
    cadastro, permanentes = _aplicar_alteracoes_permanentes(
        cadastro_original, modif_obj
    )
    confhd = confhd_obj.usinas.copy()
    exph = exph_obj.expansoes.copy() if exph_obj is not None else None
    volref = _ler_volref(volref_obj)
    temporais = _extrair_alteracoes_temporais(modif_obj, confhd)

    return ContextoNewave(
        dger=dger,
        confhd=confhd,
        cadastro_original=cadastro_original,
        cadastro=cadastro,
        exph=exph,
        modif=modif_obj,
        volref=volref,
        alteracoes_permanentes=permanentes,
        alteracoes_temporais=temporais,
        arquivos_presentes=frozenset(presentes),
        tratamento_fpha=bytes_tratamento,
    )


def _aplicar_alteracoes_permanentes(
    cadastro: pd.DataFrame, modif: Any | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica VOLMIN/VOLMAX do MODIF, exatamente os que afetam esta aba."""
    resultado = cadastro.copy()
    registros: list[dict[str, Any]] = []
    if modif is None:
        return resultado, pd.DataFrame(registros)

    for usina in modif.usina() or []:
        codigo = int(usina.codigo)
        if codigo not in resultado.index:
            continue
        for registro in modif.modificacoes_usina(codigo):
            tipo = type(registro).__name__
            if tipo not in {"VOLMIN", "VOLMAX"}:
                continue
            coluna = "volume_minimo" if tipo == "VOLMIN" else "volume_maximo"
            anterior = float(resultado.loc[codigo, coluna])
            novo = float(registro.volume)
            resultado.loc[codigo, coluna] = novo
            registros.append(
                {
                    "codigo_newave": codigo,
                    "tipo": tipo,
                    "valor_hidr": anterior,
                    "valor_aplicado": novo,
                }
            )
    return resultado, pd.DataFrame(registros)


def _extrair_alteracoes_temporais(
    modif: Any | None, confhd: pd.DataFrame
) -> dict[int, list[dict[str, Any]]]:
    """Extrai CFUGA e CMONT datados do MODIF.DAT."""
    if modif is None:
        return {}
    codigos = {int(c) for c in confhd["codigo_usina"]}
    resultado: dict[int, list[dict[str, Any]]] = {}
    for usina in modif.usina() or []:
        codigo = int(usina.codigo)
        if codigo not in codigos:
            continue
        itens: list[dict[str, Any]] = []
        for registro in modif.modificacoes_usina(codigo):
            tipo = type(registro).__name__
            if tipo not in {"CFUGA", "CMONT"}:
                continue
            data_inicio = registro.data_inicio
            itens.append(
                {
                    "type": tipo,
                    "month": int(data_inicio.month),
                    "year": int(data_inicio.year),
                    "value": float(registro.nivel),
                }
            )
        if itens:
            resultado[codigo] = itens
    return resultado


def _ler_volref(volref_obj: Any | None) -> dict[int, dict[int, float]]:
    """Lê volumes úteis mensais e descarta a linha sentinela toda zerada."""
    if volref_obj is None or volref_obj.volumes is None:
        return {}
    df = volref_obj.volumes
    if df.empty:
        return {}
    por_usina: dict[int, dict[int, float]] = {}
    for _, linha in df.iterrows():
        codigo = int(linha["codigo_usina"])
        mes = int(linha["mes"])
        por_usina.setdefault(codigo, {})[mes] = float(linha["valor"])
    return {
        codigo: meses
        for codigo, meses in por_usina.items()
        if any(valor > 0.0 for valor in meses.values())
    }


def _ficticias(confhd: pd.DataFrame, cadastro: pd.DataFrame) -> set[int]:
    existentes = confhd[confhd["usina_existente"] == "EX"]
    rho: dict[int, float] = {}
    posto: dict[int, int] = {}
    for _, linha in existentes.iterrows():
        codigo = int(linha["codigo_usina"])
        if codigo not in cadastro.index:
            continue
        rho[codigo] = float(cadastro.loc[codigo, "produtibilidade_especifica"])
        posto[codigo] = int(linha["posto"])
    postos_geradores = {posto[c] for c, valor in rho.items() if valor > 0.0}
    return {
        codigo
        for codigo, valor in rho.items()
        if valor == 0.0 and posto[codigo] in postos_geradores
    }


def _usinas_enchimento(confhd: pd.DataFrame, exph: pd.DataFrame | None) -> set[int]:
    if exph is None or exph.empty:
        return set()
    codigos_ne = {
        int(codigo)
        for codigo in confhd.loc[
            confhd["usina_existente"] == "NE", "codigo_usina"
        ]
    }
    codigos_com_data = {
        int(codigo)
        for codigo in exph.loc[
            exph["data_inicio_enchimento"].notna(), "codigo_usina"
        ]
    }
    return codigos_ne & codigos_com_data


def _codigos_ativos(ctx: ContextoNewave) -> tuple[list[int], set[int], set[int]]:
    fict = _ficticias(ctx.confhd, ctx.cadastro)
    enchimento = _usinas_enchimento(ctx.confhd, ctx.exph)
    codigos = []
    for _, linha in ctx.confhd.iterrows():
        codigo = int(linha["codigo_usina"])
        existente_real = linha["usina_existente"] == "EX" and codigo not in fict
        if existente_real or codigo in enchimento:
            codigos.append(codigo)
    return sorted(set(codigos)), fict, enchimento


def _ler_json(conteudo: bytes, nome: str) -> dict[str, Any]:
    try:
        objeto = json.loads(conteudo.decode("utf-8-sig"))
    except Exception as exc:
        raise ErroAba04(f"Não foi possível ler {nome}: {exc}") from exc
    if not isinstance(objeto, dict):
        raise ErroAba04(f"{nome} não possui um objeto JSON na raiz.")
    return objeto


def _ler_cobre(
    arquivos_brutos: dict[str, bytes],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    arquivos = expandir_arquivos(arquivos_brutos)
    nome_parquet, bytes_parquet = _localizar(
        arquivos, "HYDRO_ENERGY_PRODUCTIVITY.PARQUET", obrigatorio=True
    )
    nome_modelos, bytes_modelos = _localizar(
        arquivos, "HYDRO_PRODUCTION_MODELS.JSON", obrigatorio=True
    )
    nome_hydros, bytes_hydros = _localizar(
        arquivos, "HYDROS.JSON", obrigatorio=True
    )
    nome_stages, bytes_stages = _localizar(
        arquivos, "STAGES.JSON", obrigatorio=True
    )
    try:
        parquet = pd.read_parquet(io.BytesIO(bytes_parquet))
    except Exception as exc:
        raise ErroAba04(
            f"Não foi possível ler hydro_energy_productivity.parquet: {exc}"
        ) from exc
    return (
        parquet,
        _ler_json(bytes_modelos, nome_modelos),
        _ler_json(bytes_hydros, nome_hydros),
        _ler_json(bytes_stages, nome_stages),
        {nome_parquet, nome_modelos, nome_hydros, nome_stages},
    )


def _avaliar_polinomio(coeficientes: Iterable[float], volume: float) -> float:
    a0, a1, a2, a3, a4 = [float(v) for v in coeficientes]
    return a0 + a1 * volume + a2 * volume**2 + a3 * volume**3 + a4 * volume**4


def _aplicar_perda(queda_bruta: float, tipo_perda: int, perda: float) -> float:
    if math.isnan(perda) or perda <= 0.0:
        return queda_bruta
    if tipo_perda == 1:
        return queda_bruta * (1.0 - perda / 100.0)
    if tipo_perda == 2:
        return queda_bruta - perda
    return queda_bruta


def _calcular_produtividade(
    cadastro: pd.Series,
    *,
    volume_util: float | None = None,
    cfuga: float | None = None,
    cmont: float | None = None,
) -> dict[str, Any]:
    """Reproduz compute_productivity do cobre-bridge 0.15 com memória completa."""
    coeficientes = [float(cadastro[f"a{i}_volume_cota"]) for i in range(5)]
    vmin = float(cadastro["volume_minimo"])
    vmax = float(cadastro["volume_maximo"])
    vref = float(cadastro["volume_referencia"])
    regulacao = str(cadastro["tipo_regulacao"]).strip()
    cfuga_aplicado = (
        float(cfuga) if cfuga is not None else float(cadastro["canal_fuga_medio"])
    )

    if cmont is not None:
        volume = None
        origem_volume = "Não utilizado — CMONT substitui o polinômio"
        cota_montante = float(cmont)
        origem_cota = "MODIF.DAT — CMONT"
    else:
        if volume_util is not None:
            volume = vmin + float(volume_util)
            origem_volume = "VOLREF_SAZ.DAT — VMIN + volume útil mensal"
        elif regulacao == "M":
            volume = vmin + 0.65 * (vmax - vmin)
            origem_volume = "HIDR.DAT — 65% do volume útil"
        else:
            volume = vref
            origem_volume = "HIDR.DAT — volume de referência"

        if all(valor == 0.0 for valor in coeficientes):
            cota_montante = None
            origem_cota = "Polinômio volume–cota totalmente nulo"
        else:
            cota_montante = _avaliar_polinomio(coeficientes, volume)
            origem_cota = "HIDR.DAT — polinômio volume–cota"

    if cota_montante is None:
        queda_bruta = None
        queda_liquida = None
        produtividade = 0.0
    else:
        queda_bruta = cota_montante - cfuga_aplicado
        tipo_perda = int(cadastro["tipo_perda"])
        perda = float(cadastro["perdas"])
        queda_liquida = _aplicar_perda(queda_bruta, tipo_perda, perda)
        produtividade = float(cadastro["produtibilidade_especifica"]) * queda_liquida

    return {
        "vmin_hm3": vmin,
        "vmax_hm3": vmax,
        "vref_hm3": vref,
        "regulacao": regulacao,
        "volume_util_sazonal_hm3": volume_util,
        "volume_absoluto_utilizado_hm3": volume,
        "origem_volume": origem_volume,
        **{f"a{i}_volume_cota": coeficientes[i] for i in range(5)},
        "cota_montante_m": cota_montante,
        "origem_cota_montante": origem_cota,
        "cfmed_hidr_m": float(cadastro["canal_fuga_medio"]),
        "cfuga_modif_m": cfuga,
        "canal_fuga_aplicado_m": cfuga_aplicado,
        "cmont_modif_m": cmont,
        "queda_bruta_m": queda_bruta,
        "tipo_perda": int(cadastro["tipo_perda"]),
        "perda_hidraulica": float(cadastro["perdas"]),
        "queda_liquida_m": queda_liquida,
        "produtividade_especifica_mw_por_m3s_por_m": float(
            cadastro["produtibilidade_especifica"]
        ),
        "produtividade_propria_esperada_mw_por_m3s": produtividade,
    }


def _horizonte(dger: Any) -> dict[str, int]:
    ano = int(dger.ano_inicio_estudo)
    mes = int(dger.mes_inicio_estudo)
    anos = int(dger.num_anos_estudo or 1)
    anos_pos = int(dger.num_anos_pos_estudo or 0)
    meses_estudo = (13 - mes) + (anos - 1) * 12
    return {
        "ano_inicio": ano,
        "mes_inicio": mes,
        "anos_estudo": anos,
        "anos_pos": anos_pos,
        "meses_estudo": meses_estudo,
        "total_estagios": meses_estudo + anos_pos * 12,
    }


def _data_estagio(ano_inicio: int, mes_inicio: int, stage_id: int) -> date:
    indice = mes_inicio - 1 + stage_id
    return date(ano_inicio + indice // 12, indice % 12 + 1, 1)


def _alteracoes_por_estagio(
    alteracoes: list[dict[str, Any]], dger: Any, total: int
) -> list[tuple[float | None, float | None]]:
    """Expande CFUGA/CMONT como funções em degrau, incluindo sazonalização."""
    if not alteracoes:
        return [(None, None)] * total
    ano_inicio = int(dger.ano_inicio_estudo)
    mes_inicio = int(dger.mes_inicio_estudo)
    sazonaliza = int(getattr(dger, "sazonaliza_cfuga_cmont", 0) or 0) == 1
    eventos: dict[int, list[tuple[float | None, float | None]]] = {}
    ultimo_estagio = -1
    for item in alteracoes:
        stage_id = (int(item["year"]) - ano_inicio) * 12 + (
            int(item["month"]) - mes_inicio
        )
        ultimo_estagio = max(ultimo_estagio, stage_id)
        par = (
            (float(item["value"]), None)
            if item["type"] == "CFUGA"
            else (None, float(item["value"]))
        )
        eventos.setdefault(stage_id, []).append(par)

    saz_cfuga: dict[int, float] = {}
    saz_cmont: dict[int, float] = {}
    ano_cfuga: dict[int, int] = {}
    ano_cmont: dict[int, int] = {}
    if sazonaliza:
        for item in alteracoes:
            mes = int(item["month"])
            ano = int(item["year"])
            valor = float(item["value"])
            if item["type"] == "CFUGA" and ano > ano_cfuga.get(mes, -10**9):
                ano_cfuga[mes] = ano
                saz_cfuga[mes] = valor
            if item["type"] == "CMONT" and ano > ano_cmont.get(mes, -10**9):
                ano_cmont[mes] = ano
                saz_cmont[mes] = valor

    resultado: list[tuple[float | None, float | None]] = []
    cfuga_ativo: float | None = None
    cmont_ativo: float | None = None
    for stage_id in range(total):
        estagios_aplicaveis = (
            sorted(e for e in eventos if e <= 0)
            if stage_id == 0
            else ([stage_id] if stage_id in eventos else [])
        )
        for estagio in estagios_aplicaveis:
            for cfuga, cmont in eventos[estagio]:
                if cfuga is not None:
                    cfuga_ativo = cfuga
                if cmont is not None:
                    cmont_ativo = cmont

        mes_calendario = ((mes_inicio - 1 + stage_id) % 12) + 1
        if sazonaliza and stage_id > ultimo_estagio:
            if mes_calendario in saz_cfuga:
                cfuga_ativo = saz_cfuga[mes_calendario]
            if mes_calendario in saz_cmont:
                cmont_ativo = saz_cmont[mes_calendario]
        resultado.append((cfuga_ativo, cmont_ativo))
    return resultado


def _resolver_cascata_ficticia(
    ctx: ContextoNewave, ativos: set[int], ficticias: set[int], enchimento: set[int]
) -> dict[int, dict[str, Any]]:
    """Resolve a cadeia removida; a contribuição é zero pela própria classificação."""
    linhas = {
        int(linha["codigo_usina"]): linha for _, linha in ctx.confhd.iterrows()
    }
    ausentes = {
        codigo
        for codigo, linha in linhas.items()
        if codigo not in ativos and codigo not in ficticias
    }
    fict_por_posto: dict[int, int] = {}
    for codigo in ficticias:
        linha = linhas.get(codigo)
        if linha is not None and not pd.isna(linha.get("posto")):
            fict_por_posto.setdefault(int(linha["posto"]), codigo)

    def caminhar(inicio: int) -> tuple[int | None, list[int]]:
        atual = inicio
        vistos: set[int] = set()
        cadeia: list[int] = []
        while atual not in {0, None} and atual not in vistos:
            vistos.add(atual)
            if atual in ativos:
                return atual, cadeia
            linha = linhas.get(atual)
            if linha is None:
                break
            if atual in ficticias:
                cadeia.append(atual)
            proximo = linha.get("codigo_usina_jusante")
            atual = 0 if proximo is None or pd.isna(proximo) else int(proximo)
        return None, cadeia

    resultado: dict[int, dict[str, Any]] = {}
    for codigo in ativos:
        linha = linhas[codigo]
        jusante = linha.get("codigo_usina_jusante")
        jusante_codigo = 0 if jusante is None or pd.isna(jusante) else int(jusante)
        if jusante_codigo:
            _, cadeia = caminhar(jusante_codigo)
        else:
            posto = linha.get("posto")
            fict = (
                fict_por_posto.get(int(posto))
                if posto is not None and not pd.isna(posto)
                else None
            )
            _, cadeia = caminhar(fict or 0)
        resultado[codigo] = {
            "cadeia_ficticia": ", ".join(map(str, cadeia)) if cadeia else "",
            "contribuicao_ficticia_mw_por_m3s": 0.0,
        }
    return resultado


def _montar_depara(
    ctx: ContextoNewave, hydros_json: dict[str, Any]
) -> tuple[pd.DataFrame, dict[int, int], set[int], set[int]]:
    codigos, ficticias, enchimento = _codigos_ativos(ctx)
    mapa = {codigo: hydro_id for hydro_id, codigo in enumerate(codigos)}
    nomes_newave = {
        int(linha["codigo_usina"]): str(linha["nome_usina"]).strip()
        for _, linha in ctx.confhd.iterrows()
    }
    hydros = hydros_json.get("hydros", [])
    if not isinstance(hydros, list):
        raise ErroAba04("hydros.json não possui a lista 'hydros'.")
    por_id = {
        int(item["id"]): item
        for item in hydros
        if isinstance(item, dict) and item.get("id") is not None
    }
    registros: list[dict[str, Any]] = []
    for codigo, hydro_id in mapa.items():
        item = por_id.get(hydro_id)
        nome_esperado = nomes_newave.get(codigo, "")
        nome_cobre = item.get("name", "") if item else ""
        if item is None:
            resultado = "Usina NEWAVE não representada no COBRE"
        elif normalizar_nome(nome_esperado) != normalizar_nome(nome_cobre):
            resultado = "ID associado a uma usina diferente"
        else:
            resultado = STATUS_COINCIDENTE
        registros.append(
            {
                "codigo_newave": codigo,
                "nome_newave": nome_esperado,
                "hydro_id_esperado": hydro_id,
                "nome_cobre": nome_cobre,
                "resultado": resultado,
            }
        )

    ids_esperados = set(mapa.values())
    for hydro_id in sorted(set(por_id) - ids_esperados):
        registros.append(
            {
                "codigo_newave": pd.NA,
                "nome_newave": "",
                "hydro_id_esperado": hydro_id,
                "nome_cobre": por_id[hydro_id].get("name", ""),
                "resultado": "Usina COBRE sem correspondente NEWAVE esperado",
            }
        )
    return pd.DataFrame(registros), mapa, ficticias, enchimento


def _valor_stage_chave(valor: Any) -> int | None:
    return None if valor is None or pd.isna(valor) else int(valor)


def _montar_produtividade_esperada(
    ctx: ContextoNewave,
    mapa: dict[int, int],
    ficticias: set[int],
    enchimento: set[int],
) -> pd.DataFrame:
    horizonte = _horizonte(ctx.dger)
    total = horizonte["total_estagios"]
    ativos = set(mapa)
    cascata = _resolver_cascata_ficticia(
        ctx, ativos, ficticias=ficticias, enchimento=enchimento
    )
    registros: list[dict[str, Any]] = []
    for codigo in sorted(mapa):
        if codigo not in ctx.cadastro.index:
            continue
        cadastro = ctx.cadastro.loc[codigo]
        alteracoes = ctx.alteracoes_temporais.get(codigo, [])
        volumes_mensais = ctx.volref.get(codigo)
        por_estagio = bool(alteracoes) or bool(volumes_mensais)
        estagios: list[int | None] = list(range(total)) if por_estagio else [None]
        quedas = _alteracoes_por_estagio(alteracoes, ctx.dger, total)
        for stage_id in estagios:
            indice = 0 if stage_id is None else stage_id
            data_ref = _data_estagio(
                horizonte["ano_inicio"], horizonte["mes_inicio"], indice
            )
            mes = data_ref.month
            cfuga, cmont = quedas[indice]
            volume_util = volumes_mensais.get(mes) if volumes_mensais else None
            memoria = _calcular_produtividade(
                cadastro, volume_util=volume_util, cfuga=cfuga, cmont=cmont
            )
            fict = cascata.get(codigo, {})
            extra = float(fict.get("contribuicao_ficticia_mw_por_m3s", 0.0))
            esperado = (
                float(memoria["produtividade_propria_esperada_mw_por_m3s"])
                + extra
            )
            registros.append(
                {
                    "codigo_newave": codigo,
                    "nome_newave": str(cadastro["nome_usina"]).strip(),
                    "hydro_id": mapa[codigo],
                    "stage_id": stage_id,
                    "tipo_linha": "Padrão para todos os estágios"
                    if stage_id is None
                    else "Específica por estágio",
                    "data_estagio": None if stage_id is None else data_ref.isoformat(),
                    "mes_calendario": mes,
                    "mes_nome": MESES[mes - 1],
                    **memoria,
                    **fict,
                    "produtividade_esperada_mw_por_m3s": esperado,
                }
            )
    return pd.DataFrame(registros)


def _motivo_nao_validado_produtividade(
    ctx: ContextoNewave, linha: pd.Series
) -> str | None:
    if "MODIF.DAT" not in ctx.arquivos_presentes:
        return "MODIF.DAT não fornecido; não é possível excluir alterações de volume, CFUGA ou CMONT"
    if (
        linha.get("tipo_linha") == "Específica por estágio"
        and "VOLREF_SAZ.DAT" not in ctx.arquivos_presentes
    ):
        return "VOLREF_SAZ.DAT não fornecido; não é possível excluir uma referência sazonal"
    return None


def _comparar_produtividade(
    esperado: pd.DataFrame,
    encontrado: pd.DataFrame,
    ctx: ContextoNewave,
    tolerancia: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    colunas = {
        "hydro_id",
        "stage_id",
        "equivalent_productivity_mw_per_m3s",
        "reference_outflow_m3s",
        "specific_productivity_mw_per_m3s_per_m",
    }
    ausentes = colunas - set(encontrado.columns)
    if ausentes:
        raise ErroAba04(
            "O parquet não possui as colunas obrigatórias: "
            + ", ".join(sorted(ausentes))
        )

    por_chave: dict[tuple[int, int | None], list[pd.Series]] = {}
    for _, linha in encontrado.iterrows():
        chave = (int(linha["hydro_id"]), _valor_stage_chave(linha["stage_id"]))
        por_chave.setdefault(chave, []).append(linha)

    comparacoes: list[dict[str, Any]] = []
    memorias: list[dict[str, Any]] = []
    inconsistencias: list[dict[str, Any]] = []
    chaves_esperadas: set[tuple[int, int | None]] = set()

    for _, linha in esperado.iterrows():
        chave = (int(linha["hydro_id"]), _valor_stage_chave(linha["stage_id"]))
        chaves_esperadas.add(chave)
        achadas = por_chave.get(chave, [])
        valor_esperado = float(linha["produtividade_esperada_mw_por_m3s"])
        valor_cobre = (
            float(achadas[0]["equivalent_productivity_mw_per_m3s"])
            if achadas
            else None
        )
        diferenca = (
            abs(valor_esperado - valor_cobre) if valor_cobre is not None else None
        )
        motivo = _motivo_nao_validado_produtividade(ctx, linha)
        if (
            motivo is None
            and "VOLREF_SAZ.DAT" not in ctx.arquivos_presentes
            and not achadas
            and any(
                outra_chave[0] == chave[0] and outra_chave[1] is not None
                for outra_chave in por_chave
            )
        ):
            motivo = (
                "VOLREF_SAZ.DAT não fornecido; o COBRE possui linhas por estágio"
            )
        if motivo:
            resultado = f"{STATUS_NAO_VALIDADO} — {motivo}"
        elif not achadas:
            resultado = f"{STATUS_DIVERGENTE} — linha esperada ausente no parquet"
        elif len(achadas) > 1:
            resultado = f"{STATUS_DIVERGENTE} — chave duplicada no parquet"
        elif diferenca is not None and diferenca <= tolerancia:
            resultado = STATUS_COINCIDENTE
        else:
            resultado = STATUS_DIVERGENTE

        resumo = {
            "codigo_newave": linha["codigo_newave"],
            "nome_newave": linha["nome_newave"],
            "hydro_id": linha["hydro_id"],
            "stage_id": linha["stage_id"],
            "data_estagio": linha["data_estagio"],
            "mes_nome": linha["mes_nome"],
            "tipo_linha": linha["tipo_linha"],
            "origem_volume": linha["origem_volume"],
            "produtividade_esperada_mw_por_m3s": valor_esperado,
            "produtividade_cobre_mw_por_m3s": valor_cobre,
            "diferenca_absoluta": diferenca,
            "tolerancia": tolerancia,
            "resultado": resultado,
        }
        comparacoes.append(resumo)
        memoria = linha.to_dict()
        memoria.update(
            {
                "produtividade_cobre_mw_por_m3s": valor_cobre,
                "diferenca_absoluta": diferenca,
                "tolerancia": tolerancia,
                "resultado": resultado,
            }
        )
        memorias.append(memoria)

        if achadas:
            registro = achadas[0]
            for campo in (
                "reference_outflow_m3s",
                "specific_productivity_mw_per_m3s_per_m",
            ):
                if not pd.isna(registro[campo]):
                    inconsistencias.append(
                        {
                            "categoria": "Esquema do parquet",
                            "codigo_newave": linha["codigo_newave"],
                            "hydro_id": linha["hydro_id"],
                            "stage_id": linha["stage_id"],
                            "campo": campo,
                            "esperado": "nulo",
                            "encontrado": registro[campo],
                            "resultado": STATUS_DIVERGENTE,
                        }
                    )

    for chave, linhas in por_chave.items():
        if chave in chaves_esperadas:
            continue
        for linha in linhas:
            faltantes = []
            if "MODIF.DAT" not in ctx.arquivos_presentes:
                faltantes.append("MODIF.DAT")
            if "VOLREF_SAZ.DAT" not in ctx.arquivos_presentes:
                faltantes.append("VOLREF_SAZ.DAT")
            if faltantes:
                resultado_extra = (
                    f"{STATUS_NAO_VALIDADO} — "
                    + " e ".join(faltantes)
                    + " não fornecido(s)"
                )
            else:
                resultado_extra = (
                    f"{STATUS_DIVERGENTE} — linha não esperada no parquet"
                )
            comparacoes.append(
                {
                    "codigo_newave": pd.NA,
                    "nome_newave": "",
                    "hydro_id": chave[0],
                    "stage_id": chave[1],
                    "data_estagio": None,
                    "mes_nome": "",
                    "tipo_linha": "Linha extra",
                    "origem_volume": "",
                    "produtividade_esperada_mw_por_m3s": None,
                    "produtividade_cobre_mw_por_m3s": linha[
                        "equivalent_productivity_mw_per_m3s"
                    ],
                    "diferenca_absoluta": None,
                    "tolerancia": tolerancia,
                    "resultado": resultado_extra,
                }
            )
    return (
        pd.DataFrame(comparacoes),
        pd.DataFrame(memorias),
        inconsistencias,
    )


def _fpha_elegivel(cadastro: pd.Series) -> bool:
    coeficientes = [float(cadastro[f"a{i}_volume_cota"]) for i in range(5)]
    rho = float(cadastro["produtibilidade_especifica"])
    return not all(v == 0.0 for v in coeficientes) and not math.isnan(rho) and rho > 0


def _configuracao_fpha(cadastro: pd.Series) -> dict[str, Any]:
    if str(cadastro["tipo_regulacao"]).strip() == "M":
        minimo = float(cadastro["volume_minimo"])
        maximo = float(cadastro["volume_maximo"])
    else:
        minimo = maximo = float(cadastro["volume_referencia"])
    return {
        "source": "computed",
        "fitting_window": {
            "volume_min_hm3": minimo,
            "volume_max_hm3": maximo,
        },
    }


def _parsear_reducao_planos(conteudo: bytes | None) -> dict[str, Any] | None:
    if conteudo is None:
        return None
    texto = conteudo.decode("utf-8-sig", errors="replace")
    metodos: list[dict[str, Any]] = []
    for bruta in texto.splitlines():
        linha = bruta.strip()
        if not linha or linha.startswith("&"):
            continue
        partes = [p.strip() for p in linha.split(";")]
        if len(partes) < 2:
            continue
        try:
            valor = float(partes[1].replace(",", "."))
        except ValueError:
            continue
        token = partes[0].upper()
        if token.endswith("ANGULO-PADRAO"):
            metodos.append({"method": "angle", "tolerance_deg": valor})
        elif token.endswith("DISTANCIA-PADRAO"):
            metodos.append(
                {"method": "distance", "tolerance_pct": valor, "n_samples": 100}
            )
    return metodos[0] if metodos else None


def _modelos_esperados(
    ctx: ContextoNewave, mapa: dict[int, int]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    fpha_global = getattr(ctx.dger, "funcao_producao_uhe", None) == 0
    elegiveis = {
        codigo
        for codigo in mapa
        if codigo in ctx.cadastro.index
        and fpha_global
        and _fpha_elegivel(ctx.cadastro.loc[codigo])
    }
    volref_modelo = ctx.volref if fpha_global else {}
    modelos: dict[int, dict[str, Any]] = {}
    for codigo in sorted(mapa):
        if codigo not in ctx.cadastro.index:
            continue
        hydro_id = mapa[codigo]
        cadastro = ctx.cadastro.loc[codigo]
        is_fpha = codigo in elegiveis
        meses = volref_modelo.get(codigo)
        fpha_config = _configuracao_fpha(cadastro) if is_fpha else None
        if meses:
            vmin = float(cadastro["volume_minimo"])
            vmax = float(cadastro["volume_maximo"])
            estacoes: list[dict[str, Any]] = []
            for mes in range(1, 13):
                util = float(meses.get(mes, 0.0))
                absoluto = max(vmin, min(vmax, vmin + util))
                estacao: dict[str, Any] = {
                    "season_id": mes - 1,
                    "model": "fpha" if is_fpha else "constant_productivity",
                    "reference_volume": {"volume_hm3": absoluto},
                }
                if is_fpha:
                    estacao["fpha_config"] = fpha_config
                estacoes.append(estacao)
            modelo = {
                "hydro_id": hydro_id,
                "selection_mode": "seasonal",
                "default_model": "constant_productivity",
                "seasons": estacoes,
            }
        elif is_fpha:
            modelo = {
                "hydro_id": hydro_id,
                "selection_mode": "stage_ranges",
                "stage_ranges": [
                    {
                        "start_stage_id": 0,
                        "end_stage_id": None,
                        "model": "fpha",
                        "fpha_config": fpha_config,
                        "reference_volume": {"percentile": 0.65},
                    }
                ],
            }
        else:
            modelo = {
                "hydro_id": hydro_id,
                "selection_mode": "stage_ranges",
                "stage_ranges": [
                    {
                        "start_stage_id": 0,
                        "end_stage_id": None,
                        "model": "constant_productivity",
                    }
                ],
            }
        modelos[hydro_id] = modelo

    arquivo: dict[str, Any] = {"production_models": list(modelos.values())}
    reducao = _parsear_reducao_planos(ctx.tratamento_fpha)
    if elegiveis and reducao is not None:
        arquivo["fpha_plane_reduction"] = reducao
    return arquivo, modelos


_AUSENTE = object()


def _achatar(objeto: Any, prefixo: str = "") -> dict[str, Any]:
    """Achata JSON preservando posições de listas e valores nulos."""
    if isinstance(objeto, dict):
        resultado: dict[str, Any] = {}
        if not objeto:
            resultado[prefixo] = {}
        for chave, valor in objeto.items():
            caminho = f"{prefixo}.{chave}" if prefixo else str(chave)
            resultado.update(_achatar(valor, caminho))
        return resultado
    if isinstance(objeto, list):
        resultado = {}
        if not objeto:
            resultado[prefixo] = []
        for indice, valor in enumerate(objeto):
            resultado.update(_achatar(valor, f"{prefixo}[{indice}]"))
        return resultado
    return {prefixo: objeto}


def _modelo_principal(modelo: dict[str, Any] | None) -> str:
    if not modelo:
        return "Ausente"
    if modelo.get("selection_mode") == "seasonal":
        valores = {
            str(item.get("model", "")) for item in modelo.get("seasons", [])
        }
    else:
        valores = {
            str(item.get("model", "")) for item in modelo.get("stage_ranges", [])
        }
    return ", ".join(sorted(v for v in valores if v)) or "Não informado"


def _comparar_valores(
    esperado: Any, encontrado: Any, caminho: str, tol_volume: float
) -> tuple[bool, float | None, float]:
    tolerancia = tol_volume if "volume" in caminho.lower() else 0.0
    if esperado is _AUSENTE or encontrado is _AUSENTE:
        return False, None, tolerancia
    if isinstance(esperado, (int, float)) and isinstance(
        encontrado, (int, float)
    ):
        diferenca = abs(float(esperado) - float(encontrado))
        return diferenca <= tolerancia, diferenca, tolerancia
    return esperado == encontrado, None, tolerancia


def _comparar_modelos(
    ctx: ContextoNewave,
    esperado_arquivo: dict[str, Any],
    esperado_por_id: dict[int, dict[str, Any]],
    encontrado_arquivo: dict[str, Any],
    mapa: dict[int, int],
    tol_volume: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    lista_encontrada = encontrado_arquivo.get("production_models", [])
    if not isinstance(lista_encontrada, list):
        raise ErroAba04(
            "hydro_production_models.json não possui a lista 'production_models'."
        )
    encontrado_por_id = {
        int(item["hydro_id"]): item
        for item in lista_encontrada
        if isinstance(item, dict) and item.get("hydro_id") is not None
    }
    codigo_por_id = {hid: codigo for codigo, hid in mapa.items()}
    nome_por_codigo = {
        int(linha["codigo_usina"]): str(linha["nome_usina"]).strip()
        for _, linha in ctx.confhd.iterrows()
    }

    resumos: list[dict[str, Any]] = []
    parametros: list[dict[str, Any]] = []
    fitting: list[dict[str, Any]] = []
    volumes: list[dict[str, Any]] = []
    inconsistencias: list[dict[str, Any]] = []

    todos_ids = sorted(set(esperado_por_id) | set(encontrado_por_id))
    for hydro_id in todos_ids:
        esperado = esperado_por_id.get(hydro_id)
        encontrado = encontrado_por_id.get(hydro_id)
        codigo = codigo_por_id.get(hydro_id)
        nome = nome_por_codigo.get(codigo, "") if codigo is not None else ""
        plano = _achatar(esperado) if esperado is not None else {}
        real = _achatar(encontrado) if encontrado is not None else {}
        resultados_usina: list[str] = []
        sazonal_sem_fonte = (
            "VOLREF_SAZ.DAT" not in ctx.arquivos_presentes
            and (
                (esperado or {}).get("selection_mode") == "seasonal"
                or (encontrado or {}).get("selection_mode") == "seasonal"
            )
        )
        for caminho in sorted(set(plano) | set(real)):
            valor_esperado = plano.get(caminho, _AUSENTE)
            valor_cobre = real.get(caminho, _AUSENTE)
            igual, diferenca, tolerancia = _comparar_valores(
                valor_esperado, valor_cobre, caminho, tol_volume
            )
            if sazonal_sem_fonte:
                resultado = (
                    f"{STATUS_NAO_VALIDADO} — VOLREF_SAZ.DAT não fornecido"
                )
            elif (
                "MODIF.DAT" not in ctx.arquivos_presentes
                and (
                    "fitting_window" in caminho
                    or "reference_volume.volume_hm3" in caminho
                )
            ):
                resultado = f"{STATUS_NAO_VALIDADO} — MODIF.DAT não fornecido"
            else:
                resultado = STATUS_COINCIDENTE if igual else STATUS_DIVERGENTE
            resultados_usina.append(resultado)
            parametros.append(
                {
                    "codigo_newave": codigo,
                    "nome_newave": nome,
                    "hydro_id": hydro_id,
                    "campo": caminho,
                    "valor_esperado": "<ausente>"
                    if valor_esperado is _AUSENTE
                    else valor_esperado,
                    "valor_cobre": "<ausente>"
                    if valor_cobre is _AUSENTE
                    else valor_cobre,
                    "diferenca_absoluta": diferenca,
                    "tolerancia": tolerancia,
                    "resultado": resultado,
                }
            )

        if any(r.startswith(STATUS_DIVERGENTE) for r in resultados_usina):
            resultado_geral = STATUS_DIVERGENTE
        elif any(r.startswith(STATUS_NAO_VALIDADO) for r in resultados_usina):
            resultado_geral = STATUS_NAO_VALIDADO
        elif resultados_usina:
            resultado_geral = STATUS_COINCIDENTE
        else:
            resultado_geral = STATUS_DIVERGENTE
        resumos.append(
            {
                "codigo_newave": codigo,
                "nome_newave": nome,
                "hydro_id": hydro_id,
                "selection_mode_esperado": esperado.get("selection_mode")
                if esperado
                else "Ausente",
                "selection_mode_cobre": encontrado.get("selection_mode")
                if encontrado
                else "Ausente",
                "modelo_esperado": _modelo_principal(esperado),
                "modelo_cobre": _modelo_principal(encontrado),
                "resultado": resultado_geral,
            }
        )

        if esperado:
            itens = (
                esperado.get("seasons", [])
                if esperado.get("selection_mode") == "seasonal"
                else esperado.get("stage_ranges", [])
            )
            itens_cobre = (
                encontrado.get("seasons", [])
                if encontrado and encontrado.get("selection_mode") == "seasonal"
                else encontrado.get("stage_ranges", []) if encontrado else []
            )
            por_season_cobre = {
                item.get("season_id", indice): item
                for indice, item in enumerate(itens_cobre)
            }
            for indice, item in enumerate(itens):
                chave = item.get("season_id", indice)
                item_cobre = por_season_cobre.get(chave, {})
                config = item.get("fpha_config", {})
                config_cobre = item_cobre.get("fpha_config", {})
                janela = config.get("fitting_window")
                if janela:
                    janela_cobre = config_cobre.get("fitting_window", {})
                    for limite in ("volume_min_hm3", "volume_max_hm3"):
                        esperado_limite = janela.get(limite)
                        cobre_limite = janela_cobre.get(limite)
                        diferenca = (
                            abs(float(esperado_limite) - float(cobre_limite))
                            if isinstance(esperado_limite, (int, float))
                            and isinstance(cobre_limite, (int, float))
                            else None
                        )
                        if "MODIF.DAT" not in ctx.arquivos_presentes:
                            resultado_limite = STATUS_NAO_VALIDADO
                        else:
                            resultado_limite = (
                                STATUS_COINCIDENTE
                                if diferenca is not None
                                and diferenca <= tol_volume
                                else STATUS_DIVERGENTE
                            )
                        fitting.append(
                            {
                                "codigo_newave": codigo,
                                "nome_newave": nome,
                                "hydro_id": hydro_id,
                                "season_id": item.get("season_id"),
                                "limite": limite,
                                "valor_esperado_hm3": esperado_limite,
                                "valor_cobre_hm3": cobre_limite,
                                "diferenca_absoluta": diferenca,
                                "resultado": resultado_limite,
                            }
                        )
                referencia = item.get("reference_volume", {})
                if "volume_hm3" in referencia:
                    referencia_cobre = item_cobre.get("reference_volume", {})
                    valor_esperado = referencia["volume_hm3"]
                    valor_cobre = referencia_cobre.get("volume_hm3")
                    diferenca = (
                        abs(float(valor_esperado) - float(valor_cobre))
                        if isinstance(valor_cobre, (int, float))
                        else None
                    )
                    if "VOLREF_SAZ.DAT" not in ctx.arquivos_presentes:
                        resultado_volume = STATUS_NAO_VALIDADO
                    elif "MODIF.DAT" not in ctx.arquivos_presentes:
                        resultado_volume = STATUS_NAO_VALIDADO
                    else:
                        resultado_volume = (
                            STATUS_COINCIDENTE
                            if diferenca is not None and diferenca <= tol_volume
                            else STATUS_DIVERGENTE
                        )
                    volumes.append(
                        {
                            "codigo_newave": codigo,
                            "nome_newave": nome,
                            "hydro_id": hydro_id,
                            "season_id": item.get("season_id"),
                            "mes": MESES[int(item.get("season_id", 0))],
                            "volume_esperado_hm3": valor_esperado,
                            "volume_cobre_hm3": valor_cobre,
                            "diferenca_absoluta": diferenca,
                            "resultado": resultado_volume,
                        }
                    )

    reducao_esperada = esperado_arquivo.get("fpha_plane_reduction", _AUSENTE)
    reducao_cobre = encontrado_arquivo.get("fpha_plane_reduction", _AUSENTE)
    if reducao_esperada is _AUSENTE and reducao_cobre is not _AUSENTE:
        if ctx.tratamento_fpha is None:
            inconsistencias.append(
                {
                    "categoria": "Redução de planos FPHA",
                    "codigo_newave": pd.NA,
                    "hydro_id": pd.NA,
                    "stage_id": pd.NA,
                    "campo": "fpha_plane_reduction",
                    "esperado": "Não validado sem o arquivo de tratamento FPHA",
                    "encontrado": json.dumps(reducao_cobre, ensure_ascii=False),
                    "resultado": f"{STATUS_NAO_VALIDADO} — arquivo de tratamento FPHA não fornecido",
                }
            )
        else:
            inconsistencias.append(
                {
                    "categoria": "Redução de planos FPHA",
                    "codigo_newave": pd.NA,
                    "hydro_id": pd.NA,
                    "stage_id": pd.NA,
                    "campo": "fpha_plane_reduction",
                    "esperado": "ausente",
                    "encontrado": json.dumps(reducao_cobre, ensure_ascii=False),
                    "resultado": STATUS_DIVERGENTE,
                }
            )
    elif reducao_esperada is not _AUSENTE:
        igual = reducao_esperada == reducao_cobre
        inconsistencias.append(
            {
                "categoria": "Redução de planos FPHA",
                "codigo_newave": pd.NA,
                "hydro_id": pd.NA,
                "stage_id": pd.NA,
                "campo": "fpha_plane_reduction",
                "esperado": json.dumps(reducao_esperada, ensure_ascii=False),
                "encontrado": "ausente"
                if reducao_cobre is _AUSENTE
                else json.dumps(reducao_cobre, ensure_ascii=False),
                "resultado": STATUS_COINCIDENTE if igual else STATUS_DIVERGENTE,
            }
        )

    return (
        pd.DataFrame(resumos),
        pd.DataFrame(parametros),
        pd.DataFrame(fitting),
        pd.DataFrame(volumes),
        inconsistencias,
    )


def _validar_consistencias(
    ctx: ContextoNewave,
    mapa: dict[int, int],
    depara: pd.DataFrame,
    produtividade: pd.DataFrame,
    modelos: pd.DataFrame,
    parquet: pd.DataFrame,
    hydros_json: dict[str, Any],
    stages_json: dict[str, Any],
    inconsistencias_iniciais: list[dict[str, Any]],
) -> pd.DataFrame:
    itens = list(inconsistencias_iniciais)

    for _, linha in depara.iterrows():
        if linha["resultado"] != STATUS_COINCIDENTE:
            itens.append(
                {
                    "categoria": "De-para de usinas",
                    "codigo_newave": linha["codigo_newave"],
                    "hydro_id": linha["hydro_id_esperado"],
                    "stage_id": pd.NA,
                    "campo": "código NEWAVE → hydro_id",
                    "esperado": linha["nome_newave"],
                    "encontrado": linha["nome_cobre"],
                    "resultado": STATUS_DIVERGENTE,
                }
            )

    hydros = {
        int(item["id"]): item
        for item in hydros_json.get("hydros", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    for _, linha in modelos.iterrows():
        hydro_id = int(linha["hydro_id"])
        modelo_hydros = (
            hydros.get(hydro_id, {}).get("generation", {}).get("model")
        )
        esperado = linha["modelo_esperado"]
        if "," not in esperado and esperado != "Ausente" and modelo_hydros != esperado:
            itens.append(
                {
                    "categoria": "Consistência entre arquivos COBRE",
                    "codigo_newave": linha["codigo_newave"],
                    "hydro_id": hydro_id,
                    "stage_id": pd.NA,
                    "campo": "hydros.json generation.model",
                    "esperado": esperado,
                    "encontrado": modelo_hydros,
                    "resultado": STATUS_DIVERGENTE,
                }
            )

    horizonte = _horizonte(ctx.dger)
    stages = stages_json.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    por_id = {
        int(item["id"]): item
        for item in stages
        if isinstance(item, dict) and item.get("id") is not None
    }
    if len(por_id) != horizonte["total_estagios"]:
        itens.append(
            {
                "categoria": "Horizonte",
                "codigo_newave": pd.NA,
                "hydro_id": pd.NA,
                "stage_id": pd.NA,
                "campo": "quantidade de estágios",
                "esperado": horizonte["total_estagios"],
                "encontrado": len(por_id),
                "resultado": STATUS_DIVERGENTE,
            }
        )
    for stage_id in range(horizonte["total_estagios"]):
        esperado_data = _data_estagio(
            horizonte["ano_inicio"], horizonte["mes_inicio"], stage_id
        )
        item = por_id.get(stage_id)
        if item is None:
            continue
        data_cobre = str(item.get("start_date", ""))[:10]
        season_cobre = item.get("season_id")
        if data_cobre != esperado_data.isoformat() or season_cobre != esperado_data.month - 1:
            itens.append(
                {
                    "categoria": "Horizonte",
                    "codigo_newave": pd.NA,
                    "hydro_id": pd.NA,
                    "stage_id": stage_id,
                    "campo": "data/season_id",
                    "esperado": f"{esperado_data.isoformat()} / {esperado_data.month - 1}",
                    "encontrado": f"{data_cobre} / {season_cobre}",
                    "resultado": STATUS_DIVERGENTE,
                }
            )

    # Uma linha nula deve ser única; uma série por estágio deve cobrir o horizonte.
    for hydro_id, grupo in parquet.groupby("hydro_id", dropna=False):
        stages_ids = [_valor_stage_chave(v) for v in grupo["stage_id"]]
        tem_nulo = any(v is None for v in stages_ids)
        if tem_nulo and len(stages_ids) != 1:
            itens.append(
                {
                    "categoria": "Cobertura do parquet",
                    "codigo_newave": pd.NA,
                    "hydro_id": int(hydro_id),
                    "stage_id": pd.NA,
                    "campo": "stage_id nulo",
                    "esperado": "uma única linha padrão",
                    "encontrado": f"{len(stages_ids)} linhas",
                    "resultado": STATUS_DIVERGENTE,
                }
            )

    if not itens:
        return pd.DataFrame(
            columns=[
                "categoria",
                "codigo_newave",
                "hydro_id",
                "stage_id",
                "campo",
                "esperado",
                "encontrado",
                "resultado",
            ]
        )
    return pd.DataFrame(itens)


def executar_validacao(
    arquivos_newave: dict[str, bytes],
    arquivos_cobre: dict[str, bytes],
    *,
    tolerancia_produtividade: float = 1e-8,
    tolerancia_volume: float = 1e-6,
) -> ResultadoAba04:
    """Executa toda a Aba 4 e devolve tabelas independentes da interface."""
    ctx = _ler_newave(arquivos_newave)
    parquet, modelos_json, hydros_json, stages_json, arquivos_cobre_presentes = (
        _ler_cobre(arquivos_cobre)
    )
    depara, mapa, ficticias, enchimento = _montar_depara(ctx, hydros_json)
    if (
        "EXPH.DAT" not in ctx.arquivos_presentes
        and not depara.empty
        and (depara["resultado"] != STATUS_COINCIDENTE).any()
    ):
        raise ErroAba04(
            "O de-para não pôde ser reconstruído com segurança e o EXPH.DAT "
            "não foi fornecido. Envie esse arquivo para verificar se existem "
            "usinas futuras admitidas pelo cronograma de enchimento."
        )
    esperado_prod = _montar_produtividade_esperada(
        ctx, mapa, ficticias=ficticias, enchimento=enchimento
    )
    produtividade, memoria, inconsistencias_parquet = _comparar_produtividade(
        esperado_prod,
        parquet,
        ctx,
        tolerancia=tolerancia_produtividade,
    )

    esperado_arquivo, esperado_por_id = _modelos_esperados(ctx, mapa)
    modelos, parametros, fitting, volumes, inconsistencias_modelo = _comparar_modelos(
        ctx,
        esperado_arquivo,
        esperado_por_id,
        modelos_json,
        mapa,
        tolerancia_volume,
    )
    inconsistencias = _validar_consistencias(
        ctx,
        mapa,
        depara,
        produtividade,
        modelos,
        parquet,
        hydros_json,
        stages_json,
        inconsistencias_parquet + inconsistencias_modelo,
    )

    usados = []
    for nome in sorted(ctx.arquivos_presentes):
        usados.append({"modelo": "NEWAVE", "arquivo": nome, "situação": "Lido"})
    for nome in sorted(arquivos_cobre_presentes):
        usados.append({"modelo": "COBRE", "arquivo": nome, "situação": "Lido"})
    for nome in ("EXPH.DAT", "MODIF.DAT", "VOLREF_SAZ.DAT"):
        if nome not in ctx.arquivos_presentes:
            usados.append(
                {
                    "modelo": "NEWAVE",
                    "arquivo": nome,
                    "situação": "Não fornecido — validação condicional",
                }
            )

    tabelas_status = [produtividade, parametros, inconsistencias]
    total_coincidentes = sum(
        int((df["resultado"] == STATUS_COINCIDENTE).sum())
        for df in tabelas_status
        if not df.empty and "resultado" in df.columns
    )
    total_divergencias = sum(
        int(df["resultado"].astype(str).str.startswith(STATUS_DIVERGENTE).sum())
        for df in tabelas_status
        if not df.empty and "resultado" in df.columns
    )
    total_nao_validados = sum(
        int(df["resultado"].astype(str).str.startswith(STATUS_NAO_VALIDADO).sum())
        for df in tabelas_status
        if not df.empty and "resultado" in df.columns
    )
    aprovada = total_divergencias == 0 and total_nao_validados == 0

    return ResultadoAba04(
        depara=depara,
        produtividade=produtividade,
        memoria_calculo=memoria,
        modelos=modelos,
        parametros_modelos=parametros,
        fitting_windows=fitting,
        volumes_referencia=volumes,
        inconsistencias=inconsistencias,
        arquivos_utilizados=pd.DataFrame(usados),
        total_usinas_newave=len(mapa),
        total_usinas_cobre=len(hydros_json.get("hydros", [])),
        total_linhas_parquet=len(parquet),
        total_parametros_json=len(parametros),
        total_coincidentes=total_coincidentes,
        total_divergencias=total_divergencias,
        total_nao_validados=total_nao_validados,
        aprovada=aprovada,
    )


def _bytes_uploads(uploads: list[Any] | None) -> dict[str, bytes]:
    return {arquivo.name: arquivo.getvalue() for arquivo in uploads or []}


def _assinatura_arquivos(*grupos: dict[str, bytes]) -> str:
    hash_total = hashlib.sha256()
    for grupo in grupos:
        for nome in sorted(grupo):
            hash_total.update(nome.encode("utf-8"))
            hash_total.update(grupo[nome])
    return hash_total.hexdigest()


def _filtrar_tabela(
    df: pd.DataFrame,
    *,
    texto: str,
    resultado: str,
    somente_divergencias: bool,
) -> pd.DataFrame:
    filtrado = df.copy()
    if texto and not filtrado.empty:
        alvo = texto.upper()
        mascara = pd.Series(False, index=filtrado.index)
        for coluna in ("nome_newave", "codigo_newave", "hydro_id"):
            if coluna in filtrado.columns:
                mascara |= filtrado[coluna].astype(str).str.upper().str.contains(
                    re.escape(alvo), na=False
                )
        filtrado = filtrado[mascara]
    if resultado != "Todos" and "resultado" in filtrado.columns:
        filtrado = filtrado[
            filtrado["resultado"].astype(str).str.startswith(resultado)
        ]
    if somente_divergencias and "resultado" in filtrado.columns:
        filtrado = filtrado[
            ~filtrado["resultado"].astype(str).str.startswith(STATUS_COINCIDENTE)
        ]
    return filtrado


def mostrar_aba() -> None:
    """Renderiza a quarta aba no Streamlit."""
    import streamlit as st

    st.subheader("Produtividade energética e modelo de produção hidráulica")
    st.info(
        "**Como esta validação funciona**\n\n"
        "O NEWAVE não fornece a produtividade do parquet como um número pronto. "
        "O conversor calcula esse coeficiente a partir dos dados hidráulicos do "
        "deck NEWAVE e o grava como entrada do COBRE.\n\n"
        "Esta aba refaz a mesma transformação. Para cada usina e estágio, são "
        "mostrados os valores originais, a regra aplicada, o resultado esperado "
        "e o valor encontrado no COBRE.\n\n"
        "Quando `stage_id` está vazio, o coeficiente é o padrão válido para todos "
        "os estágios. Quando existe um número, o valor foi calculado especificamente "
        "para aquele estágio.\n\n"
        "O `hydro_production_models.json` também é validado porque define se a "
        "usina usa FPHA ou produtividade constante, além dos volumes e estágios "
        "que configuram esse modelo."
    )

    st.markdown(
        "Envie os arquivos separados **ou o ZIP completo de cada deck**. "
        "O programa localizará internamente os nomes necessários."
    )
    coluna_nw, coluna_cobre = st.columns(2)
    with coluna_nw:
        uploads_newave = st.file_uploader(
            "Deck NEWAVE",
            accept_multiple_files=True,
            key="aba04_newave",
            help=(
                "Obrigatórios: DGER.DAT, CONFHD.DAT e HIDR.DAT. "
                "Quando aplicáveis: EXPH.DAT, MODIF.DAT, VOLREF_SAZ.DAT e "
                "arquivo de tratamento FPHA. Também pode ser enviado um ZIP."
            ),
        )
    with coluna_cobre:
        uploads_cobre = st.file_uploader(
            "Deck COBRE",
            accept_multiple_files=True,
            key="aba04_cobre",
            help=(
                "Necessários: hydro_energy_productivity.parquet, "
                "hydro_production_models.json, hydros.json e stages.json. "
                "Também pode ser enviado o ZIP completo do deck."
            ),
        )

    with st.expander("Tolerâncias", expanded=False):
        col_tol_prod, col_tol_vol = st.columns(2)
        with col_tol_prod:
            tol_prod = st.number_input(
                "Produtividade — MW/(m³/s)",
                min_value=0.0,
                value=1e-8,
                format="%.10f",
                key="aba04_tol_prod",
            )
        with col_tol_vol:
            tol_volume = st.number_input(
                "Volumes e fitting window — hm³",
                min_value=0.0,
                value=1e-6,
                format="%.8f",
                key="aba04_tol_volume",
            )

    arquivos_nw = _bytes_uploads(uploads_newave)
    arquivos_cb = _bytes_uploads(uploads_cobre)
    executar = st.button(
        "Executar validação da Aba 4",
        type="primary",
        disabled=not arquivos_nw or not arquivos_cb,
        key="aba04_executar",
    )

    if executar:
        try:
            with st.spinner("Recalculando a produtividade e validando os modelos..."):
                resultado = executar_validacao(
                    arquivos_nw,
                    arquivos_cb,
                    tolerancia_produtividade=float(tol_prod),
                    tolerancia_volume=float(tol_volume),
                )
            st.session_state["aba04_resultado"] = resultado
            st.session_state["aba04_assinatura"] = _assinatura_arquivos(
                arquivos_nw, arquivos_cb
            )
        except ErroAba04 as exc:
            st.error(str(exc))
            st.session_state.pop("aba04_resultado", None)
            return
        except Exception as exc:
            st.error(f"Erro inesperado durante a validação: {exc}")
            st.session_state.pop("aba04_resultado", None)
            return

    resultado: ResultadoAba04 | None = st.session_state.get("aba04_resultado")
    if not arquivos_nw or not arquivos_cb:
        st.caption("Envie os dois conjuntos de arquivos e execute a validação.")
        return
    if resultado is None:
        st.caption("Envie os dois conjuntos de arquivos e execute a validação.")
        return

    assinatura_atual = _assinatura_arquivos(arquivos_nw, arquivos_cb)
    if arquivos_nw and arquivos_cb and assinatura_atual != st.session_state.get(
        "aba04_assinatura"
    ):
        st.warning(
            "Os arquivos selecionados mudaram. Clique novamente em “Executar "
            "validação” para atualizar os resultados."
        )

    if resultado.aprovada:
        st.success(
            "Validação aprovada: os cálculos reconstruídos e os modelos COBRE "
            "coincidem com o deck NEWAVE."
        )
    elif resultado.total_divergencias:
        st.error(
            f"Foram encontradas {resultado.total_divergencias} divergências. "
            "Use as tabelas abaixo para localizar a etapa responsável."
        )
    else:
        st.warning(
            "Não há divergências comprovadas, mas existem itens não validados "
            "por falta de arquivo condicional."
        )

    metricas = st.columns(7)
    valores = [
        ("Usinas NEWAVE", resultado.total_usinas_newave),
        ("Usinas COBRE", resultado.total_usinas_cobre),
        ("Linhas do parquet", resultado.total_linhas_parquet),
        ("Parâmetros JSON", resultado.total_parametros_json),
        ("Coincidências", resultado.total_coincidentes),
        ("Divergências", resultado.total_divergencias),
        ("Não validados", resultado.total_nao_validados),
    ]
    for coluna, (rotulo, valor) in zip(metricas, valores):
        coluna.metric(rotulo, valor)

    st.divider()
    filtro_col1, filtro_col2, filtro_col3 = st.columns([2, 1, 1])
    with filtro_col1:
        busca = st.text_input(
            "Filtrar por código, nome ou hydro_id", key="aba04_busca"
        )
    with filtro_col2:
        filtro_resultado = st.selectbox(
            "Resultado",
            ["Todos", STATUS_COINCIDENTE, STATUS_DIVERGENTE, STATUS_NAO_VALIDADO],
            key="aba04_resultado_filtro",
        )
    with filtro_col3:
        somente_diferencas = st.checkbox(
            "Somente diferenças", key="aba04_so_diferencas"
        )

    abas = st.tabs(
        [
            "Produtividade",
            "Memória de cálculo",
            "Modelos",
            "Fitting window",
            "Volumes de referência",
            "Consistências",
            "Arquivos",
        ]
    )

    def filtrar(df: pd.DataFrame) -> pd.DataFrame:
        return _filtrar_tabela(
            df,
            texto=busca,
            resultado=filtro_resultado,
            somente_divergencias=somente_diferencas,
        )

    with abas[0]:
        tabela = filtrar(resultado.produtividade)
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar comparação da produtividade",
            gerar_csv(tabela),
            "aba_04_produtividade.csv",
            "text/csv",
        )

    with abas[1]:
        tabela = filtrar(resultado.memoria_calculo)
        st.caption(
            "Cada coluna registra um dado original ou uma etapa intermediária "
            "do cálculo. Selecione uma usina com o filtro superior."
        )
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar memória de cálculo",
            gerar_csv(tabela),
            "aba_04_memoria_calculo.csv",
            "text/csv",
        )

    with abas[2]:
        tabela_resumo = filtrar(resultado.modelos)
        tabela_parametros = filtrar(resultado.parametros_modelos)
        st.markdown("**Resumo por usina**")
        st.dataframe(tabela_resumo, use_container_width=True, hide_index=True)
        st.markdown("**Conferência campo a campo do JSON**")
        st.dataframe(tabela_parametros, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar validação do hydro_production_models.json",
            gerar_csv(tabela_parametros),
            "aba_04_modelos_producao.csv",
            "text/csv",
        )

    with abas[3]:
        tabela = filtrar(resultado.fitting_windows)
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar fitting windows",
            gerar_csv(tabela),
            "aba_04_fitting_windows.csv",
            "text/csv",
        )

    with abas[4]:
        tabela = filtrar(resultado.volumes_referencia)
        st.caption(
            "O volume do JSON é absoluto: VMIN + volume útil mensal, limitado "
            "entre VMIN e VMAX."
        )
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar volumes de referência",
            gerar_csv(tabela),
            "aba_04_volumes_referencia.csv",
            "text/csv",
        )

    with abas[5]:
        tabela = filtrar(resultado.inconsistencias)
        if tabela.empty:
            st.success("Nenhuma inconsistência adicional foi encontrada.")
        else:
            st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.markdown("**De-para reconstruído de forma independente**")
        st.dataframe(resultado.depara, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar inconsistências",
            gerar_csv(tabela),
            "aba_04_inconsistencias.csv",
            "text/csv",
        )

    with abas[6]:
        st.dataframe(
            resultado.arquivos_utilizados, use_container_width=True, hide_index=True
        )
