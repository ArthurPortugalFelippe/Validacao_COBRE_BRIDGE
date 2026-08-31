"""Validador de dados convertidos do NEWAVE para o COBRE.

Este arquivo e autocontido. Ao ser executado pelo botao "Run Python File" do
VS Code, verifica as dependencias, inicia o Streamlit e abre a interface no
navegador.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata


# -----------------------------------------------------------------------------
# Inicializacao simples pelo botao triangular do VS Code
# -----------------------------------------------------------------------------

_STREAMLIT_CHILD = "VALIDADOR_CARGA_STREAMLIT_CHILD"
_DEPENDENCIES = {
    "streamlit": "streamlit>=1.38,<2",
    "pandas": "pandas>=2.2,<3.1",
    "pyarrow": "pyarrow>=15,<26",
}


def _prepare_execution() -> None:
    """Instala dependencias ausentes e inicia este arquivo pelo Streamlit."""

    missing = [
        package
        for module, package in _DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("\nPreparando o Validador de Carga pela primeira vez...")
        print("Isso pode demorar alguns minutos. Mantenha a internet conectada.\n")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        except subprocess.CalledProcessError as error:
            print("\nNao foi possivel instalar automaticamente as dependencias.")
            print("Confirme se o Python selecionado no VS Code possui acesso ao pip.")
            print(f"Detalhe tecnico: {error}")
            try:
                input("\nPressione Enter para fechar...")
            except EOFError:
                pass
            raise SystemExit(1) from error

    if os.environ.get(_STREAMLIT_CHILD) == "1":
        return

    # Se o usuario ja usou `streamlit run main.py`, nao cria outro processo.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            return
    except Exception:
        pass

    print("\nAbrindo o Validador de Carga no navegador...")
    environment = os.environ.copy()
    environment[_STREAMLIT_CHILD] = "1"
    # Evita a pergunta de cadastro que o Streamlit mostra na primeira execucao.
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless=false",
        "--server.showEmailPrompt=false",
        "--browser.gatherUsageStats=false",
    ]
    try:
        raise SystemExit(subprocess.call(command, env=environment))
    except KeyboardInterrupt:
        raise SystemExit(0)


_prepare_execution()


# As bibliotecas sao importadas somente depois da preparacao automatica.
from dataclasses import dataclass  # noqa: E402
from io import BytesIO  # noqa: E402
from typing import BinaryIO  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402


# -----------------------------------------------------------------------------
# Leitura dos arquivos NEWAVE e COBRE
# -----------------------------------------------------------------------------

YEAR_ROW = re.compile(r"^\s*(\d{4}|POS)\s+(.*)$", re.IGNORECASE)
CODE_ROW = re.compile(r"^\s*(\d+)\s*$")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d*)?")
REPORT_SYSTEM = re.compile(
    r"SISTEMA\s+(\d+)\s*:\s*([A-Z0-9_.-]+)", re.IGNORECASE
)
REPORT_SUBSYSTEM = re.compile(r"SUBSISTEMA\s*:\s*(.+)$", re.IGNORECASE)

MONTH_NAMES = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Marco",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}


@dataclass(frozen=True)
class ParsedSistema:
    market: pd.DataFrame
    subsystem_names: dict[int, str]


@dataclass(frozen=True)
class ComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class FactorComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class NonControllableComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class NonControllableFactorComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class NetLoadComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


def _read_text(source: str | Path | bytes | BinaryIO) -> str:
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source.read()
        if isinstance(raw, str):
            return raw

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Nao foi possivel identificar a codificacao do arquivo de texto.")


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.upper().split())


def _month_numbers(value_count: int) -> list[int]:
    if not 1 <= value_count <= 12:
        raise ValueError(
            f"Esperava-se de 1 a 12 valores mensais; encontrados {value_count}."
        )
    # No primeiro ano parcial do arquivo de entrada, os valores ficam a direita.
    return list(range(13 - value_count, 13))


def _monthly_rows(
    code: int,
    year_token: str,
    values_text: str,
    extra: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    values = [float(token) for token in NUMBER.findall(values_text)]
    months = _month_numbers(len(values))
    period_type = "POS" if year_token.upper() == "POS" else "ESTUDO"
    year = None if period_type == "POS" else int(year_token)
    rows: list[dict[str, object]] = []
    for month, value in zip(months, values, strict=True):
        row: dict[str, object] = {
            "codigo_submercado": int(code),
            "tipo_periodo": period_type,
            "ano": year,
            "mes": int(month),
            "valor_mw": float(value),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _find_block(lines: list[str], title: str, start_at: int = 0) -> int:
    normalized_title = _normalized(title)
    for index in range(start_at, len(lines)):
        if normalized_title in _normalized(lines[index]):
            return index
    raise ValueError(f"A secao '{title}' nao foi encontrada.")


def _subsystem_names_sistema(lines: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in lines:
        match = re.match(r"^\s*(\d+)\s+([A-Za-z][A-Za-z0-9_.-]*)\s+", line)
        if match:
            code = int(match.group(1))
            name = match.group(2).strip()
            if 1 <= code <= 999 and not name.upper().startswith("XXX"):
                names.setdefault(code, name)
    return names


def parse_sistema(source: str | Path | bytes | BinaryIO) -> ParsedSistema:
    lines = _read_text(source).splitlines()
    start = _find_block(lines, "MERCADO DE ENERGIA TOTAL") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].strip() == "999"),
        None,
    )
    if end is None:
        raise ValueError("O final do bloco de mercado (999) nao foi encontrado.")

    rows: list[dict[str, object]] = []
    current_code: int | None = None
    for line in lines[start:end]:
        code_match = CODE_ROW.match(line)
        if code_match:
            current_code = int(code_match.group(1))
            continue
        year_match = YEAR_ROW.match(line)
        if year_match and current_code is not None:
            rows.extend(
                _monthly_rows(current_code, year_match.group(1), year_match.group(2))
            )

    market = pd.DataFrame(rows)
    if market.empty:
        raise ValueError("Nenhum mercado foi lido no SISTEMA.DAT.")
    if market.duplicated(
        ["codigo_submercado", "tipo_periodo", "ano", "mes"]
    ).any():
        raise ValueError("Existem valores de mercado duplicados no SISTEMA.DAT.")
    return ParsedSistema(
        market=market,
        subsystem_names=_subsystem_names_sistema(lines),
    )


def parse_c_adic(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    lines = _read_text(source).splitlines()
    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_name = ""
    current_reason = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "999" or stripped.upper().startswith("XXX"):
            continue

        year_match = YEAR_ROW.match(line)
        if year_match and current_code is not None:
            rows.extend(
                _monthly_rows(
                    current_code,
                    year_match.group(1),
                    year_match.group(2),
                    {
                        "subsistema": current_name,
                        "razao": current_reason or "Carga adicional",
                    },
                )
            )
            continue

        code_field = line[:4].strip()
        if code_field.isdigit():
            current_code = int(code_field)
            current_name = line[5:15].strip()
            current_reason = line[20:].strip()
            if not current_reason:
                parts = re.split(r"\s{2,}", stripped, maxsplit=2)
                if len(parts) == 3:
                    _, current_name, current_reason = parts

    if not rows:
        return pd.DataFrame(
            columns=[
                "codigo_submercado",
                "subsistema",
                "razao",
                "tipo_periodo",
                "ano",
                "mes",
                "valor_mw",
            ]
        )
    return pd.DataFrame(rows)


def _report_system_names(lines: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in lines:
        for match in REPORT_SYSTEM.finditer(_normalized(line)):
            names.setdefault(int(match.group(1)), match.group(2).strip())
    return dict(sorted(names.items()))


def _parse_pmo_section(
    lines: list[str],
    start: int,
    end: int,
    name_to_code: dict[str, int],
    is_additional: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_name = ""

    for line in lines[start:end]:
        subsystem_match = REPORT_SUBSYSTEM.search(_normalized(line))
        if subsystem_match:
            current_name = subsystem_match.group(1).strip()
            key = _normalized(current_name)
            if key not in name_to_code:
                name_to_code[key] = max(name_to_code.values(), default=0) + 1
            current_code = name_to_code[key]
            continue

        year_match = YEAR_ROW.match(line)
        if year_match and current_code is not None:
            extra = None
            if is_additional:
                extra = {
                    "subsistema": current_name,
                    "razao": "Carga adicional agregada no pmo.dat (inclui MMGD)",
                }
            rows.extend(
                _monthly_rows(
                    current_code,
                    year_match.group(1),
                    year_match.group(2),
                    extra,
                )
            )

    result = pd.DataFrame(rows)
    if result.empty:
        section = "carga adicional" if is_additional else "mercado total"
        raise ValueError(f"Nenhum valor foi lido na secao de {section} do pmo.dat.")
    return result


def _remove_pmo_pre_study_zeros(
    market: pd.DataFrame, additional: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    study = market.loc[market["tipo_periodo"] == "ESTUDO"].copy()
    active = (
        study.groupby(["ano", "mes"], as_index=False)["valor_mw"]
        .sum()
        .query("valor_mw != 0")
        .sort_values(["ano", "mes"])
    )
    if active.empty:
        raise ValueError("Nao foi possivel identificar o inicio do estudo no pmo.dat.")
    first_year = int(active.iloc[0]["ano"])
    first_month = int(active.iloc[0]["mes"])
    first_key = first_year * 100 + first_month

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        keys = frame["ano"].fillna(0).astype(int) * 100 + frame["mes"].astype(int)
        mask = frame["tipo_periodo"].eq("POS") | keys.ge(first_key)
        return frame.loc[mask].reset_index(drop=True)

    return keep(market), keep(additional)


def parse_pmo(
    source: str | Path | bytes | BinaryIO,
) -> tuple[ParsedSistema, pd.DataFrame]:
    lines = _read_text(source).splitlines()
    names = _report_system_names(lines)
    if not names:
        raise ValueError("Os subsistemas nao foram identificados no pmo.dat.")
    name_to_code = {_normalized(name): code for code, name in names.items()}

    additional_title = _find_block(lines, "DADOS DE CARGA ADICIONAL DE ENERGIA")
    market_title = _find_block(
        lines, "DADOS DE MERCADO TOTAL DE ENERGIA", additional_title + 1
    )
    next_title = _find_block(
        lines, "DADOS DE GERACAO DE PEQUENAS USINAS", market_title + 1
    )

    additional = _parse_pmo_section(
        lines,
        additional_title + 1,
        market_title,
        name_to_code,
        is_additional=True,
    )
    market = _parse_pmo_section(
        lines,
        market_title + 1,
        next_title,
        name_to_code,
        is_additional=False,
    )
    market, additional = _remove_pmo_pre_study_zeros(market, additional)
    return ParsedSistema(market=market, subsystem_names=names), additional


def parse_load_parquet(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(source, bytes):
        parquet_source: object = BytesIO(source)
    elif isinstance(source, (str, Path)):
        parquet_source = source
    else:
        raw = source.read()
        parquet_source = BytesIO(raw)

    frame = pd.read_parquet(parquet_source)
    required = ["bus_id", "stage_id", "mean_mw", "std_mw"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Colunas obrigatorias ausentes no Parquet: " + ", ".join(missing)
        )
    frame = frame[required].copy()
    if frame.duplicated(["bus_id", "stage_id"]).any():
        raise ValueError("O Parquet possui linhas duplicadas para a mesma barra e estagio.")
    frame["bus_id"] = frame["bus_id"].astype("int32")
    frame["stage_id"] = frame["stage_id"].astype("int32")
    frame["mean_mw"] = frame["mean_mw"].astype("float64")
    frame["std_mw"] = frame["std_mw"].astype("float64")
    return frame.sort_values(["bus_id", "stage_id"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Montagem da comparacao
# -----------------------------------------------------------------------------

def infer_bus_mapping(sistema: ParsedSistema, load: pd.DataFrame) -> pd.DataFrame:
    market_codes = set(sistema.market["codigo_submercado"].astype(int))
    codes = sorted(market_codes | set(sistema.subsystem_names))
    bus_ids = sorted(load["bus_id"].astype(int).unique())

    # Mantem eventuais barras sem mercado para comprovar tambem suas cargas zero.
    while len(codes) < len(bus_ids):
        synthetic_code = max(codes, default=0) + 1
        codes.append(synthetic_code)
        sistema.subsystem_names.setdefault(synthetic_code, f"Barra {bus_ids[len(codes)-1]}")

    rows = []
    for position, code in enumerate(codes[: len(bus_ids)]):
        rows.append(
            {
                "codigo_submercado": int(code),
                "subsistema": sistema.subsystem_names.get(code, f"Subsistema {code}"),
                "bus_id": int(bus_ids[position]),
                "tem_mercado": code in market_codes,
            }
        )
    return pd.DataFrame(rows)


def _period_sequence(market: pd.DataFrame, total_stages: int) -> pd.DataFrame:
    study = (
        market.loc[market["tipo_periodo"] == "ESTUDO", ["ano", "mes"]]
        .drop_duplicates()
        .sort_values(["ano", "mes"])
        .reset_index(drop=True)
    )
    if study.empty:
        raise ValueError("Nao existem periodos de estudo nos dados de mercado.")
    if total_stages < len(study):
        raise ValueError(
            "O Parquet possui menos estagios do que os periodos do NEWAVE."
        )

    periods: list[dict[str, object]] = []
    for stage_id, row in study.iterrows():
        periods.append(
            {
                "stage_id": int(stage_id),
                "tipo_periodo": "ESTUDO",
                "ano": int(row["ano"]),
                "mes": int(row["mes"]),
                "ciclo_pos": None,
            }
        )

    year = int(study.iloc[-1]["ano"])
    month = int(study.iloc[-1]["mes"])
    for stage_id in range(len(study), total_stages):
        month += 1
        if month > 12:
            month = 1
            year += 1
        pos_index = stage_id - len(study)
        periods.append(
            {
                "stage_id": int(stage_id),
                "tipo_periodo": "POS",
                "ano": year,
                "mes": month,
                "ciclo_pos": pos_index // 12 + 1,
            }
        )
    return pd.DataFrame(periods)


def _additional_aggregates(additional: pd.DataFrame) -> pd.DataFrame:
    keys = ["codigo_submercado", "tipo_periodo", "ano", "mes"]
    if additional.empty:
        return pd.DataFrame(columns=keys + ["c_adic_mw", "componentes_c_adic"])

    totals = (
        additional.groupby(keys, dropna=False, as_index=False)["valor_mw"]
        .sum()
        .rename(columns={"valor_mw": "c_adic_mw"})
    )
    components = (
        additional.assign(
            componente=additional.apply(
                lambda row: f"{row['razao']}: {float(row['valor_mw']):.2f} MW",
                axis=1,
            )
        )
        .groupby(keys, dropna=False, as_index=False)["componente"]
        .agg(" + ".join)
        .rename(columns={"componente": "componentes_c_adic"})
    )
    return totals.merge(components, on=keys, how="left")


def build_comparison(
    sistema: ParsedSistema,
    additional: pd.DataFrame,
    load: pd.DataFrame,
    mapping: pd.DataFrame,
    tolerance_mw: float,
) -> ComparisonResult:
    if tolerance_mw < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    stages_by_bus = load.groupby("bus_id")["stage_id"].nunique()
    if stages_by_bus.empty or stages_by_bus.nunique() != 1:
        raise ValueError("Todas as barras devem possuir a mesma quantidade de estagios.")
    total_stages = int(stages_by_bus.iloc[0])
    periods = _period_sequence(sistema.market, total_stages)
    additional_sum = _additional_aggregates(additional)

    base = mapping[["codigo_submercado", "subsistema", "bus_id"]].merge(
        periods, how="cross"
    )
    market = sistema.market.rename(columns={"valor_mw": "mercado_total_mw"}).copy()
    market["ano_chave"] = market["ano"].fillna(-1)
    additional_sum["ano_chave"] = additional_sum["ano"].fillna(-1)
    base["ano_chave"] = base["ano"].where(base["tipo_periodo"] == "ESTUDO", -1)
    keys = ["codigo_submercado", "tipo_periodo", "ano_chave", "mes"]

    base = base.merge(
        market[[*keys, "mercado_total_mw"]], on=keys, how="left"
    ).merge(
        additional_sum[[*keys, "c_adic_mw", "componentes_c_adic"]],
        on=keys,
        how="left",
    )
    base["mercado_total_mw"] = base["mercado_total_mw"].fillna(0.0)
    base["c_adic_mw"] = base["c_adic_mw"].fillna(0.0)
    base["componentes_c_adic"] = base["componentes_c_adic"].fillna(
        "Sem carga adicional"
    )
    base["carga_bruta_esperada_mw"] = (
        base["mercado_total_mw"] + base["c_adic_mw"]
    )

    detail = base.merge(load, on=["bus_id", "stage_id"], how="left").rename(
        columns={
            "mean_mw": "carga_bruta_cobre_mw",
            "std_mw": "desvio_padrao_cobre_mw",
        }
    )
    detail["diferenca_mw"] = (
        detail["carga_bruta_cobre_mw"] - detail["carga_bruta_esperada_mw"]
    )
    detail["resultado"] = detail["diferenca_mw"].abs().le(tolerance_mw).map(
        {True: "OK", False: "DIVERGENTE"}
    )
    detail["periodo"] = detail.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if row["tipo_periodo"] == "ESTUDO"
            else f"POS {int(row['ciclo_pos'])} - {MONTH_NAMES[int(row['mes'])]}"
        ),
        axis=1,
    )

    summary = (
        detail.groupby(
            ["bus_id", "codigo_submercado", "subsistema"], as_index=False
        )
        .agg(
            estagios=("stage_id", "size"),
            estagios_ok=("resultado", lambda series: int((series == "OK").sum())),
            divergencias=(
                "resultado", lambda series: int((series != "OK").sum())
            ),
            maior_diferenca_abs_mw=(
                "diferenca_mw", lambda series: float(series.abs().max())
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "bus_id",
        "codigo_submercado",
        "subsistema",
        "stage_id",
        "periodo",
        "tipo_periodo",
        "ano",
        "mes",
        "mercado_total_mw",
        "c_adic_mw",
        "componentes_c_adic",
        "carga_bruta_esperada_mw",
        "carga_bruta_cobre_mw",
        "diferenca_mw",
        "desvio_padrao_cobre_mw",
        "resultado",
    ]
    return ComparisonResult(
        detail=detail[ordered]
        .sort_values(["bus_id", "stage_id"])
        .reset_index(drop=True),
        summary=summary,
    )


# -----------------------------------------------------------------------------
# Fatores de desagregacao da carga por patamar
# -----------------------------------------------------------------------------

def parse_patamar_load_factors(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le somente o bloco CARGA (P.U. DEMANDA MED.) do PATAMAR.DAT."""

    lines = _read_text(source).splitlines()
    start = _find_block(lines, "CARGA(P.U.DEMANDA MED.)") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].strip() == "9999"),
        None,
    )
    if end is None:
        raise ValueError("O final do bloco de fatores de carga (9999) nao foi encontrado.")

    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_year: int | None = None
    current_block: int | None = None

    for line in lines[start:end]:
        stripped = line.strip()
        if CODE_ROW.fullmatch(line):
            current_code = int(stripped)
            current_year = None
            current_block = None
            continue

        year_match = re.match(r"^\s*(\d{4})\s+(.+)$", line)
        if year_match and current_code is not None:
            current_year = int(year_match.group(1))
            current_block = 0
            values_text = year_match.group(2)
        elif (
            current_code is not None
            and current_year is not None
            and current_block is not None
            and re.match(r"^\s+[-+]?\d", line)
        ):
            current_block += 1
            values_text = stripped
        else:
            continue

        values = [float(token) for token in NUMBER.findall(values_text)]
        if len(values) != 12:
            raise ValueError(
                "Esperavam-se 12 fatores mensais para cada patamar do PATAMAR.DAT."
            )
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "codigo_submercado": current_code,
                    "ano_fonte": current_year,
                    "mes": month,
                    "block_id": current_block,
                    "fator_newave": value,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nenhum fator de carga foi lido no PATAMAR.DAT.")
    keys = ["codigo_submercado", "ano_fonte", "mes", "block_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Existem fatores de carga duplicados no PATAMAR.DAT.")

    blocks = sorted(frame["block_id"].unique())
    if blocks != list(range(len(blocks))):
        raise ValueError("A numeracao dos patamares no PATAMAR.DAT nao e continua.")
    return frame.sort_values(keys).reset_index(drop=True)


def parse_load_factors_json(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Transforma o load_factors.json em uma tabela com uma linha por fator."""

    try:
        payload = json.loads(_read_text(source))
    except json.JSONDecodeError as error:
        raise ValueError(f"O load_factors.json nao e um JSON valido: {error}") from error

    records = payload.get("load_factors") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("A lista 'load_factors' nao foi encontrada no JSON.")

    rows: list[dict[str, object]] = []
    for record in records:
        try:
            bus_id = int(record["bus_id"])
            stage_id = int(record["stage_id"])
            block_factors = record["block_factors"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Existe um registro incompleto no load_factors.json.") from error
        if not isinstance(block_factors, list):
            raise ValueError("O campo 'block_factors' precisa ser uma lista.")
        for block in block_factors:
            try:
                rows.append(
                    {
                        "bus_id": bus_id,
                        "stage_id": stage_id,
                        "block_id": int(block["block_id"]),
                        "fator_cobre": float(block["factor"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Existe um fator incompleto no load_factors.json."
                ) from error

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nenhum fator foi encontrado no load_factors.json.")
    keys = ["bus_id", "stage_id", "block_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Existem fatores duplicados no load_factors.json.")
    return frame.sort_values(keys).reset_index(drop=True)


def _load_factor_mapping(sistema: ParsedSistema) -> pd.DataFrame:
    """Associa os subsistemas com mercado aos bus_id sequenciais do COBRE."""

    codes = sorted(sistema.market["codigo_submercado"].astype(int).unique())
    return pd.DataFrame(
        [
            {
                "codigo_submercado": code,
                "subsistema": sistema.subsystem_names.get(code, f"Subsistema {code}"),
                "bus_id": position,
            }
            for position, code in enumerate(codes)
        ]
    )


def build_factor_comparison(
    sistema: ParsedSistema,
    patamar: pd.DataFrame,
    cobre_factors: pd.DataFrame,
    tolerance: float,
) -> FactorComparisonResult:
    """Compara os fatores do PATAMAR.DAT com os fatores do COBRE."""

    if tolerance < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    total_stages = int(cobre_factors["stage_id"].max()) + 1
    periods = _period_sequence(sistema.market, total_stages)
    mapping = _load_factor_mapping(sistema)
    last_source_year = int(patamar["ano_fonte"].max())

    expected = mapping.merge(periods, how="cross")
    expected["ano_fonte"] = expected["ano"].where(
        expected["tipo_periodo"] == "ESTUDO", last_source_year
    )
    expected = expected.merge(
        patamar,
        on=["codigo_submercado", "ano_fonte", "mes"],
        how="left",
    )

    keys = ["bus_id", "stage_id", "block_id"]
    detail = expected.merge(cobre_factors, on=keys, how="outer", indicator=True)
    detail["subsistema"] = detail["subsistema"].fillna(
        detail["bus_id"].map(lambda value: f"Barra {int(value)}")
    )
    detail["diferenca"] = detail["fator_cobre"] - detail["fator_newave"]
    detail["resultado"] = "DIVERGENTE"
    detail.loc[detail["_merge"] == "left_only", "resultado"] = "AUSENTE NO COBRE"
    detail.loc[detail["_merge"] == "right_only", "resultado"] = "AUSENTE NO NEWAVE"
    comparable = detail["_merge"] == "both"
    detail.loc[
        comparable & detail["diferenca"].abs().le(tolerance), "resultado"
    ] = "OK"
    detail["patamar"] = detail["block_id"] + 1
    detail["periodo"] = detail.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if pd.notna(row.get("mes")) and row.get("tipo_periodo") == "ESTUDO"
            else (
                f"POS {int(row['ciclo_pos'])} - {MONTH_NAMES[int(row['mes'])]}"
                if pd.notna(row.get("mes"))
                else "Período não identificado"
            )
        ),
        axis=1,
    )

    summary = (
        detail.groupby(["bus_id", "subsistema"], dropna=False, as_index=False)
        .agg(
            fatores=("resultado", "size"),
            fatores_ok=("resultado", lambda values: int((values == "OK").sum())),
            divergencias=(
                "resultado", lambda values: int((values != "OK").sum())
            ),
            maior_diferenca_abs=(
                "diferenca",
                lambda values: float(values.abs().max())
                if values.notna().any()
                else float("nan"),
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "bus_id",
        "codigo_submercado",
        "subsistema",
        "stage_id",
        "periodo",
        "tipo_periodo",
        "ano",
        "mes",
        "ano_fonte",
        "patamar",
        "block_id",
        "fator_newave",
        "fator_cobre",
        "diferenca",
        "resultado",
    ]
    return FactorComparisonResult(
        detail=detail[ordered]
        .sort_values(["bus_id", "stage_id", "block_id"])
        .reset_index(drop=True),
        summary=summary,
    )


# =============================================================================
# ABA 3 - GERACAO NAO CONTROLADA MEDIA
# Palavra-chave para localizar esta logica: ABA 3 GERACAO NAO CONTROLADA
# =============================================================================

def parse_sistema_non_controllable(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le a geracao media das usinas nao simuladas no SISTEMA.DAT."""

    lines = _read_text(source).splitlines()
    start = _find_block(lines, "GERACAO DE USINAS NAO SIMULADAS") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].strip() == "999"),
        None,
    )
    if end is None:
        raise ValueError(
            "O final do bloco de geracao de usinas nao simuladas nao foi encontrado."
        )

    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_source_index: int | None = None
    current_source_label = ""
    source_row = re.compile(r"^\s*(\d{1,3})\s+(\d+)\s*(.*?)\s*$")

    for line in lines[start:end]:
        year_match = YEAR_ROW.match(line)
        if (
            year_match
            and current_code is not None
            and current_source_index is not None
        ):
            values = [float(token) for token in NUMBER.findall(year_match.group(2))]
            months = _month_numbers(len(values))
            period_type = (
                "POS" if year_match.group(1).upper() == "POS" else "ESTUDO"
            )
            year = None if period_type == "POS" else int(year_match.group(1))
            for month, value in zip(months, values, strict=True):
                rows.append(
                    {
                        "codigo_submercado": current_code,
                        "indice_fonte": current_source_index,
                        "tipo_fonte": current_source_label,
                        "tipo_periodo": period_type,
                        "ano": year,
                        "mes": int(month),
                        "geracao_newave_mw": float(value),
                    }
                )
            continue

        match = source_row.match(line)
        if match and int(match.group(1)) < 1000:
            current_code = int(match.group(1))
            current_source_index = int(match.group(2))
            current_source_label = match.group(3).strip()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "Nenhuma geracao de usina nao simulada foi lida no SISTEMA.DAT."
        )
    keys = [
        "codigo_submercado",
        "indice_fonte",
        "tipo_periodo",
        "ano",
        "mes",
    ]
    if frame.duplicated(keys).any():
        raise ValueError(
            "Existem valores duplicados no bloco de usinas nao simuladas."
        )
    return frame.sort_values(keys, na_position="last").reset_index(drop=True)


def parse_non_controllable_sources_json(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le o cadastro COBRE sem usar o nome da fonte como chave."""

    try:
        payload = json.loads(_read_text(source))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"O non_controllable_sources.json nao e um JSON valido: {error}"
        ) from error

    records = (
        payload.get("non_controllable_sources") if isinstance(payload, dict) else None
    )
    if not isinstance(records, list):
        raise ValueError(
            "A lista 'non_controllable_sources' nao foi encontrada no JSON."
        )

    rows: list[dict[str, object]] = []
    for record in records:
        try:
            rows.append(
                {
                    "ncs_id": int(record["id"]),
                    "bus_id_cobre": int(record["bus_id"]),
                    "max_generation_mw": float(record["max_generation_mw"]),
                    # O nome e apenas informativo e pode estar vazio ou ausente.
                    "nome_fonte_cobre": str(record.get("name") or "").strip(),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Existe uma fonte incompleta no non_controllable_sources.json."
            ) from error

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "Nenhuma fonte foi encontrada no non_controllable_sources.json."
        )
    if frame["ncs_id"].duplicated().any():
        raise ValueError("Existem identificadores de fonte duplicados no JSON.")
    if (frame["max_generation_mw"] < 0).any():
        raise ValueError("O JSON possui potencia maxima negativa.")
    return frame.sort_values("ncs_id").reset_index(drop=True)


def parse_non_controllable_stats_parquet(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le a disponibilidade media por fonte e estagio no COBRE."""

    if isinstance(source, bytes):
        parquet_source: object = BytesIO(source)
    elif isinstance(source, (str, Path)):
        parquet_source = source
    else:
        parquet_source = BytesIO(source.read())

    frame = pd.read_parquet(parquet_source)
    required = ["ncs_id", "stage_id", "mean", "std"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Colunas obrigatorias ausentes no non_controllable_stats.parquet: "
            + ", ".join(missing)
        )
    frame = frame[required].copy()
    if frame.duplicated(["ncs_id", "stage_id"]).any():
        raise ValueError(
            "O Parquet possui linhas duplicadas para a mesma fonte e estagio."
        )
    frame["ncs_id"] = frame["ncs_id"].astype("int32")
    frame["stage_id"] = frame["stage_id"].astype("int32")
    frame["mean"] = frame["mean"].astype("float64")
    frame["std"] = frame["std"].astype("float64")
    return frame.sort_values(["ncs_id", "stage_id"]).reset_index(drop=True)


def _non_controllable_source_mapping(
    sistema: ParsedSistema,
    generation: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduz a ordem tecnica (subsistema, indice do bloco) usada no conversor."""

    bus_mapping = _load_factor_mapping(sistema)[
        ["codigo_submercado", "subsistema", "bus_id"]
    ]
    groups = (
        generation[["codigo_submercado", "indice_fonte", "tipo_fonte"]]
        .drop_duplicates(["codigo_submercado", "indice_fonte"])
        .merge(bus_mapping, on="codigo_submercado", how="inner")
        .sort_values(["codigo_submercado", "indice_fonte"])
        .reset_index(drop=True)
    )
    # Esta e a chave: o nome nao participa da atribuicao do ncs_id.
    groups.insert(0, "ncs_id", range(len(groups)))
    return groups


def build_non_controllable_comparison(
    sistema: ParsedSistema,
    generation: pd.DataFrame,
    sources: pd.DataFrame,
    stats: pd.DataFrame,
    tolerance_mw: float,
) -> NonControllableComparisonResult:
    """Compara a geracao media NEWAVE com max_generation_mw vezes mean."""

    if tolerance_mw < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    if stats.empty:
        raise ValueError("O non_controllable_stats.parquet esta vazio.")

    total_stages = int(stats["stage_id"].max()) + 1
    periods = _period_sequence(sistema.market, total_stages)
    mapping = _non_controllable_source_mapping(sistema, generation)

    source_keys = ["codigo_submercado", "indice_fonte", "mes"]
    study_values = generation[generation["tipo_periodo"] == "ESTUDO"].copy()
    exact_lookup = {
        (int(row.codigo_submercado), int(row.indice_fonte), int(row.ano), int(row.mes)):
        float(row.geracao_newave_mw)
        for row in study_values.itertuples()
    }

    fallback_values = generation.copy()
    fallback_values["prioridade_ano"] = fallback_values["ano"].fillna(9999)
    fallback_values = (
        fallback_values.sort_values("prioridade_ano")
        .drop_duplicates(source_keys, keep="last")
    )
    fallback_lookup = {
        (int(row.codigo_submercado), int(row.indice_fonte), int(row.mes)):
        float(row.geracao_newave_mw)
        for row in fallback_values.itertuples()
    }

    expected = mapping.merge(periods, how="cross")

    def expected_mw(row: pd.Series) -> float | None:
        group_month = (
            int(row["codigo_submercado"]),
            int(row["indice_fonte"]),
            int(row["mes"]),
        )
        if row["tipo_periodo"] == "ESTUDO":
            exact_key = (
                int(row["codigo_submercado"]),
                int(row["indice_fonte"]),
                int(row["ano"]),
                int(row["mes"]),
            )
            return exact_lookup.get(exact_key, fallback_lookup.get(group_month))
        return fallback_lookup.get(group_month)

    expected["geracao_newave_mw"] = expected.apply(expected_mw, axis=1)

    cobre = stats.merge(sources, on="ncs_id", how="left")
    cobre["geracao_cobre_mw"] = cobre["max_generation_mw"] * cobre["mean"]

    detail = expected.merge(
        cobre,
        on=["ncs_id", "stage_id"],
        how="outer",
        indicator=True,
    )
    detail["diferenca_mw"] = (
        detail["geracao_cobre_mw"] - detail["geracao_newave_mw"]
    )
    detail["resultado"] = "DIVERGENTE"
    detail.loc[detail["_merge"] == "left_only", "resultado"] = "AUSENTE NO COBRE"
    detail.loc[detail["_merge"] == "right_only", "resultado"] = "AUSENTE NO NEWAVE"

    comparable = detail["_merge"] == "both"
    missing_registry = comparable & detail["max_generation_mw"].isna()
    detail.loc[missing_registry, "resultado"] = "CADASTRO COBRE AUSENTE"
    bus_matches = detail["bus_id_cobre"].eq(detail["bus_id"])
    detail.loc[
        comparable & ~missing_registry & ~bus_matches,
        "resultado",
    ] = "BARRA DIVERGENTE"
    detail.loc[
        comparable
        & ~missing_registry
        & bus_matches
        & detail["geracao_newave_mw"].notna()
        & detail["geracao_cobre_mw"].notna()
        & detail["diferenca_mw"].abs().le(tolerance_mw),
        "resultado",
    ] = "OK"

    detail["subsistema"] = detail["subsistema"].fillna(
        detail["bus_id_cobre"].map(
            lambda value: f"Barra {int(value)}" if pd.notna(value) else "Nao identificado"
        )
    )
    detail["tipo_fonte"] = detail["tipo_fonte"].fillna("Nao identificado")
    detail["periodo"] = detail.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if pd.notna(row.get("mes")) and row.get("tipo_periodo") == "ESTUDO"
            else (
                f"POS {int(row['ciclo_pos'])} - {MONTH_NAMES[int(row['mes'])]}"
                if pd.notna(row.get("mes"))
                else "Periodo nao identificado"
            )
        ),
        axis=1,
    )

    summary = (
        detail.groupby(
            ["bus_id", "codigo_submercado", "subsistema", "indice_fonte", "tipo_fonte"],
            dropna=False,
            as_index=False,
        )
        .agg(
            estagios=("resultado", "size"),
            estagios_ok=("resultado", lambda values: int((values == "OK").sum())),
            divergencias=(
                "resultado", lambda values: int((values != "OK").sum())
            ),
            maior_diferenca_abs_mw=(
                "diferenca_mw",
                lambda values: float(values.abs().max())
                if values.notna().any()
                else float("nan"),
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "ncs_id",
        "bus_id",
        "bus_id_cobre",
        "codigo_submercado",
        "subsistema",
        "indice_fonte",
        "tipo_fonte",
        "nome_fonte_cobre",
        "stage_id",
        "periodo",
        "tipo_periodo",
        "ano",
        "mes",
        "max_generation_mw",
        "mean",
        "std",
        "geracao_newave_mw",
        "geracao_cobre_mw",
        "diferenca_mw",
        "resultado",
    ]
    return NonControllableComparisonResult(
        detail=detail[ordered]
        .sort_values(["ncs_id", "stage_id"])
        .reset_index(drop=True),
        summary=summary,
    )


# =============================================================================
# ABA 4 - FATORES DA GERACAO NAO CONTROLADA POR PATAMAR
# Palavra-chave para localizar esta logica: ABA 4 FATORES GERACAO NAO CONTROLADA
# =============================================================================

def parse_patamar_non_controllable_factors(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le os fatores das usinas nao simuladas no PATAMAR.DAT."""

    lines = _read_text(source).splitlines()
    start = _find_block(
        lines, "BLOCO DE USINAS NAO SIMULADAS (P.U. MONTANTE MED.)"
    ) + 1

    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_source_index: int | None = None
    current_year: int | None = None
    current_block: int | None = None
    source_row = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

    for line in lines[start:]:
        source_match = source_row.fullmatch(line)
        if source_match:
            current_code = int(source_match.group(1))
            current_source_index = int(source_match.group(2))
            current_year = None
            current_block = None
            continue

        year_match = re.match(r"^\s*(\d{4})\s+(.+)$", line)
        if (
            year_match
            and current_code is not None
            and current_source_index is not None
        ):
            current_year = int(year_match.group(1))
            current_block = 0
            values_text = year_match.group(2)
        elif (
            current_code is not None
            and current_source_index is not None
            and current_year is not None
            and current_block is not None
            and re.match(r"^\s+[-+]?\d", line)
        ):
            current_block += 1
            values_text = line.strip()
        else:
            continue

        values = [float(token) for token in NUMBER.findall(values_text)]
        if len(values) != 12:
            raise ValueError(
                "Esperavam-se 12 fatores mensais para cada patamar das "
                "usinas nao simuladas no PATAMAR.DAT."
            )
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "codigo_submercado": current_code,
                    "indice_fonte": current_source_index,
                    "ano_fonte": current_year,
                    "mes": month,
                    "block_id": current_block,
                    "fator_newave": value,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "Nenhum fator de usina nao simulada foi lido no PATAMAR.DAT."
        )
    keys = [
        "codigo_submercado",
        "indice_fonte",
        "ano_fonte",
        "mes",
        "block_id",
    ]
    if frame.duplicated(keys).any():
        raise ValueError(
            "Existem fatores duplicados para a mesma fonte no PATAMAR.DAT."
        )
    blocks = sorted(int(value) for value in frame["block_id"].unique())
    if blocks != list(range(len(blocks))):
        raise ValueError("A numeracao dos patamares no PATAMAR.DAT nao e continua.")
    return frame.sort_values(keys).reset_index(drop=True)


def parse_non_controllable_factors_json(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Transforma o JSON COBRE em uma linha por fonte, estagio e patamar."""

    try:
        payload = json.loads(_read_text(source))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"O non_controllable_factors.json nao e um JSON valido: {error}"
        ) from error

    records = (
        payload.get("non_controllable_factors")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(records, list):
        raise ValueError(
            "A lista 'non_controllable_factors' nao foi encontrada no JSON."
        )

    rows: list[dict[str, object]] = []
    for record in records:
        try:
            ncs_id = int(record["ncs_id"])
            stage_id = int(record["stage_id"])
            block_factors = record["block_factors"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Existe um registro incompleto no non_controllable_factors.json."
            ) from error
        if not isinstance(block_factors, list):
            raise ValueError("O campo 'block_factors' precisa ser uma lista.")
        for block in block_factors:
            try:
                rows.append(
                    {
                        "ncs_id": ncs_id,
                        "stage_id": stage_id,
                        "block_id": int(block["block_id"]),
                        "fator_cobre": float(block["factor"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Existe um fator incompleto no non_controllable_factors.json."
                ) from error

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "Nenhum fator foi encontrado no non_controllable_factors.json."
        )
    keys = ["ncs_id", "stage_id", "block_id"]
    if frame.duplicated(keys).any():
        raise ValueError(
            "Existem fatores duplicados no non_controllable_factors.json."
        )
    return frame.sort_values(keys).reset_index(drop=True)


def build_non_controllable_factor_comparison(
    sistema: ParsedSistema,
    generation: pd.DataFrame,
    patamar: pd.DataFrame,
    cobre_factors: pd.DataFrame,
    tolerance: float,
) -> NonControllableFactorComparisonResult:
    """Compara os fatores por fonte e patamar, incluindo o piso do COBRE."""

    if tolerance < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    if cobre_factors.empty:
        raise ValueError("O non_controllable_factors.json esta vazio.")

    total_stages = int(cobre_factors["stage_id"].max()) + 1
    periods = _period_sequence(sistema.market, total_stages)
    mapping = _non_controllable_source_mapping(sistema, generation)
    last_source_year = int(patamar["ano_fonte"].max())

    expected = mapping.merge(periods, how="cross")
    expected["ano_fonte"] = expected["ano"].where(
        expected["tipo_periodo"] == "ESTUDO", last_source_year
    )
    expected = expected.merge(
        patamar,
        on=["codigo_submercado", "indice_fonte", "ano_fonte", "mes"],
        how="left",
    )

    keys = ["ncs_id", "stage_id", "block_id"]
    detail = expected.merge(cobre_factors, on=keys, how="outer", indicator=True)

    # Completa informacoes de linhas que porventura existam somente no COBRE.
    for column in [
        "codigo_submercado",
        "subsistema",
        "indice_fonte",
        "tipo_fonte",
        "bus_id",
    ]:
        lookup = mapping.set_index("ncs_id")[column]
        detail[column] = detail[column].fillna(detail["ncs_id"].map(lookup))
    for column in ["tipo_periodo", "ano", "mes", "ciclo_pos"]:
        lookup = periods.set_index("stage_id")[column]
        detail[column] = detail[column].fillna(detail["stage_id"].map(lookup))

    detail["diferenca"] = detail["fator_cobre"] - detail["fator_newave"]
    detail["resultado"] = "DIVERGENTE"
    detail.loc[detail["_merge"] == "left_only", "resultado"] = "AUSENTE NO COBRE"
    detail.loc[detail["_merge"] == "right_only", "resultado"] = "AUSENTE NO NEWAVE"

    comparable = detail["_merge"] == "both"
    within_tolerance = detail["diferenca"].abs().le(tolerance)
    detail.loc[comparable & within_tolerance, "resultado"] = "OK"

    # O esquema do COBRE exige fator positivo. Por isso o conversor troca
    # somente o zero exato do NEWAVE pelo piso tecnico de 0,000001.
    minimum_adjustment = (
        comparable
        & detail["fator_newave"].eq(0.0)
        & detail["fator_cobre"].sub(0.000001).abs().le(1e-12)
    )
    detail.loc[
        minimum_adjustment & within_tolerance, "resultado"
    ] = "OK — AJUSTE MÍNIMO COBRE"

    detail["patamar"] = detail["block_id"] + 1
    detail["periodo"] = detail.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if pd.notna(row.get("mes")) and row.get("tipo_periodo") == "ESTUDO"
            else (
                f"POS {int(row['ciclo_pos'])} - {MONTH_NAMES[int(row['mes'])]}"
                if pd.notna(row.get("mes")) and pd.notna(row.get("ciclo_pos"))
                else "Periodo nao identificado"
            )
        ),
        axis=1,
    )
    detail["subsistema"] = detail["subsistema"].fillna("Nao identificado")
    detail["tipo_fonte"] = detail["tipo_fonte"].fillna("Nao identificado")

    approved = detail["resultado"].str.startswith("OK")
    adjusted = detail["resultado"].eq("OK — AJUSTE MÍNIMO COBRE")
    detail_for_summary = detail.assign(_aprovado=approved, _ajuste=adjusted)
    summary = (
        detail_for_summary.groupby(
            [
                "ncs_id",
                "codigo_submercado",
                "subsistema",
                "indice_fonte",
                "tipo_fonte",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            fatores=("resultado", "size"),
            fatores_corretos=("_aprovado", "sum"),
            ajustes_minimos=("_ajuste", "sum"),
            divergencias=("_aprovado", lambda values: int((~values).sum())),
            maior_diferenca_abs=(
                "diferenca",
                lambda values: float(values.abs().max())
                if values.notna().any()
                else float("nan"),
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "ncs_id",
        "bus_id",
        "codigo_submercado",
        "subsistema",
        "indice_fonte",
        "tipo_fonte",
        "stage_id",
        "periodo",
        "tipo_periodo",
        "ano",
        "mes",
        "ano_fonte",
        "patamar",
        "block_id",
        "fator_newave",
        "fator_cobre",
        "diferenca",
        "resultado",
    ]
    return NonControllableFactorComparisonResult(
        detail=detail[ordered]
        .sort_values(["ncs_id", "stage_id", "block_id"])
        .reset_index(drop=True),
        summary=summary,
    )


# =============================================================================
# ABA 5 - CARGA LIQUIDA POR SUBSISTEMA E PATAMAR
# Palavra-chave para localizar esta logica: ABA 5 CARGA LIQUIDA
# =============================================================================

def parse_buses_json(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le o cadastro de barras do COBRE."""

    try:
        payload = json.loads(_read_text(source))
    except json.JSONDecodeError as error:
        raise ValueError(f"O buses.json nao e um JSON valido: {error}") from error

    records = payload.get("buses") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("A lista 'buses' nao foi encontrada no buses.json.")

    rows: list[dict[str, object]] = []
    for record in records:
        try:
            rows.append(
                {
                    "bus_id": int(record["id"]),
                    "nome_barra_cobre": str(record.get("name") or "").strip(),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Existe uma barra incompleta no buses.json.") from error

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nenhuma barra foi encontrada no buses.json.")
    if frame["bus_id"].duplicated().any():
        raise ValueError("Existem identificadores de barra duplicados no buses.json.")
    return frame.sort_values("bus_id").reset_index(drop=True)


def parse_pmo_net_load(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le o mercado liquido por subsistema e patamar no pmo.dat."""

    lines = _read_text(source).splitlines()
    system_names = _report_system_names(lines)
    if not system_names:
        raise ValueError("Os subsistemas nao foram identificados no pmo.dat.")
    name_to_code = {
        _normalized(name): int(code) for code, name in system_names.items()
    }

    start = _find_block(lines, "DADOS DE MERCADO LIQUIDO DE ENERGIA") + 1
    end = _find_block(lines, "ASSOCIACAO ENTRE REEs E SUBSISTEMAS", start)

    rows: list[dict[str, object]] = []
    current_code: int | None = None
    current_name = ""
    current_block: int | None = None

    for line in lines[start:end]:
        subsystem_match = REPORT_SUBSYSTEM.search(line)
        if subsystem_match:
            current_name = subsystem_match.group(1).strip()
            current_code = name_to_code.get(_normalized(current_name))
            if current_code is None:
                raise ValueError(
                    f"O subsistema '{current_name}' do mercado liquido nao "
                    "foi associado a um codigo no pmo.dat."
                )
            current_block = None
            continue

        block_match = re.match(r"^\s*PATAMAR:\s*(\d+)\s*$", line, re.IGNORECASE)
        if block_match:
            current_block = int(block_match.group(1)) - 1
            continue

        year_match = YEAR_ROW.match(line)
        if (
            year_match
            and current_code is not None
            and current_block is not None
        ):
            rows.extend(
                _monthly_rows(
                    current_code,
                    year_match.group(1),
                    year_match.group(2),
                    {
                        "subsistema": current_name,
                        "block_id": current_block,
                        "patamar": current_block + 1,
                    },
                )
            )

    frame = pd.DataFrame(rows).rename(
        columns={"valor_mw": "carga_liquida_pmo_mw"}
    )
    if frame.empty:
        raise ValueError(
            "Nenhum valor foi lido no bloco de mercado liquido do pmo.dat."
        )
    keys = [
        "codigo_submercado",
        "tipo_periodo",
        "ano",
        "mes",
        "block_id",
    ]
    if frame.duplicated(keys).any():
        raise ValueError("Existem valores duplicados no mercado liquido do pmo.dat.")
    return frame.sort_values(keys, na_position="last").reset_index(drop=True)


def build_net_load_comparison(
    sistema: ParsedSistema,
    pmo_net_load: pd.DataFrame,
    buses: pd.DataFrame,
    load: pd.DataFrame,
    load_factors: pd.DataFrame,
    sources: pd.DataFrame,
    stats: pd.DataFrame,
    non_controllable_factors: pd.DataFrame,
    tolerance_mw: float,
) -> NetLoadComparisonResult:
    """Reconstroi a carga liquida do COBRE e compara com o pmo.dat."""

    if tolerance_mw < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    if load_factors.empty or non_controllable_factors.empty:
        raise ValueError("Os arquivos de fatores do COBRE nao podem estar vazios.")

    source_ids = set(int(value) for value in sources["ncs_id"].unique())
    stats_ids = set(int(value) for value in stats["ncs_id"].unique())
    factor_ids = set(
        int(value) for value in non_controllable_factors["ncs_id"].unique()
    )
    if source_ids != stats_ids or source_ids != factor_ids:
        raise ValueError(
            "Os ncs_id nao sao os mesmos nos tres arquivos da geracao nao "
            "controlada."
        )

    load_blocks = sorted(int(value) for value in load_factors["block_id"].unique())
    ncs_blocks = sorted(
        int(value) for value in non_controllable_factors["block_id"].unique()
    )
    if load_blocks != ncs_blocks:
        raise ValueError(
            "Os block_id dos fatores de carga e da geracao nao controlada "
            "nao coincidem."
        )
    if load_blocks != list(range(len(load_blocks))):
        raise ValueError("A numeracao dos patamares do COBRE nao e continua.")

    total_stages = max(
        int(load["stage_id"].max()),
        int(load_factors["stage_id"].max()),
        int(stats["stage_id"].max()),
        int(non_controllable_factors["stage_id"].max()),
    ) + 1
    periods = _period_sequence(sistema.market, total_stages)
    mapping = _load_factor_mapping(sistema)

    missing_buses = set(int(value) for value in mapping["bus_id"]) - set(
        int(value) for value in buses["bus_id"]
    )
    if missing_buses:
        raise ValueError(
            "O buses.json nao possui as barras esperadas: "
            + ", ".join(str(value) for value in sorted(missing_buses))
        )

    blocks = pd.DataFrame(
        {"block_id": load_blocks, "patamar": [value + 1 for value in load_blocks]}
    )
    expected = mapping.merge(periods, how="cross").merge(blocks, how="cross")
    expected = expected.merge(buses, on="bus_id", how="left")

    # Mercado liquido do PMO: o perfil POS e repetido em todos os ciclos POS.
    pmo = pmo_net_load.copy()
    pmo["ano_chave"] = pmo["ano"].fillna(-1)
    expected["ano_chave"] = expected["ano"].where(
        expected["tipo_periodo"] == "ESTUDO", -1
    )
    pmo_keys = [
        "codigo_submercado",
        "tipo_periodo",
        "ano_chave",
        "mes",
        "block_id",
    ]
    expected = expected.merge(
        pmo[[*pmo_keys, "carga_liquida_pmo_mw"]],
        on=pmo_keys,
        how="left",
    )

    carga = load.rename(columns={"std_mw": "desvio_carga_mw"}).merge(
        load_factors.rename(columns={"fator_cobre": "fator_carga"}),
        on=["bus_id", "stage_id"],
        how="outer",
    )
    carga["carga_bruta_patamar_mw"] = carga["mean_mw"] * carga["fator_carga"]
    expected = expected.merge(
        carga[
            [
                "bus_id",
                "stage_id",
                "block_id",
                "mean_mw",
                "desvio_carga_mw",
                "fator_carga",
                "carga_bruta_patamar_mw",
            ]
        ],
        on=["bus_id", "stage_id", "block_id"],
        how="left",
    )

    cadastro_fontes = sources.rename(columns={"bus_id_cobre": "bus_id"}).copy()
    source_universe = cadastro_fontes.merge(periods, how="cross").merge(
        blocks, how="cross"
    )
    source_universe = source_universe.merge(
        stats,
        on=["ncs_id", "stage_id"],
        how="left",
    ).merge(
        non_controllable_factors.rename(
            columns={"fator_cobre": "fator_geracao_nao_controlada"}
        ),
        on=["ncs_id", "stage_id", "block_id"],
        how="left",
    )
    source_universe["geracao_fonte_patamar_mw"] = (
        source_universe["max_generation_mw"]
        * source_universe["mean"]
        * source_universe["fator_geracao_nao_controlada"]
    )
    source_universe["dados_fonte_completos"] = source_universe[
        ["max_generation_mw", "mean", "fator_geracao_nao_controlada"]
    ].notna().all(axis=1)

    ncs_by_bus = (
        source_universe.groupby(
            ["bus_id", "stage_id", "block_id"], as_index=False
        )
        .agg(
            geracao_nao_simulada_patamar_mw=(
                "geracao_fonte_patamar_mw",
                lambda values: values.sum(min_count=1),
            ),
            quantidade_fontes=("ncs_id", "nunique"),
            dados_ncs_completos=("dados_fonte_completos", "all"),
        )
    )
    expected = expected.merge(
        ncs_by_bus,
        on=["bus_id", "stage_id", "block_id"],
        how="left",
    )

    buses_with_sources = set(int(value) for value in cadastro_fontes["bus_id"])
    without_sources = ~expected["bus_id"].isin(buses_with_sources)
    expected.loc[without_sources, "geracao_nao_simulada_patamar_mw"] = 0.0
    expected.loc[without_sources, "quantidade_fontes"] = 0
    expected.loc[without_sources, "dados_ncs_completos"] = True

    expected["carga_liquida_cobre_calculada_mw"] = (
        expected["carga_bruta_patamar_mw"]
        - expected["geracao_nao_simulada_patamar_mw"]
    )
    expected["diferenca_mw"] = (
        expected["carga_liquida_cobre_calculada_mw"]
        - expected["carga_liquida_pmo_mw"]
    )

    expected["resultado"] = "DIVERGENTE"
    expected.loc[
        expected["carga_liquida_pmo_mw"].isna(), "resultado"
    ] = "AUSENTE NO PMO"
    missing_load = expected[["mean_mw", "fator_carga"]].isna().any(axis=1)
    expected.loc[missing_load, "resultado"] = "DADO DE CARGA COBRE AUSENTE"
    missing_ncs = ~expected["dados_ncs_completos"].fillna(False).astype(bool)
    expected.loc[missing_ncs, "resultado"] = "DADO DE GERACAO COBRE AUSENTE"

    comparable = (
        expected["carga_liquida_pmo_mw"].notna()
        & ~missing_load
        & ~missing_ncs
    )
    expected.loc[
        comparable & expected["diferenca_mw"].abs().le(tolerance_mw),
        "resultado",
    ] = "OK"

    expected["periodo"] = expected.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if row["tipo_periodo"] == "ESTUDO"
            else f"POS {int(row['ciclo_pos'])} - {MONTH_NAMES[int(row['mes'])]}"
        ),
        axis=1,
    )

    summary = (
        expected.groupby(
            [
                "bus_id",
                "codigo_submercado",
                "subsistema",
                "patamar",
                "block_id",
            ],
            as_index=False,
        )
        .agg(
            valores=("resultado", "size"),
            valores_ok=("resultado", lambda values: int((values == "OK").sum())),
            divergencias=(
                "resultado", lambda values: int((values != "OK").sum())
            ),
            maior_diferenca_abs_mw=(
                "diferenca_mw",
                lambda values: float(values.abs().max())
                if values.notna().any()
                else float("nan"),
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "bus_id",
        "codigo_submercado",
        "subsistema",
        "nome_barra_cobre",
        "stage_id",
        "periodo",
        "tipo_periodo",
        "ano",
        "mes",
        "patamar",
        "block_id",
        "mean_mw",
        "fator_carga",
        "carga_bruta_patamar_mw",
        "quantidade_fontes",
        "geracao_nao_simulada_patamar_mw",
        "carga_liquida_cobre_calculada_mw",
        "carga_liquida_pmo_mw",
        "diferenca_mw",
        "resultado",
    ]
    return NetLoadComparisonResult(
        detail=expected[ordered]
        .sort_values(["bus_id", "stage_id", "block_id"])
        .reset_index(drop=True),
        summary=summary,
    )


# =============================================================================
# ABA 6 - DURACAO DOS PATAMARES EM PU
# Palavra-chave para localizar esta logica: ABA 6 DURACAO DOS PATAMARES
# =============================================================================

@dataclass(frozen=True)
class BlockDurationComparisonResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


def parse_patamar_block_durations(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le a duracao mensal em pu dos patamares no inicio do PATAMAR.DAT."""

    lines = _read_text(source).splitlines()
    number_start = _find_block(lines, "NUMERO DE PATAMARES")
    duration_title = _find_block(
        lines, "DURACAO MENSAL DOS PATAMARES DE CARGA", number_start
    )

    number_of_blocks: int | None = None
    for line in lines[number_start + 1 : duration_title]:
        if CODE_ROW.fullmatch(line):
            number_of_blocks = int(line.strip())
            break
    if number_of_blocks is None or number_of_blocks <= 0:
        raise ValueError("O numero de patamares nao foi identificado no PATAMAR.DAT.")

    end = next(
        (
            index
            for index in range(duration_title + 1, len(lines))
            if _normalized(lines[index]).startswith("SUBSISTEMA")
        ),
        None,
    )
    if end is None:
        raise ValueError(
            "O final do bloco de duracao dos patamares nao foi encontrado."
        )

    rows: list[dict[str, object]] = []
    current_year: int | None = None
    current_block: int | None = None

    for line in lines[duration_title + 1 : end]:
        year_match = re.match(r"^\s*(\d{4})\s+(.+)$", line)
        if year_match:
            current_year = int(year_match.group(1))
            current_block = 0
            values_text = year_match.group(2)
        elif (
            current_year is not None
            and current_block is not None
            and re.match(r"^\s+[-+]?\d", line)
        ):
            current_block += 1
            values_text = line.strip()
        else:
            continue

        if current_block >= number_of_blocks:
            raise ValueError(
                "Foram encontradas mais linhas de duracao do que o numero "
                "de patamares declarado no PATAMAR.DAT."
            )
        values = [float(token) for token in NUMBER.findall(values_text)]
        if len(values) != 12:
            raise ValueError(
                "Esperavam-se 12 duracoes mensais para cada patamar do "
                "PATAMAR.DAT."
            )
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "ano_fonte": current_year,
                    "mes": month,
                    "block_id": current_block,
                    "duracao_newave_pu": value,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nenhuma duracao foi lida no PATAMAR.DAT.")

    keys = ["ano_fonte", "mes", "block_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Existem duracoes duplicadas no PATAMAR.DAT.")

    counts = frame.groupby(["ano_fonte", "mes"])["block_id"].nunique()
    if not counts.eq(number_of_blocks).all():
        raise ValueError(
            "Algum mes do PATAMAR.DAT nao possui todos os patamares declarados."
        )

    frame["soma_duracoes_newave_pu"] = frame.groupby(
        ["ano_fonte", "mes"]
    )["duracao_newave_pu"].transform("sum")
    return frame.sort_values(keys).reset_index(drop=True)


def parse_stages_block_durations(
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Le as horas dos blocos do stages.json e as normaliza dentro do estagio."""

    try:
        payload = json.loads(_read_text(source))
    except json.JSONDecodeError as error:
        raise ValueError(f"O stages.json nao e um JSON valido: {error}") from error

    stages = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError("A lista 'stages' nao foi encontrada no stages.json.")

    rows: list[dict[str, object]] = []
    stage_ids: set[int] = set()
    for stage in stages:
        try:
            stage_id = int(stage["id"])
            start_date = pd.Timestamp(str(stage["start_date"]))
            end_date = pd.Timestamp(str(stage["end_date"]))
            blocks = stage["blocks"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Existe um estagio incompleto no stages.json.") from error

        if stage_id in stage_ids:
            raise ValueError(f"O stage_id {stage_id} esta duplicado no stages.json.")
        stage_ids.add(stage_id)
        if end_date <= start_date:
            raise ValueError(
                f"O stage_id {stage_id} possui datas inicial e final invalidas."
            )
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"O stage_id {stage_id} nao possui blocos de carga.")

        parsed_blocks: list[tuple[int, str, float]] = []
        block_ids: set[int] = set()
        for block in blocks:
            try:
                block_id = int(block["id"])
                name = str(block.get("name", f"Bloco {block_id}"))
                hours = float(block["hours"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Existe um bloco incompleto no stage_id {stage_id}."
                ) from error
            if block_id in block_ids:
                raise ValueError(
                    f"O block_id {block_id} esta duplicado no stage_id {stage_id}."
                )
            if hours != hours or hours in (float("inf"), float("-inf")) or hours < 0:
                raise ValueError(
                    f"O block_id {block_id} do stage_id {stage_id} possui horas invalidas."
                )
            block_ids.add(block_id)
            parsed_blocks.append((block_id, name, hours))

        total_hours = sum(hours for _, _, hours in parsed_blocks)
        if total_hours <= 0 or total_hours in (float("inf"), float("-inf")):
            raise ValueError(
                f"A soma das horas do stage_id {stage_id} precisa ser positiva."
            )
        calendar_hours = (end_date - start_date).total_seconds() / 3600.0

        for block_id, name, hours in parsed_blocks:
            rows.append(
                {
                    "stage_id": stage_id,
                    "data_inicio": start_date.strftime("%Y-%m-%d"),
                    "data_fim": end_date.strftime("%Y-%m-%d"),
                    "ano": int(start_date.year),
                    "mes": int(start_date.month),
                    "block_id": block_id,
                    "nome_bloco_cobre": name,
                    "horas_cobre": hours,
                    "total_horas_estagio": total_hours,
                    "horas_calendario_estagio": calendar_hours,
                    "duracao_cobre_pu": hours / total_hours,
                }
            )

    frame = pd.DataFrame(rows)
    keys = ["stage_id", "block_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Existem blocos duplicados no stages.json.")
    frame["soma_duracoes_cobre_pu"] = frame.groupby("stage_id")[
        "duracao_cobre_pu"
    ].transform("sum")
    return frame.sort_values(keys).reset_index(drop=True)


def build_block_duration_comparison(
    patamar: pd.DataFrame,
    cobre_durations: pd.DataFrame,
    tolerance: float,
) -> BlockDurationComparisonResult:
    """Compara a duracao NEWAVE com as horas normalizadas de cada bloco COBRE."""

    if tolerance < 0:
        raise ValueError("A tolerancia nao pode ser negativa.")
    if patamar.empty or cobre_durations.empty:
        raise ValueError("Nao existem duracoes suficientes para a comparacao.")

    source_years = sorted(int(value) for value in patamar["ano_fonte"].unique())
    last_source_year = source_years[-1]
    stage_columns = [
        "stage_id",
        "data_inicio",
        "data_fim",
        "ano",
        "mes",
        "horas_calendario_estagio",
    ]
    stage_table = (
        cobre_durations[stage_columns]
        .drop_duplicates("stage_id")
        .sort_values("stage_id")
        .reset_index(drop=True)
    )
    stage_table["tipo_periodo"] = stage_table["ano"].map(
        lambda year: "ESTUDO" if int(year) <= last_source_year else "POS"
    )
    stage_table["ano_fonte"] = stage_table["ano"].map(
        lambda year: last_source_year if int(year) > last_source_year else int(year)
    )

    expected = stage_table.merge(
        patamar,
        on=["ano_fonte", "mes"],
        how="left",
    )
    cobre_values = cobre_durations[
        [
            "stage_id",
            "block_id",
            "nome_bloco_cobre",
            "horas_cobre",
            "total_horas_estagio",
            "duracao_cobre_pu",
            "soma_duracoes_cobre_pu",
        ]
    ]
    detail = expected.merge(
        cobre_values,
        on=["stage_id", "block_id"],
        how="outer",
        indicator=True,
    )

    stage_lookup = stage_table.set_index("stage_id")
    for column in [
        "data_inicio",
        "data_fim",
        "ano",
        "mes",
        "horas_calendario_estagio",
        "tipo_periodo",
        "ano_fonte",
    ]:
        detail[column] = detail[column].fillna(
            detail["stage_id"].map(stage_lookup[column])
        )

    detail["diferenca_pu"] = (
        detail["duracao_cobre_pu"] - detail["duracao_newave_pu"]
    )
    detail["resultado"] = "DIVERGENTE"
    detail.loc[detail["_merge"] == "left_only", "resultado"] = "AUSENTE NO COBRE"
    detail.loc[detail["_merge"] == "right_only", "resultado"] = "AUSENTE NO NEWAVE"
    comparable = detail["_merge"] == "both"
    detail.loc[
        comparable & detail["diferenca_pu"].abs().le(tolerance),
        "resultado",
    ] = "OK"

    detail["patamar"] = detail["block_id"] + 1
    detail["periodo"] = detail.apply(
        lambda row: (
            f"{MONTH_NAMES[int(row['mes'])]}/{int(row['ano'])}"
            if pd.notna(row.get("mes")) and pd.notna(row.get("ano"))
            else "Periodo nao identificado"
        ),
        axis=1,
    )

    summary = (
        detail.groupby(["patamar", "block_id"], dropna=False, as_index=False)
        .agg(
            valores=("resultado", "size"),
            valores_ok=("resultado", lambda values: int((values == "OK").sum())),
            divergencias=(
                "resultado", lambda values: int((values != "OK").sum())
            ),
            maior_diferenca_abs_pu=(
                "diferenca_pu",
                lambda values: float(values.abs().max())
                if values.notna().any()
                else float("nan"),
            ),
        )
    )
    summary["resultado"] = summary["divergencias"].eq(0).map(
        {True: "OK", False: "DIVERGENTE"}
    )

    ordered = [
        "stage_id",
        "periodo",
        "tipo_periodo",
        "data_inicio",
        "data_fim",
        "ano",
        "mes",
        "ano_fonte",
        "patamar",
        "block_id",
        "nome_bloco_cobre",
        "horas_cobre",
        "total_horas_estagio",
        "horas_calendario_estagio",
        "duracao_newave_pu",
        "soma_duracoes_newave_pu",
        "duracao_cobre_pu",
        "soma_duracoes_cobre_pu",
        "diferenca_pu",
        "resultado",
    ]
    return BlockDurationComparisonResult(
        detail=detail[ordered]
        .sort_values(["stage_id", "block_id"], na_position="last")
        .reset_index(drop=True),
        summary=summary.sort_values("block_id").reset_index(drop=True),
    )


# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Validador NEWAVE → COBRE",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fc; }
    .main .block-container { max-width: 1450px; padding-top: 2rem; }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #0c4160 0%, #176b87 55%, #2a9d8f 100%);
        box-shadow: 0 12px 28px rgba(12, 65, 96, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .55rem 0 0; opacity: .94; }
    .formula-card {
        background: white;
        border: 1px solid #dce7ee;
        border-left: 6px solid #2a9d8f;
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        margin: .8rem 0 1rem;
        box-shadow: 0 6px 18px rgba(18, 70, 90, 0.06);
        text-align: center;
    }
    .formula-card .label { color: #537080; font-size: .9rem; margin-bottom: .35rem; }
    .formula-card .formula { color: #123c50; font-weight: 700; font-size: 1.22rem; }
    .explain-card {
        background: #eef7f7;
        border-radius: 13px;
        padding: 1rem 1.2rem;
        color: #234b57;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e8ed;
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>⚡ Validador NEWAVE → COBRE</h1>
      <p>Comprova a carga bruta, a geração não controlada e a carga líquida por patamar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
@st.cache_data(show_spinner=False)
def _load_direct_sources(
    sistema_bytes: bytes, additional_bytes: bytes, parquet_bytes: bytes
) -> tuple[ParsedSistema, pd.DataFrame, pd.DataFrame]:
    return (
        parse_sistema(sistema_bytes),
        parse_c_adic(additional_bytes),
        parse_load_parquet(parquet_bytes),
    )


@st.cache_data(show_spinner=False)
def _load_pmo_sources(
    pmo_bytes: bytes, parquet_bytes: bytes
) -> tuple[ParsedSistema, pd.DataFrame, pd.DataFrame]:
    sistema, additional = parse_pmo(pmo_bytes)
    return sistema, additional, parse_load_parquet(parquet_bytes)


@st.cache_data(show_spinner=False)
def _load_factor_sources(
    sistema_bytes: bytes, patamar_bytes: bytes, json_bytes: bytes
) -> tuple[ParsedSistema, pd.DataFrame, pd.DataFrame]:
    return (
        parse_sistema(sistema_bytes),
        parse_patamar_load_factors(patamar_bytes),
        parse_load_factors_json(json_bytes),
    )


@st.cache_data(show_spinner=False)
def _load_non_controllable_sources(
    sistema_bytes: bytes, sources_bytes: bytes, stats_bytes: bytes
) -> tuple[ParsedSistema, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        parse_sistema(sistema_bytes),
        parse_sistema_non_controllable(sistema_bytes),
        parse_non_controllable_sources_json(sources_bytes),
        parse_non_controllable_stats_parquet(stats_bytes),
    )


@st.cache_data(show_spinner=False)
def _load_non_controllable_factor_sources(
    sistema_bytes: bytes, patamar_bytes: bytes, json_bytes: bytes
) -> tuple[ParsedSistema, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        parse_sistema(sistema_bytes),
        parse_sistema_non_controllable(sistema_bytes),
        parse_patamar_non_controllable_factors(patamar_bytes),
        parse_non_controllable_factors_json(json_bytes),
    )


@st.cache_data(show_spinner=False)
def _load_net_load_sources(
    pmo_bytes: bytes,
    buses_bytes: bytes,
    load_stats_bytes: bytes,
    load_factors_bytes: bytes,
    ncs_sources_bytes: bytes,
    ncs_stats_bytes: bytes,
    ncs_factors_bytes: bytes,
) -> tuple[
    ParsedSistema,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    sistema, _ = parse_pmo(pmo_bytes)
    return (
        sistema,
        parse_pmo_net_load(pmo_bytes),
        parse_buses_json(buses_bytes),
        parse_load_parquet(load_stats_bytes),
        parse_load_factors_json(load_factors_bytes),
        parse_non_controllable_sources_json(ncs_sources_bytes),
        parse_non_controllable_stats_parquet(ncs_stats_bytes),
        parse_non_controllable_factors_json(ncs_factors_bytes),
    )


@st.cache_data(show_spinner=False)
def _load_block_duration_sources(
    patamar_bytes: bytes, stages_bytes: bytes
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        parse_patamar_block_durations(patamar_bytes),
        parse_stages_block_durations(stages_bytes),
    )


def _format_pt(value: float, decimals: int = 4) -> str:
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def _render_load_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 1 — FORMAÇÃO DA CARGA BRUTA</div>
          <div class="formula">Carga bruta COBRE = Mercado total NEWAVE + Carga adicional NEWAVE (inclui MMGD)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>Diferença de representação</b><br>
          No NEWAVE, a carga fica dividida entre mercado total e cargas adicionais.
          A MMGD faz parte da carga adicional. No COBRE, essas parcelas são consolidadas
          na coluna <code>mean_mw</code> de <code>load_seasonal_stats.parquet</code>.
          O mercado líquido do NEWAVE não é usado nesta validação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Como deseja fornecer os dados do NEWAVE?",
        ["Relatório pmo.dat", "SISTEMA.DAT + C_ADIC.DAT"],
        horizontal=True,
        help="Os dois modos executam a mesma validação.",
        key="load_input_mode",
    )

    pmo_file = None
    sistema_file = None
    additional_file = None
    if input_mode == "Relatório pmo.dat":
        upload1, upload2 = st.columns(2)
        pmo_file = upload1.file_uploader(
            "pmo.dat", type=["dat", "txt"], key="load_pmo"
        )
        parquet_file = upload2.file_uploader(
            "load_seasonal_stats.parquet",
            type=["parquet"],
            key="load_parquet_pmo",
        )
        ready = pmo_file is not None and parquet_file is not None
        needed_message = "Carregue o pmo.dat e o load_seasonal_stats.parquet."
    else:
        upload1, upload2, upload3 = st.columns(3)
        sistema_file = upload1.file_uploader(
            "SISTEMA.DAT", type=["dat", "txt"], key="load_sistema"
        )
        additional_file = upload2.file_uploader(
            "C_ADIC.DAT", type=["dat", "txt"], key="load_cadic"
        )
        parquet_file = upload3.file_uploader(
            "load_seasonal_stats.parquet",
            type=["parquet"],
            key="load_parquet_direct",
        )
        ready = all(
            item is not None
            for item in (sistema_file, additional_file, parquet_file)
        )
        needed_message = (
            "Carregue o SISTEMA.DAT, o C_ADIC.DAT e o "
            "load_seasonal_stats.parquet."
        )

    tolerance = st.number_input(
        "Tolerância da comparação da carga (MW)",
        min_value=0.0,
        value=0.01,
        step=0.01,
        format="%.4f",
        key="load_tolerance",
    )
    if not ready:
        st.info(needed_message)
        return

    try:
        with st.spinner("Lendo os arquivos e montando a comparação da carga..."):
            parquet_data = parquet_file.getvalue()
            if input_mode == "Relatório pmo.dat":
                source_bytes = pmo_file.getvalue()
                sistema, additional, load = _load_pmo_sources(
                    source_bytes, parquet_data
                )
                source_label = "pmo.dat + load_seasonal_stats.parquet"
            else:
                sistema_bytes = sistema_file.getvalue()
                additional_bytes = additional_file.getvalue()
                source_bytes = sistema_bytes + additional_bytes
                sistema, additional, load = _load_direct_sources(
                    sistema_bytes, additional_bytes, parquet_data
                )
                source_label = (
                    "SISTEMA.DAT + C_ADIC.DAT + load_seasonal_stats.parquet"
                )

            case_key = hashlib.sha256(source_bytes + parquet_data).hexdigest()[:12]
            mapping = infer_bus_mapping(sistema, load)
            result = build_comparison(
                sistema=sistema,
                additional=additional,
                load=load,
                mapping=mapping,
                tolerance_mw=float(tolerance),
            )
    except Exception as error:
        st.error(f"Não foi possível realizar a validação da carga: {error}")
        return

    st.caption(f"Origem: **{source_label}** · Caso: {case_key}")
    total = len(result.detail)
    divergences = int((result.detail["resultado"] != "OK").sum())
    correct = total - divergences
    max_difference = float(result.detail["diferenca_mw"].abs().max())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas comparadas", total)
    col2.metric("Linhas corretas", correct)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", f"{_format_pt(max_difference)} MW")

    if divergences == 0:
        st.success(
            "Validação aprovada: mercado total + carga adicional = carga bruta COBRE "
            "em todos os estágios."
        )
    else:
        st.warning(
            f"Foram encontradas {divergences} divergências acima da tolerância."
        )

    with st.expander("Correspondência identificada entre NEWAVE e COBRE"):
        st.dataframe(
            mapping,
            hide_index=True,
            use_container_width=True,
            column_config={
                "codigo_submercado": "Código/número NEWAVE",
                "subsistema": "Subsistema",
                "bus_id": "bus_id COBRE",
                "tem_mercado": "Possui mercado",
            },
        )

    st.subheader("Resumo por subsistema")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "bus_id": "Barra COBRE",
            "codigo_submercado": "Código/número NEWAVE",
            "subsistema": "Subsistema",
            "estagios": "Estágios",
            "estagios_ok": "Corretos",
            "divergencias": "Divergências",
            "maior_diferenca_abs_mw": st.column_config.NumberColumn(
                "Maior diferença (MW)", format="%.4f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação estágio a estágio")
    filter1, filter2, filter3 = st.columns(3)
    subsystem_options = result.detail["subsistema"].drop_duplicates().tolist()
    selected_subsystems = filter1.multiselect(
        "Subsistemas",
        subsystem_options,
        default=subsystem_options,
        key="load_filter_subsystems",
    )
    period_options = result.detail["tipo_periodo"].drop_duplicates().tolist()
    selected_periods = filter2.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="load_filter_periods",
    )
    selected_results = filter3.multiselect(
        "Resultado",
        ["OK", "DIVERGENTE"],
        default=["OK", "DIVERGENTE"],
        key="load_filter_results",
    )

    shown = result.detail[
        result.detail["subsistema"].isin(selected_subsystems)
        & result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["resultado"].isin(selected_results)
    ]
    visible_columns = [
        "subsistema",
        "bus_id",
        "stage_id",
        "periodo",
        "mercado_total_mw",
        "c_adic_mw",
        "componentes_c_adic",
        "carga_bruta_esperada_mw",
        "carga_bruta_cobre_mw",
        "diferenca_mw",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config={
            "subsistema": "Subsistema",
            "bus_id": "Barra",
            "stage_id": "Estágio",
            "periodo": "Período",
            "mercado_total_mw": st.column_config.NumberColumn(
                "Mercado total NEWAVE (MW)", format="%.2f"
            ),
            "c_adic_mw": st.column_config.NumberColumn(
                "Carga adicional (MW)", format="%.2f"
            ),
            "componentes_c_adic": "Parcelas da carga adicional",
            "carga_bruta_esperada_mw": st.column_config.NumberColumn(
                "Soma esperada (MW)", format="%.2f"
            ),
            "carga_bruta_cobre_mw": st.column_config.NumberColumn(
                "Carga bruta COBRE (MW)", format="%.2f"
            ),
            "diferenca_mw": st.column_config.NumberColumn(
                "Diferença (MW)", format="%.4f"
            ),
            "resultado": "Resultado",
        },
    )

    if input_mode == "SISTEMA.DAT + C_ADIC.DAT":
        mmgd_rows = additional[
            additional["razao"].astype(str).str.contains(
                "MMGD", case=False, na=False
            )
        ]
        with st.expander("Parcelas de MMGD encontradas no C_ADIC.DAT"):
            if mmgd_rows.empty:
                st.info(
                    "Nenhuma parcela com a identificação MMGD foi encontrada neste caso."
                )
            else:
                mmgd_summary = (
                    mmgd_rows.groupby(
                        ["codigo_submercado", "subsistema", "razao"],
                        as_index=False,
                    )
                    .agg(
                        mínimo_mw=("valor_mw", "min"),
                        máximo_mw=("valor_mw", "max"),
                    )
                )
                st.dataframe(
                    mmgd_summary, hide_index=True, use_container_width=True
                )

    st.download_button(
        "Baixar relatório da carga em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_validacao_carga_newave_cobre.csv",
        mime="text/csv",
        key="download_load_report",
    )


def _render_factor_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 2 — FATORES DE CARGA POR PATAMAR</div>
          <div class="formula">PATAMAR.DAT · CARGA (p.u. demanda média) = load_factors.json · factor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>O que está sendo comparado?</b><br>
          O fator do PATAMAR.DAT transforma a carga média mensal em carga de cada
          patamar. No COBRE, o mesmo fator aparece em <code>block_factors.factor</code>.
          Nesta etapa são comparados somente os fatores; a duração dos patamares não
          faz parte desta validação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload1, upload2, upload3 = st.columns(3)
    sistema_file = upload1.file_uploader(
        "SISTEMA.DAT", type=["dat", "txt"], key="factor_sistema"
    )
    patamar_file = upload2.file_uploader(
        "PATAMAR.DAT", type=["dat", "txt"], key="factor_patamar"
    )
    json_file = upload3.file_uploader(
        "load_factors.json", type=["json"], key="factor_json"
    )
    tolerance = st.number_input(
        "Tolerância da comparação dos fatores",
        min_value=0.0,
        value=0.000001,
        step=0.000001,
        format="%.8f",
        key="factor_tolerance",
    )

    if not all(item is not None for item in (sistema_file, patamar_file, json_file)):
        st.info("Carregue o SISTEMA.DAT, o PATAMAR.DAT e o load_factors.json.")
        return

    try:
        with st.spinner("Lendo os fatores e montando a comparação por patamar..."):
            sistema_bytes = sistema_file.getvalue()
            patamar_bytes = patamar_file.getvalue()
            json_bytes = json_file.getvalue()
            sistema, patamar, cobre_factors = _load_factor_sources(
                sistema_bytes, patamar_bytes, json_bytes
            )
            result = build_factor_comparison(
                sistema=sistema,
                patamar=patamar,
                cobre_factors=cobre_factors,
                tolerance=float(tolerance),
            )
            case_key = hashlib.sha256(
                sistema_bytes + patamar_bytes + json_bytes
            ).hexdigest()[:12]
    except Exception as error:
        st.error(f"Não foi possível realizar a validação dos fatores: {error}")
        return

    st.caption(
        "Origem: **SISTEMA.DAT + PATAMAR.DAT + load_factors.json** "
        f"· Caso: {case_key}"
    )
    total = len(result.detail)
    divergences = int((result.detail["resultado"] != "OK").sum())
    correct = total - divergences
    differences = result.detail["diferenca"].abs().dropna()
    max_difference = float(differences.max()) if not differences.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fatores comparados", total)
    col2.metric("Fatores corretos", correct)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", _format_pt(max_difference, 8))

    if divergences == 0:
        st.success(
            "Validação aprovada: todos os fatores de carga por patamar do NEWAVE "
            "foram representados corretamente no COBRE."
        )
    else:
        st.warning(
            f"Foram encontrados {divergences} fatores divergentes ou ausentes."
        )

    mapping = _load_factor_mapping(sistema)
    with st.expander("Correspondência usada na comparação"):
        correspondence = mapping.copy()
        correspondence["patamares"] = (
            "Patamar NEWAVE 1, 2, 3 → block_id COBRE 0, 1, 2"
        )
        st.dataframe(
            correspondence,
            hide_index=True,
            use_container_width=True,
            column_config={
                "codigo_submercado": "Código NEWAVE",
                "subsistema": "Subsistema",
                "bus_id": "bus_id COBRE",
                "patamares": "Correspondência dos patamares",
            },
        )
        st.caption(
            "Nos estágios posteriores ao estudo, o perfil do último ano do "
            "PATAMAR.DAT é repetido sazonalmente."
        )

    st.subheader("Resumo por subsistema")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "bus_id": "Barra COBRE",
            "subsistema": "Subsistema",
            "fatores": "Fatores",
            "fatores_ok": "Corretos",
            "divergencias": "Divergências",
            "maior_diferenca_abs": st.column_config.NumberColumn(
                "Maior diferença", format="%.8f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação fator a fator")
    filter1, filter2, filter3, filter4 = st.columns(4)
    subsystem_options = result.detail["subsistema"].drop_duplicates().tolist()
    selected_subsystems = filter1.multiselect(
        "Subsistemas",
        subsystem_options,
        default=subsystem_options,
        key="factor_filter_subsystems",
    )
    period_options = [
        value
        for value in result.detail["tipo_periodo"].dropna().drop_duplicates().tolist()
    ]
    selected_periods = filter2.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="factor_filter_periods",
    )
    patamar_options = sorted(
        int(value) for value in result.detail["patamar"].dropna().unique()
    )
    selected_patamares = filter3.multiselect(
        "Patamares",
        patamar_options,
        default=patamar_options,
        key="factor_filter_blocks",
    )
    result_options = result.detail["resultado"].drop_duplicates().tolist()
    selected_results = filter4.multiselect(
        "Resultado",
        result_options,
        default=result_options,
        key="factor_filter_results",
    )

    shown = result.detail[
        result.detail["subsistema"].isin(selected_subsystems)
        & result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["patamar"].isin(selected_patamares)
        & result.detail["resultado"].isin(selected_results)
    ]
    visible_columns = [
        "subsistema",
        "bus_id",
        "stage_id",
        "periodo",
        "patamar",
        "block_id",
        "fator_newave",
        "fator_cobre",
        "diferenca",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=540,
        column_config={
            "subsistema": "Subsistema",
            "bus_id": "Barra",
            "stage_id": "Estágio",
            "periodo": "Período",
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "fator_newave": st.column_config.NumberColumn(
                "Fator NEWAVE", format="%.4f"
            ),
            "fator_cobre": st.column_config.NumberColumn(
                "Fator COBRE", format="%.4f"
            ),
            "diferenca": st.column_config.NumberColumn(
                "Diferença", format="%.8f"
            ),
            "resultado": "Resultado",
        },
    )

    st.download_button(
        "Baixar relatório dos fatores em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_validacao_fatores_patamar.csv",
        mime="text/csv",
        key="download_factor_report",
    )


# =============================================================================
# ABA 3 - INTERFACE DA GERACAO NAO CONTROLADA MEDIA
# Palavra-chave para localizar a tela: ABA 3 INTERFACE GERACAO NAO CONTROLADA
# =============================================================================
def _render_non_controllable_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 3 — GERAÇÃO NÃO CONTROLADA MÉDIA</div>
          <div class="formula">Geração COBRE (MW) = max_generation_mw × mean</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>O que está sendo validado?</b><br>
          O SISTEMA.DAT informa a geração média de cada tipo de fonte por
          subsistema e mês. No COBRE, o cadastro informa
          <code>max_generation_mw</code> e o Parquet informa a disponibilidade
          média <code>mean</code>. O programa multiplica essas duas parcelas e
          compara o resultado por estágio. O nome da fonte é apenas informativo:
          <b>ele nunca é usado como chave.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload1, upload2, upload3 = st.columns(3)
    sistema_file = upload1.file_uploader(
        "SISTEMA.DAT", type=["dat", "txt"], key="ncs_sistema"
    )
    sources_file = upload2.file_uploader(
        "non_controllable_sources.json", type=["json"], key="ncs_sources"
    )
    stats_file = upload3.file_uploader(
        "non_controllable_stats.parquet", type=["parquet"], key="ncs_stats"
    )
    tolerance = st.number_input(
        "Tolerância da comparação da geração (MW)",
        min_value=0.0,
        value=0.01,
        step=0.01,
        format="%.4f",
        key="ncs_tolerance",
    )

    if not all(item is not None for item in (sistema_file, sources_file, stats_file)):
        st.info(
            "Carregue o SISTEMA.DAT, o non_controllable_sources.json e o "
            "non_controllable_stats.parquet."
        )
        return

    try:
        with st.spinner("Lendo e comparando a geração não controlada média..."):
            sistema_bytes = sistema_file.getvalue()
            sources_bytes = sources_file.getvalue()
            stats_bytes = stats_file.getvalue()
            sistema, generation, sources, stats = _load_non_controllable_sources(
                sistema_bytes, sources_bytes, stats_bytes
            )
            result = build_non_controllable_comparison(
                sistema=sistema,
                generation=generation,
                sources=sources,
                stats=stats,
                tolerance_mw=float(tolerance),
            )
            case_key = hashlib.sha256(
                sistema_bytes + sources_bytes + stats_bytes
            ).hexdigest()[:12]
    except Exception as error:
        st.error(
            "Não foi possível realizar a validação da geração não "
            f"controlada: {error}"
        )
        return

    st.caption(
        "Origem: **SISTEMA.DAT + non_controllable_sources.json + "
        f"non_controllable_stats.parquet** · Caso: {case_key}"
    )
    total = len(result.detail)
    divergences = int((result.detail["resultado"] != "OK").sum())
    correct = total - divergences
    differences = result.detail["diferenca_mw"].abs().dropna()
    max_difference = float(differences.max()) if not differences.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valores comparados", total)
    col2.metric("Valores corretos", correct)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", f"{_format_pt(max_difference)} MW")

    if divergences == 0:
        st.success(
            "Validação aprovada: a geração não controlada média foi "
            "representada corretamente em todos os estágios."
        )
    else:
        st.warning(
            f"Foram encontrados {divergences} valores divergentes ou ausentes."
        )

    mapping = _non_controllable_source_mapping(sistema, generation)
    with st.expander("Indexação técnica usada na comparação"):
        st.dataframe(
            mapping,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ncs_id": "ncs_id COBRE",
                "codigo_submercado": "Código do subsistema NEWAVE",
                "indice_fonte": "Índice do tipo de fonte",
                "tipo_fonte": "Tipo de fonte (informativo)",
                "subsistema": "Subsistema",
                "bus_id": "bus_id esperado",
            },
        )
        st.caption(
            "O ncs_id é reproduzido pela ordenação do código do "
            "subsistema e do índice da fonte. O nome não participa da indexação."
        )

    st.subheader("Resumo por subsistema e tipo de fonte")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "bus_id": "Barra COBRE",
            "codigo_submercado": "Código NEWAVE",
            "subsistema": "Subsistema",
            "indice_fonte": "Índice da fonte",
            "tipo_fonte": "Tipo de fonte",
            "estagios": "Estágios",
            "estagios_ok": "Corretos",
            "divergencias": "Divergências",
            "maior_diferenca_abs_mw": st.column_config.NumberColumn(
                "Maior diferença (MW)", format="%.4f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação estágio a estágio")
    filter1, filter2, filter3, filter4 = st.columns(4)
    subsystem_options = result.detail["subsistema"].drop_duplicates().tolist()
    selected_subsystems = filter1.multiselect(
        "Subsistemas",
        subsystem_options,
        default=subsystem_options,
        key="ncs_filter_subsystems",
    )
    source_options = sorted(
        int(value) for value in result.detail["indice_fonte"].dropna().unique()
    )
    selected_sources = filter2.multiselect(
        "Índices dos tipos de fonte",
        source_options,
        default=source_options,
        key="ncs_filter_sources",
    )
    period_options = [
        value
        for value in result.detail["tipo_periodo"].dropna().drop_duplicates().tolist()
    ]
    selected_periods = filter3.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="ncs_filter_periods",
    )
    result_options = result.detail["resultado"].drop_duplicates().tolist()
    selected_results = filter4.multiselect(
        "Resultado",
        result_options,
        default=result_options,
        key="ncs_filter_results",
    )

    shown = result.detail[
        result.detail["subsistema"].isin(selected_subsystems)
        & result.detail["indice_fonte"].isin(selected_sources)
        & result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["resultado"].isin(selected_results)
    ].copy()
    shown["nome_fonte_cobre"] = shown["nome_fonte_cobre"].fillna("").replace(
        "", "Nome não informado"
    )
    visible_columns = [
        "subsistema",
        "codigo_submercado",
        "indice_fonte",
        "tipo_fonte",
        "ncs_id",
        "nome_fonte_cobre",
        "stage_id",
        "periodo",
        "max_generation_mw",
        "mean",
        "geracao_newave_mw",
        "geracao_cobre_mw",
        "diferenca_mw",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=540,
        column_config={
            "subsistema": "Subsistema",
            "codigo_submercado": "Código NEWAVE",
            "indice_fonte": "Índice da fonte",
            "tipo_fonte": "Tipo de fonte",
            "ncs_id": "ncs_id COBRE",
            "nome_fonte_cobre": "Nome COBRE (informativo)",
            "stage_id": "Estágio",
            "periodo": "Período",
            "max_generation_mw": st.column_config.NumberColumn(
                "max_generation_mw", format="%.4f"
            ),
            "mean": st.column_config.NumberColumn("mean", format="%.8f"),
            "geracao_newave_mw": st.column_config.NumberColumn(
                "Geração NEWAVE (MW)", format="%.4f"
            ),
            "geracao_cobre_mw": st.column_config.NumberColumn(
                "Geração COBRE (MW)", format="%.4f"
            ),
            "diferenca_mw": st.column_config.NumberColumn(
                "Diferença (MW)", format="%.4f"
            ),
            "resultado": "Resultado",
        },
    )

    st.download_button(
        "Baixar relatório da geração não controlada em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_geracao_nao_controlada.csv",
        mime="text/csv",
        key="download_ncs_report",
    )


# =============================================================================
# ABA 4 - INTERFACE DOS FATORES DA GERACAO NAO CONTROLADA
# Palavra-chave para localizar a tela: ABA 4 INTERFACE FATORES GERACAO NAO CONTROLADA
# =============================================================================
def _render_non_controllable_factor_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 4 — FATORES DA GERAÇÃO NÃO CONTROLADA POR PATAMAR</div>
          <div class="formula">PATAMAR.DAT · USINAS NÃO SIMULADAS = non_controllable_factors.json · factor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>O que está sendo comparado?</b><br>
          O SISTEMA.DAT identifica os tipos de fonte existentes e sua ordem
          técnica. O PATAMAR.DAT informa o fator de cada fonte, subsistema, mês
          e patamar. No COBRE, o mesmo valor aparece em
          <code>non_controllable_factors[].block_factors[].factor</code>.<br><br>
          A chave da comparação é formada por <b>código do subsistema + índice
          da fonte + data + patamar</b>. O nome da fonte não é usado como chave.
          Nesta validação, a duração dos patamares não é comparada.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload1, upload2, upload3 = st.columns(3)
    sistema_file = upload1.file_uploader(
        "SISTEMA.DAT", type=["dat", "txt"], key="ncs_factor_sistema"
    )
    patamar_file = upload2.file_uploader(
        "PATAMAR.DAT", type=["dat", "txt"], key="ncs_factor_patamar"
    )
    json_file = upload3.file_uploader(
        "non_controllable_factors.json", type=["json"], key="ncs_factor_json"
    )
    tolerance = st.number_input(
        "Tolerância da comparação dos fatores da geração não controlada",
        min_value=0.0,
        value=0.000001,
        step=0.000001,
        format="%.10f",
        key="ncs_factor_tolerance",
    )

    if not all(item is not None for item in (sistema_file, patamar_file, json_file)):
        st.info(
            "Carregue o SISTEMA.DAT, o PATAMAR.DAT e o "
            "non_controllable_factors.json."
        )
        return

    try:
        with st.spinner(
            "Lendo e comparando os fatores da geração não controlada..."
        ):
            sistema_bytes = sistema_file.getvalue()
            patamar_bytes = patamar_file.getvalue()
            json_bytes = json_file.getvalue()
            sistema, generation, patamar, cobre_factors = (
                _load_non_controllable_factor_sources(
                    sistema_bytes, patamar_bytes, json_bytes
                )
            )
            result = build_non_controllable_factor_comparison(
                sistema=sistema,
                generation=generation,
                patamar=patamar,
                cobre_factors=cobre_factors,
                tolerance=float(tolerance),
            )
            case_key = hashlib.sha256(
                sistema_bytes + patamar_bytes + json_bytes
            ).hexdigest()[:12]
    except Exception as error:
        st.error(
            "Não foi possível validar os fatores da geração não controlada: "
            f"{error}"
        )
        return

    st.caption(
        "Origem: **SISTEMA.DAT + PATAMAR.DAT + "
        f"non_controllable_factors.json** · Caso: {case_key}"
    )
    approved = result.detail["resultado"].str.startswith("OK")
    equal_within_tolerance = int(
        result.detail["diferenca"].abs().le(float(tolerance)).sum()
    )
    adjustments = int(
        result.detail["resultado"].eq("OK — AJUSTE MÍNIMO COBRE").sum()
    )
    divergences = int((~approved).sum())
    differences = result.detail["diferenca"].abs().dropna()
    max_difference = float(differences.max()) if not differences.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fatores comparados", len(result.detail))
    col2.metric("Iguais", equal_within_tolerance)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", _format_pt(max_difference, 8))

    if divergences == 0:
        st.success(
            "Validação aprovada: todos os fatores da geração não controlada "
            "foram representados corretamente no COBRE."
        )
    else:
        st.warning(
            f"Foram encontrados {divergences} fatores divergentes ou ausentes."
        )

    if adjustments:
        st.info(
            f"Foram encontrados {adjustments} casos em que o fator 0 do NEWAVE "
            "foi gravado como 0,000001 no COBRE. Isso é um ajuste técnico "
            "esperado, pois o formato do COBRE exige fatores positivos."
        )

    mapping = _non_controllable_source_mapping(sistema, generation)
    with st.expander("Indexação técnica usada na comparação"):
        correspondence = mapping.copy()
        correspondence["patamares"] = (
            "Patamar NEWAVE 1, 2, 3 → block_id COBRE 0, 1, 2"
        )
        st.dataframe(
            correspondence,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ncs_id": "ncs_id COBRE",
                "codigo_submercado": "Código do subsistema NEWAVE",
                "indice_fonte": "Índice da fonte",
                "tipo_fonte": "Tipo de fonte (informativo)",
                "subsistema": "Subsistema",
                "bus_id": "bus_id esperado",
                "patamares": "Correspondência dos patamares",
            },
        )
        st.caption(
            "O ncs_id é reproduzido pela ordenação do código do subsistema e "
            "do índice da fonte. Nos estágios POS, repete-se sazonalmente o "
            "perfil do último ano do PATAMAR.DAT."
        )

    st.subheader("Resumo por subsistema e tipo de fonte")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "ncs_id": "ncs_id COBRE",
            "codigo_submercado": "Código NEWAVE",
            "subsistema": "Subsistema",
            "indice_fonte": "Índice da fonte",
            "tipo_fonte": "Tipo de fonte",
            "fatores": "Fatores",
            "fatores_corretos": "Corretos",
            "ajustes_minimos": "Ajustes mínimos",
            "divergencias": "Divergências",
            "maior_diferenca_abs": st.column_config.NumberColumn(
                "Maior diferença", format="%.8f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação fonte, estágio e patamar")
    filter1, filter2, filter3, filter4, filter5 = st.columns(5)
    subsystem_options = result.detail["subsistema"].drop_duplicates().tolist()
    selected_subsystems = filter1.multiselect(
        "Subsistemas",
        subsystem_options,
        default=subsystem_options,
        key="ncs_factor_filter_subsystems",
    )
    source_options = sorted(
        int(value) for value in result.detail["indice_fonte"].dropna().unique()
    )
    selected_sources = filter2.multiselect(
        "Índices das fontes",
        source_options,
        default=source_options,
        key="ncs_factor_filter_sources",
    )
    period_options = (
        result.detail["tipo_periodo"].dropna().drop_duplicates().tolist()
    )
    selected_periods = filter3.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="ncs_factor_filter_periods",
    )
    block_options = sorted(
        int(value) for value in result.detail["patamar"].dropna().unique()
    )
    selected_blocks = filter4.multiselect(
        "Patamares",
        block_options,
        default=block_options,
        key="ncs_factor_filter_blocks",
    )
    result_options = result.detail["resultado"].drop_duplicates().tolist()
    selected_results = filter5.multiselect(
        "Resultado",
        result_options,
        default=result_options,
        key="ncs_factor_filter_results",
    )

    shown = result.detail[
        result.detail["subsistema"].isin(selected_subsystems)
        & result.detail["indice_fonte"].isin(selected_sources)
        & result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["patamar"].isin(selected_blocks)
        & result.detail["resultado"].isin(selected_results)
    ]
    visible_columns = [
        "subsistema",
        "codigo_submercado",
        "indice_fonte",
        "tipo_fonte",
        "ncs_id",
        "stage_id",
        "periodo",
        "patamar",
        "block_id",
        "fator_newave",
        "fator_cobre",
        "diferenca",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=560,
        column_config={
            "subsistema": "Subsistema",
            "codigo_submercado": "Código NEWAVE",
            "indice_fonte": "Índice da fonte",
            "tipo_fonte": "Tipo de fonte (informativo)",
            "ncs_id": "ncs_id COBRE",
            "stage_id": "Estágio",
            "periodo": "Período",
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "fator_newave": st.column_config.NumberColumn(
                "Fator NEWAVE", format="%.6f"
            ),
            "fator_cobre": st.column_config.NumberColumn(
                "Fator COBRE", format="%.6f"
            ),
            "diferenca": st.column_config.NumberColumn(
                "Diferença", format="%.8f"
            ),
            "resultado": "Resultado",
        },
    )

    st.download_button(
        "Baixar relatório dos fatores da geração não controlada em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_fatores_geracao_nao_controlada.csv",
        mime="text/csv",
        key="download_ncs_factor_report",
    )


# =============================================================================
# ABA 5 - INTERFACE DA CARGA LIQUIDA POR PATAMAR
# Palavra-chave para localizar a tela: ABA 5 INTERFACE CARGA LIQUIDA
# =============================================================================
def _render_net_load_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 5 — CARGA LÍQUIDA POR SUBSISTEMA E PATAMAR</div>
          <div class="formula">
            Carga líquida COBRE = (mean_mw × fator da carga)
            − Σ(max_generation_mw × mean × fator da fonte)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>O que está sendo validado?</b><br>
          O programa calcula primeiro a carga bruta de cada patamar. Depois,
          calcula a geração de cada fonte não simulada no mesmo patamar e soma
          as fontes do subsistema. A subtração dessas parcelas produz a carga
          líquida do COBRE, que é comparada com o bloco
          <code>DADOS DE MERCADO LIQUIDO DE ENERGIA</code> do pmo.dat.<br><br>
          O pmo.dat apresenta valores sem casas decimais. O programa não
          arredonda os valores antes de comparar: a aceitação da diferença é
          controlada pela tolerância escolhida pelo usuário.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Arquivo NEWAVE**")
    pmo_file = st.file_uploader(
        "pmo.dat", type=["dat", "txt"], key="net_load_pmo"
    )

    st.markdown("**Arquivos COBRE — carga e barras**")
    load_col1, load_col2, load_col3 = st.columns(3)
    buses_file = load_col1.file_uploader(
        "buses.json", type=["json"], key="net_load_buses"
    )
    load_stats_file = load_col2.file_uploader(
        "load_seasonal_stats.parquet",
        type=["parquet"],
        key="net_load_stats",
    )
    load_factors_file = load_col3.file_uploader(
        "load_factors.json", type=["json"], key="net_load_factors"
    )

    st.markdown("**Arquivos COBRE — geração não simulada**")
    ncs_col1, ncs_col2, ncs_col3 = st.columns(3)
    ncs_sources_file = ncs_col1.file_uploader(
        "non_controllable_sources.json",
        type=["json"],
        key="net_load_ncs_sources",
    )
    ncs_stats_file = ncs_col2.file_uploader(
        "non_controllable_stats.parquet",
        type=["parquet"],
        key="net_load_ncs_stats",
    )
    ncs_factors_file = ncs_col3.file_uploader(
        "non_controllable_factors.json",
        type=["json"],
        key="net_load_ncs_factors",
    )

    tolerance = st.number_input(
        "Tolerância da comparação da carga líquida (MW)",
        min_value=0.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="net_load_tolerance",
        help=(
            "O PMO deste caso imprime a carga líquida em MW inteiros. "
            "O usuário pode reduzir ou aumentar esta tolerância para auditar "
            "a diferença sem arredondamento."
        ),
    )

    files = (
        pmo_file,
        buses_file,
        load_stats_file,
        load_factors_file,
        ncs_sources_file,
        ncs_stats_file,
        ncs_factors_file,
    )
    if not all(item is not None for item in files):
        st.info("Carregue o pmo.dat e os seis arquivos do COBRE indicados acima.")
        return

    try:
        with st.spinner("Reconstruindo a carga líquida por patamar..."):
            pmo_bytes = pmo_file.getvalue()
            buses_bytes = buses_file.getvalue()
            load_stats_bytes = load_stats_file.getvalue()
            load_factors_bytes = load_factors_file.getvalue()
            ncs_sources_bytes = ncs_sources_file.getvalue()
            ncs_stats_bytes = ncs_stats_file.getvalue()
            ncs_factors_bytes = ncs_factors_file.getvalue()

            (
                sistema,
                pmo_net_load,
                buses,
                load,
                load_factors,
                sources,
                stats,
                non_controllable_factors,
            ) = _load_net_load_sources(
                pmo_bytes,
                buses_bytes,
                load_stats_bytes,
                load_factors_bytes,
                ncs_sources_bytes,
                ncs_stats_bytes,
                ncs_factors_bytes,
            )
            result = build_net_load_comparison(
                sistema=sistema,
                pmo_net_load=pmo_net_load,
                buses=buses,
                load=load,
                load_factors=load_factors,
                sources=sources,
                stats=stats,
                non_controllable_factors=non_controllable_factors,
                tolerance_mw=float(tolerance),
            )
            case_key = hashlib.sha256(
                pmo_bytes
                + buses_bytes
                + load_stats_bytes
                + load_factors_bytes
                + ncs_sources_bytes
                + ncs_stats_bytes
                + ncs_factors_bytes
            ).hexdigest()[:12]
    except Exception as error:
        st.error(f"Não foi possível validar a carga líquida: {error}")
        return

    st.caption(
        "Origem: **pmo.dat + arquivos de carga e geração não simulada do "
        f"COBRE** · Caso: {case_key}"
    )
    total = len(result.detail)
    correct = int(result.detail["resultado"].eq("OK").sum())
    divergences = total - correct
    differences = result.detail["diferenca_mw"].abs().dropna()
    max_difference = float(differences.max()) if not differences.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valores comparados", total)
    col2.metric("Iguais", correct)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", f"{_format_pt(max_difference, 6)} MW")

    if divergences == 0:
        st.success(
            "Validação aprovada: a carga líquida calculada com os dados do "
            "COBRE coincide com o mercado líquido do pmo.dat dentro da "
            "tolerância escolhida."
        )
    else:
        st.warning(
            f"Foram encontrados {divergences} valores divergentes ou ausentes."
        )

    mapping = _load_factor_mapping(sistema).merge(buses, on="bus_id", how="left")
    source_counts = (
        sources.rename(columns={"bus_id_cobre": "bus_id"})
        .groupby("bus_id", as_index=False)["ncs_id"]
        .nunique()
        .rename(columns={"ncs_id": "quantidade_fontes"})
    )
    mapping = mapping.merge(source_counts, on="bus_id", how="left")
    mapping["quantidade_fontes"] = mapping["quantidade_fontes"].fillna(0).astype(int)
    with st.expander("Correspondência técnica usada na comparação"):
        st.dataframe(
            mapping,
            hide_index=True,
            use_container_width=True,
            column_config={
                "codigo_submercado": "Código do subsistema no PMO",
                "subsistema": "Subsistema PMO",
                "bus_id": "bus_id COBRE",
                "nome_barra_cobre": "Barra COBRE (informativo)",
                "quantidade_fontes": "Fontes não simuladas",
            },
        )
        st.caption(
            "Os subsistemas são indexados tecnicamente pelos códigos do PMO e "
            "pelos bus_id. As fontes são associadas por ncs_id e bus_id; seus "
            "nomes não são usados como chave. Patamar NEWAVE 1, 2 e 3 "
            "corresponde a block_id COBRE 0, 1 e 2."
        )

    st.subheader("Resumo por subsistema e patamar")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "bus_id": "bus_id COBRE",
            "codigo_submercado": "Código PMO",
            "subsistema": "Subsistema",
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "valores": "Valores",
            "valores_ok": "Iguais",
            "divergencias": "Divergências",
            "maior_diferenca_abs_mw": st.column_config.NumberColumn(
                "Maior diferença (MW)", format="%.6f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação da conta completa")
    filter1, filter2, filter3, filter4 = st.columns(4)
    subsystem_options = result.detail["subsistema"].drop_duplicates().tolist()
    selected_subsystems = filter1.multiselect(
        "Subsistemas",
        subsystem_options,
        default=subsystem_options,
        key="net_load_filter_subsystems",
    )
    period_options = (
        result.detail["tipo_periodo"].dropna().drop_duplicates().tolist()
    )
    selected_periods = filter2.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="net_load_filter_periods",
    )
    block_options = sorted(
        int(value) for value in result.detail["patamar"].dropna().unique()
    )
    selected_blocks = filter3.multiselect(
        "Patamares",
        block_options,
        default=block_options,
        key="net_load_filter_blocks",
    )
    result_options = result.detail["resultado"].drop_duplicates().tolist()
    selected_results = filter4.multiselect(
        "Resultado",
        result_options,
        default=result_options,
        key="net_load_filter_results",
    )

    shown = result.detail[
        result.detail["subsistema"].isin(selected_subsystems)
        & result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["patamar"].isin(selected_blocks)
        & result.detail["resultado"].isin(selected_results)
    ]
    visible_columns = [
        "subsistema",
        "bus_id",
        "stage_id",
        "periodo",
        "patamar",
        "block_id",
        "mean_mw",
        "fator_carga",
        "carga_bruta_patamar_mw",
        "quantidade_fontes",
        "geracao_nao_simulada_patamar_mw",
        "carga_liquida_cobre_calculada_mw",
        "carga_liquida_pmo_mw",
        "diferenca_mw",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=580,
        column_config={
            "subsistema": "Subsistema",
            "bus_id": "bus_id COBRE",
            "stage_id": "Estágio",
            "periodo": "Período",
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "mean_mw": st.column_config.NumberColumn(
                "Carga média COBRE (MW)", format="%.4f"
            ),
            "fator_carga": st.column_config.NumberColumn(
                "Fator da carga", format="%.6f"
            ),
            "carga_bruta_patamar_mw": st.column_config.NumberColumn(
                "Carga bruta no patamar (MW)", format="%.4f"
            ),
            "quantidade_fontes": "Número de fontes",
            "geracao_nao_simulada_patamar_mw": st.column_config.NumberColumn(
                "Geração não simulada (MW)", format="%.4f"
            ),
            "carga_liquida_cobre_calculada_mw": st.column_config.NumberColumn(
                "Carga líquida COBRE calculada (MW)", format="%.4f"
            ),
            "carga_liquida_pmo_mw": st.column_config.NumberColumn(
                "Carga líquida PMO (MW)", format="%.4f"
            ),
            "diferenca_mw": st.column_config.NumberColumn(
                "Diferença (MW)", format="%.6f"
            ),
            "resultado": "Resultado",
        },
    )

    st.download_button(
        "Baixar relatório da carga líquida em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_validacao_carga_liquida.csv",
        mime="text/csv",
        key="download_net_load_report",
    )


# =============================================================================
# ABA 6 - INTERFACE DA DURACAO DOS PATAMARES EM PU
# Palavra-chave para localizar a tela: ABA 6 INTERFACE DURACAO DOS PATAMARES
# =============================================================================
def _render_block_duration_validation() -> None:
    st.markdown(
        """
        <div class="formula-card">
          <div class="label">VALIDAÇÃO 6 — DURAÇÃO DOS PATAMARES</div>
          <div class="formula">
            Duração COBRE (pu) = horas do bloco ÷ Σ horas dos blocos do estágio
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="explain-card">
          <b>O que está sendo comparado?</b><br>
          O COBRE representa cada mês com seu real número de horas, diferentemente do NEWAVE
          em que todos os estágios do problema são representados com 730 horas, contudo, a 
          duração relativa de cada patamar deve ser a mesma.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload1, upload2 = st.columns(2)
    patamar_file = upload1.file_uploader(
        "PATAMAR.DAT", type=["dat", "txt"], key="duration_patamar"
    )
    stages_file = upload2.file_uploader(
        "stages.json", type=["json"], key="duration_stages"
    )
    tolerance = st.number_input(
        "Tolerância da comparação das durações em pu",
        min_value=0.0,
        value=0.000001,
        step=0.000001,
        format="%.10f",
        key="duration_tolerance",
        help=(
            "A diferença absoluta será considerada correta quando for menor "
            "ou igual à tolerância escolhida. Com tolerância zero, somente "
            "valores exatamente iguais serão aprovados."
        ),
    )

    if patamar_file is None or stages_file is None:
        st.info("Carregue o PATAMAR.DAT e o stages.json.")
        return

    try:
        with st.spinner("Normalizando as horas e comparando as durações..."):
            patamar_bytes = patamar_file.getvalue()
            stages_bytes = stages_file.getvalue()
            patamar, cobre_durations = _load_block_duration_sources(
                patamar_bytes, stages_bytes
            )
            result = build_block_duration_comparison(
                patamar=patamar,
                cobre_durations=cobre_durations,
                tolerance=float(tolerance),
            )
            case_key = hashlib.sha256(
                patamar_bytes + stages_bytes
            ).hexdigest()[:12]
    except Exception as error:
        st.error(f"Não foi possível validar as durações dos patamares: {error}")
        return

    st.caption(
        f"Origem: **PATAMAR.DAT + stages.json** · Caso: {case_key}"
    )
    total = len(result.detail)
    correct = int(result.detail["resultado"].eq("OK").sum())
    divergences = total - correct
    differences = result.detail["diferenca_pu"].abs().dropna()
    max_difference = float(differences.max()) if not differences.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Durações comparadas", total)
    col2.metric("Iguais", correct)
    col3.metric("Divergências", divergences)
    col4.metric("Maior diferença", _format_pt(max_difference, 16))

    if divergences == 0:
        st.success(
            "Validação aprovada: as durações normalizadas dos patamares do "
            "COBRE coincidem com as durações do NEWAVE dentro da tolerância."
        )
    else:
        st.warning(
            f"Foram encontradas {divergences} durações divergentes ou ausentes."
        )

    with st.expander("Correspondência e normalização usadas na comparação"):
        correspondence = (
            result.detail[
                ["patamar", "block_id", "nome_bloco_cobre"]
            ]
            .drop_duplicates(["patamar", "block_id"])
            .sort_values("block_id")
        )
        st.dataframe(
            correspondence,
            hide_index=True,
            use_container_width=True,
            column_config={
                "patamar": "Patamar NEWAVE",
                "block_id": "block_id COBRE",
                "nome_bloco_cobre": "Nome COBRE (informativo)",
            },
        )
        st.caption(
            "A normalização é feita separadamente em cada stage_id. Nos "
            "estágios posteriores ao último ano declarado no PATAMAR.DAT, "
            "repete-se sazonalmente o perfil mensal desse último ano."
        )

    st.subheader("Resumo por patamar")
    st.dataframe(
        result.summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "valores": "Valores",
            "valores_ok": "Iguais",
            "divergencias": "Divergências",
            "maior_diferenca_abs_pu": st.column_config.NumberColumn(
                "Maior diferença (pu)", format="%.16f"
            ),
            "resultado": "Resultado",
        },
    )

    st.subheader("Comprovação por estágio e patamar")
    filter1, filter2, filter3 = st.columns(3)
    period_options = (
        result.detail["tipo_periodo"].dropna().drop_duplicates().tolist()
    )
    selected_periods = filter1.multiselect(
        "Tipo de período",
        period_options,
        default=period_options,
        key="duration_filter_periods",
    )
    block_options = sorted(
        int(value) for value in result.detail["patamar"].dropna().unique()
    )
    selected_blocks = filter2.multiselect(
        "Patamares",
        block_options,
        default=block_options,
        key="duration_filter_blocks",
    )
    result_options = result.detail["resultado"].drop_duplicates().tolist()
    selected_results = filter3.multiselect(
        "Resultado",
        result_options,
        default=result_options,
        key="duration_filter_results",
    )

    shown = result.detail[
        result.detail["tipo_periodo"].isin(selected_periods)
        & result.detail["patamar"].isin(selected_blocks)
        & result.detail["resultado"].isin(selected_results)
    ]
    visible_columns = [
        "stage_id",
        "periodo",
        "tipo_periodo",
        "patamar",
        "block_id",
        "nome_bloco_cobre",
        "horas_cobre",
        "total_horas_estagio",
        "duracao_newave_pu",
        "duracao_cobre_pu",
        "diferenca_pu",
        "resultado",
    ]
    st.dataframe(
        shown[visible_columns],
        hide_index=True,
        use_container_width=True,
        height=580,
        column_config={
            "stage_id": "Estágio",
            "periodo": "Período",
            "tipo_periodo": "Tipo de período",
            "patamar": "Patamar NEWAVE",
            "block_id": "block_id COBRE",
            "nome_bloco_cobre": "Nome COBRE",
            "horas_cobre": st.column_config.NumberColumn(
                "Horas do bloco COBRE", format="%.6f"
            ),
            "total_horas_estagio": st.column_config.NumberColumn(
                "Total de horas do estágio", format="%.6f"
            ),
            "duracao_newave_pu": st.column_config.NumberColumn(
                "Duração NEWAVE (pu)", format="%.8f"
            ),
            "duracao_cobre_pu": st.column_config.NumberColumn(
                "Duração COBRE normalizada (pu)", format="%.8f"
            ),
            "diferenca_pu": st.column_config.NumberColumn(
                "Diferença (pu)", format="%.16f"
            ),
            "resultado": "Resultado",
        },
    )

    st.download_button(
        "Baixar relatório das durações em CSV",
        data=result.detail.to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig"),
        file_name="relatorio_validacao_duracao_patamares.csv",
        mime="text/csv",
        key="download_duration_report",
    )


(
    load_tab,
    factor_tab,
    non_controllable_tab,
    non_controllable_factor_tab,
    net_load_tab,
    block_duration_tab,
) = st.tabs(
    [
        "1. Formação da carga bruta",
        "2. Fatores por patamar",
        "3. Geração não controlada",
        "4. Fatores da geração não controlada",
        "5. Carga líquida por patamar",
        "6. Duração dos patamares",
    ]
)

with load_tab:
    _render_load_validation()

with factor_tab:
    _render_factor_validation()

with non_controllable_tab:
    _render_non_controllable_validation()

with non_controllable_factor_tab:
    _render_non_controllable_factor_validation()

with net_load_tab:
    _render_net_load_validation()

with block_duration_tab:
    _render_block_duration_validation()
