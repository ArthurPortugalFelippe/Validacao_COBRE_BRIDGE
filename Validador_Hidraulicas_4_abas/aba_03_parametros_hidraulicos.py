"""Aba 3 — parâmetros hidráulicos do cadastro NEWAVE → COBRE.

Compara os valores apresentados no pmo.dat com os campos correspondentes do
system/hydros.json. O VOLREF_SAZ.DAT é opcional e, quando fornecido, permite
validar os volumes mensais de referência usados na evaporação.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from aba_01_usinas_hidraulicas import (
    ErroLeitura,
    decodificar_texto,
    gerar_csv,
    ler_pmo,
    normalizar_nome,
    validar_usinas,
)


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

NUMERO = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
RE_PARAMETROS = re.compile(
    rf"^\s*(?P<nome>.+?)\s+"
    rf"(?P<posto>\d+)\s+"
    rf"(?P<vmin>{NUMERO})\s+"
    rf"(?P<vmax>{NUMERO})\s+"
    rf"(?P<vref>{NUMERO})\s+"
    rf"(?P<pinst>{NUMERO})\s+"
    rf"(?P<prod_esp>{NUMERO})\s+"
    rf"(?P<perda>{NUMERO})\s+"
    rf"(?P<unidade_perda>m|%)\s+"
    rf"(?P<cfmed>{NUMERO})\s+"
    rf"(?P<qmin_ignorado>{NUMERO})\s+"
    rf"(?P<teif>{NUMERO})\s+"
    rf"(?P<ip>{NUMERO})\s+"
    rf"(?P<regul>\S+)\s+"
    rf"(?P<nummaq>\d+)\s*$",
    flags=re.IGNORECASE,
)

RE_MAQUINA_EXPANSAO = re.compile(
    rf"^\s*(?P<mes>\d{{1,2}})\s+(?P<ano>\d{{4}})\s+"
    rf"(?P<conjunto>\d+)\s+(?P<maquina>\d+)\s+"
    rf"(?P<potencia>{NUMERO})\s*$"
)
RE_USINA_EXPANSAO = re.compile(
    r"^\s*(?P<nome>[A-ZÀ-Ü][A-ZÀ-Ü0-9 ._-]*?)\s+"
    r"(?P<mes>\d{1,2})\s+(?P<ano>\d{1,4})\s+"
    r"(?P<duracao>\d+)\s*$"
)


@dataclass(frozen=True)
class ResultadoParametros:
    """Tabelas e indicadores da validação dos parâmetros hidráulicos."""

    comparacao: pd.DataFrame
    divergencias: pd.DataFrame
    resumo_variaveis: pd.DataFrame
    total_usinas_pmo: int
    total_usinas_comparadas: int
    total_comparacoes: int
    total_coincidentes: int
    total_divergencias: int
    total_usinas_divergentes: int
    volref_validado: bool
    aprovada: bool
    avisos: tuple[str, ...]


def ler_parametros_cadastro_pmo(conteudo: bytes) -> pd.DataFrame:
    """Lê a tabela principal DADOS DAS USINAS HIDROELETRICAS."""
    linhas = decodificar_texto(conteudo).splitlines()
    dentro = False
    registros: list[dict[str, Any]] = []

    for linha in linhas:
        linha_maiuscula = linha.upper()
        if dentro and "POLINOMIO VOLUME-COTA" in linha_maiuscula:
            break
        if (
            "NOME" in linha_maiuscula
            and "POSTO" in linha_maiuscula
            and "PROD.ESP." in linha_maiuscula
            and "QMIN" in linha_maiuscula
        ):
            dentro = True
            continue
        if not dentro:
            continue

        match = RE_PARAMETROS.match(linha)
        if not match:
            continue
        item = match.groupdict()
        registros.append(
            {
                "nome_normalizado": normalizar_nome(item["nome"]),
                "nome_pmo": item["nome"].strip(),
                "posto": int(item["posto"]),
                "vmin_hm3": float(item["vmin"]),
                "vmax_hm3": float(item["vmax"]),
                "vref_hm3": float(item["vref"]),
                "pinst_base_mw": float(item["pinst"]),
                "prod_esp": float(item["prod_esp"]),
                "perda_hidraulica": float(item["perda"]),
                "unidade_perda": item["unidade_perda"].lower(),
                "cfmed_m": float(item["cfmed"]),
                "teif_percentual": float(item["teif"]),
                "ip_percentual": float(item["ip"]),
                "regulacao": item["regul"].upper(),
                "numero_maquinas_base": int(item["nummaq"]),
            }
        )

    if not registros:
        raise ErroLeitura(
            "Não foi encontrada a tabela DADOS DAS USINAS HIDROELETRICAS "
            "com as colunas VMIN, VMAX, PINST, PROD.ESP., P.HID. e CFMED."
        )

    df = pd.DataFrame(registros).drop_duplicates(
        subset=["nome_normalizado", "posto"], keep="first"
    )
    return df


def ler_evaporacao_pmo(conteudo: bytes) -> dict[int, list[float]]:
    """Lê os doze coeficientes mensais de evaporação por código NEWAVE."""
    linhas = decodificar_texto(conteudo).splitlines()
    dentro = False
    ja_leu_registro = False
    resultado: dict[int, list[float]] = {}

    for linha in linhas:
        linha_maiuscula = linha.upper()
        if "COEFICIENTES DE EVAPORACAO" in linha_maiuscula:
            dentro = True
            ja_leu_registro = False
            continue
        if not dentro:
            continue
        if ja_leu_registro and linha.strip().startswith("X---"):
            dentro = False
            continue

        partes = linha.split()
        if len(partes) < 14 or not partes[0].isdigit():
            continue
        try:
            valores = [float(valor) for valor in partes[-12:]]
        except ValueError:
            continue
        codigo = int(partes[0])
        resultado[codigo] = valores
        ja_leu_registro = True

    if not resultado:
        raise ErroLeitura(
            "Não foi encontrada a tabela COEFICIENTES DE EVAPORACAO no pmo.dat."
        )
    return resultado


def ler_inicio_estudo_pmo(conteudo: bytes) -> tuple[int, int]:
    """Lê o mês e o ano inicial do horizonte de estudo."""
    texto = decodificar_texto(conteudo)
    mes = re.search(
        r"MES INICIAL DO PERIODO DE ESTUDO\s+(\d{1,2})", texto, re.IGNORECASE
    )
    ano = re.search(
        r"ANO INICIAL DO PERIODO DE ESTUDO\s+(\d{4})", texto, re.IGNORECASE
    )
    if mes is None or ano is None:
        raise ErroLeitura(
            "Não foi possível localizar o mês e o ano iniciais do estudo no pmo.dat."
        )
    mes_valor = int(mes.group(1))
    ano_valor = int(ano.group(1))
    if not 1 <= mes_valor <= 12:
        raise ErroLeitura(f"O mês inicial do estudo é inválido: {mes_valor}.")
    return mes_valor, ano_valor


def ler_expansao_pmo(conteudo: bytes) -> dict[str, dict[str, Any]]:
    """Lê enchimento, entrada de máquinas e potência do cronograma de expansão."""
    linhas = decodificar_texto(conteudo).splitlines()
    dentro = False
    atual: dict[str, Any] | None = None
    resultado: dict[str, dict[str, Any]] = {}

    for linha in linhas:
        linha_maiuscula = linha.upper()
        if "C R O N O G R A M A  DE  E X P A N S A O" in linha_maiuscula:
            dentro = True
            continue
        if dentro and "SAZONALIZACAO" in linha_maiuscula:
            break
        if not dentro:
            continue

        maquina = RE_MAQUINA_EXPANSAO.match(linha)
        if maquina and atual is not None:
            item = maquina.groupdict()
            atual["maquinas"].append(
                {
                    "mes": int(item["mes"]),
                    "ano": int(item["ano"]),
                    "conjunto": int(item["conjunto"]),
                    "maquina": int(item["maquina"]),
                    "potencia_mw": float(item["potencia"]),
                }
            )
            continue

        usina = RE_USINA_EXPANSAO.match(linha)
        if not usina:
            continue
        item = usina.groupdict()
        nome_normalizado = normalizar_nome(item["nome"])
        atual = {
            "nome": item["nome"].strip(),
            "mes_inicio_enchimento": int(item["mes"]),
            "ano_inicio_enchimento": int(item["ano"]),
            "duracao_enchimento_meses": int(item["duracao"]),
            "maquinas": [],
        }
        resultado[nome_normalizado] = atual

    return resultado


def ler_volref_saz(conteudo: bytes) -> dict[int, dict[str, Any]]:
    """Lê o VOLREF_SAZ.DAT no formato de volume útil mensal em hm³."""
    linhas = decodificar_texto(conteudo).splitlines()
    resultado: dict[int, dict[str, Any]] = {}

    for numero_linha, linha in enumerate(linhas[3:], start=4):
        if not linha.strip():
            continue
        codigo_txt = linha[0:3].strip() if len(linha) >= 3 else ""
        if not codigo_txt.isdigit():
            continue
        codigo = int(codigo_txt)
        nome = linha[5:17].strip() if len(linha) >= 17 else ""
        valores: list[float] = []
        for indice in range(12):
            inicio = 19 + 10 * indice
            trecho = linha[inicio : inicio + 8].strip()
            if not trecho:
                break
            try:
                valores.append(float(trecho))
            except ValueError as exc:
                raise ErroLeitura(
                    f"Valor inválido no VOLREF_SAZ.DAT, linha {numero_linha}, "
                    f"mês {indice + 1}: {trecho!r}."
                ) from exc
        if len(valores) != 12:
            # Fallback para arquivos equivalentes separados por espaços.
            partes = linha.split()
            try:
                valores = [float(valor) for valor in partes[-12:]]
            except (ValueError, IndexError) as exc:
                raise ErroLeitura(
                    f"Não foi possível ler os 12 meses do VOLREF_SAZ.DAT "
                    f"na linha {numero_linha}."
                ) from exc
            if len(valores) != 12:
                raise ErroLeitura(
                    f"A linha {numero_linha} do VOLREF_SAZ.DAT não possui 12 meses."
                )
        resultado[codigo] = {"nome": nome, "volumes_uteis_hm3": valores}

    if not resultado:
        raise ErroLeitura("Nenhuma usina foi lida no VOLREF_SAZ.DAT.")
    return resultado


def ler_hydros_completo(conteudo: bytes) -> dict[int, dict[str, Any]]:
    """Lê os registros completos do hydros.json, indexados pelo ID COBRE."""
    try:
        objeto = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErroLeitura(f"Não foi possível ler o hydros.json: {exc}") from exc
    hydros = objeto.get("hydros") if isinstance(objeto, dict) else None
    if not isinstance(hydros, list):
        raise ErroLeitura("O hydros.json não possui uma lista válida no campo 'hydros'.")

    resultado: dict[int, dict[str, Any]] = {}
    for posicao, hydro in enumerate(hydros):
        if not isinstance(hydro, dict) or "id" not in hydro or "name" not in hydro:
            raise ErroLeitura(
                f"O registro {posicao} do hydros.json não possui os campos id e name."
            )
        identificador = int(hydro["id"])
        if identificador in resultado:
            raise ErroLeitura(f"O hydros.json possui o ID repetido {identificador}.")
        resultado[identificador] = hydro
    return resultado


def _campo_objeto(
    objeto: dict[str, Any], campo: str, contexto: str
) -> dict[str, Any]:
    valor = objeto.get(campo)
    if not isinstance(valor, dict):
        raise ErroLeitura(f"O campo {campo} de {contexto} não é um objeto válido.")
    return valor


def _numero_ou_none(valor: Any) -> float | None:
    if valor is None or isinstance(valor, bool):
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _id_estagio(mes: int, ano: int, mes_inicial: int, ano_inicial: int) -> int:
    return (ano - ano_inicial) * 12 + (mes - mes_inicial)


def _somar_meses(data_inicial: date, meses: int) -> date:
    indice = data_inicial.year * 12 + data_inicial.month - 1 + meses
    return date(indice // 12, indice % 12 + 1, 1)


def _adicionar_comparacao_numerica(
    linhas: list[dict[str, Any]],
    *,
    codigo: int,
    id_cobre: int,
    nome: str,
    grupo: str,
    variavel: str,
    periodo: str,
    unidade: str,
    valor_newave: float,
    valor_cobre: float | None,
    tolerancia: float,
    detalhe: str = "",
    tipo_esperado: str = "",
    tipo_cobre: str = "",
) -> None:
    diferenca = (
        abs(float(valor_cobre) - float(valor_newave))
        if valor_cobre is not None
        else None
    )
    tipos_coincidem = not tipo_esperado or tipo_esperado == tipo_cobre
    coincide = (
        diferenca is not None and diferenca <= tolerancia and tipos_coincidem
    )
    if valor_cobre is None:
        detalhe_final = "Campo correspondente ausente no hydros.json."
    elif not tipos_coincidem:
        detalhe_final = (
            f"Tipo esperado: {tipo_esperado}; tipo encontrado: "
            f"{tipo_cobre or 'ausente'}."
        )
    else:
        detalhe_final = detalhe

    linhas.append(
        {
            "codigo_newave": codigo,
            "id_cobre": id_cobre,
            "nome_usina": nome,
            "grupo": grupo,
            "variavel": variavel,
            "periodo": periodo,
            "unidade": unidade,
            "valor_newave": valor_newave,
            "valor_cobre": valor_cobre,
            "diferenca_absoluta": diferenca,
            "tolerancia": tolerancia,
            "tipo_esperado": tipo_esperado,
            "tipo_cobre": tipo_cobre,
            "resultado": "Coincidente" if coincide else "Divergente",
            "detalhe": detalhe_final,
        }
    )


def _adicionar_comparacao_texto(
    linhas: list[dict[str, Any]],
    *,
    codigo: int,
    id_cobre: int,
    nome: str,
    grupo: str,
    variavel: str,
    periodo: str,
    valor_newave: str,
    valor_cobre: str,
    detalhe: str = "",
) -> None:
    coincide = valor_newave == valor_cobre
    linhas.append(
        {
            "codigo_newave": codigo,
            "id_cobre": id_cobre,
            "nome_usina": nome,
            "grupo": grupo,
            "variavel": variavel,
            "periodo": periodo,
            "unidade": "",
            "valor_newave": valor_newave,
            "valor_cobre": valor_cobre,
            "diferenca_absoluta": None,
            "tolerancia": None,
            "tipo_esperado": "",
            "tipo_cobre": "",
            "resultado": "Coincidente" if coincide else "Divergente",
            "detalhe": detalhe,
        }
    )


def validar_parametros(
    pmo_bytes: bytes,
    hydros_bytes: bytes,
    volref_bytes: bytes | None,
    tolerancias: dict[str, float],
) -> ResultadoParametros:
    """Executa a validação dos parâmetros estáticos e do cronograma hidráulico."""
    identificacao = validar_usinas(pmo_bytes, hydros_bytes)
    cadastro_pmo, total_pmo, _ = ler_pmo(pmo_bytes)
    parametros = ler_parametros_cadastro_pmo(pmo_bytes)
    evaporacao = ler_evaporacao_pmo(pmo_bytes)
    expansao = ler_expansao_pmo(pmo_bytes)
    mes_inicial, ano_inicial = ler_inicio_estudo_pmo(pmo_bytes)
    hydros = ler_hydros_completo(hydros_bytes)
    volref = ler_volref_saz(volref_bytes) if volref_bytes is not None else None

    cadastro_base = cadastro_pmo[
        ["codigo_newave", "nome_usina", "nome_normalizado", "posto"]
    ]
    parametros = cadastro_base.merge(
        parametros,
        on=["nome_normalizado", "posto"],
        how="left",
        validate="one_to_one",
    )
    sem_parametros = parametros["vmin_hm3"].isna()
    if sem_parametros.any():
        faltantes = parametros.loc[
            sem_parametros, ["codigo_newave", "nome_usina"]
        ].to_dict("records")
        raise ErroLeitura(
            "Não foi possível associar a tabela de parâmetros a estas usinas: "
            f"{faltantes}."
        )

    id_por_codigo = {
        int(row["codigo_newave"]): int(row["id_cobre"])
        for _, row in identificacao.de_para.iterrows()
    }
    parametros = parametros[
        parametros["codigo_newave"].isin(id_por_codigo)
    ].copy()

    linhas: list[dict[str, Any]] = []
    avisos: list[str] = []
    ausentes = identificacao.ausentes
    if not ausentes.empty:
        avisos.append(
            f"{len(ausentes)} usina(s) esperada(s) não foram comparadas nesta aba "
            "porque não possuem registro no hydros.json. A ausência permanece "
            "evidenciada na Aba 1."
        )

    for _, row in parametros.sort_values("codigo_newave").iterrows():
        codigo = int(row["codigo_newave"])
        id_cobre = id_por_codigo[codigo]
        nome = str(row["nome_usina"])
        hydro = hydros.get(id_cobre)
        if hydro is None:
            raise ErroLeitura(
                f"O ID COBRE {id_cobre}, associado à usina {nome}, não foi encontrado."
            )
        contexto = f"usina {nome} (ID COBRE {id_cobre})"
        reservoir = _campo_objeto(hydro, "reservoir", contexto)
        generation = _campo_objeto(hydro, "generation", contexto)

        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Cadastro do reservatório",
            variavel="VMIN",
            periodo="",
            unidade="hm³",
            valor_newave=float(row["vmin_hm3"]),
            valor_cobre=_numero_ou_none(reservoir.get("min_storage_hm3")),
            tolerancia=tolerancias["volume_hm3"],
        )
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Cadastro do reservatório",
            variavel="VMAX",
            periodo="",
            unidade="hm³",
            valor_newave=float(row["vmax_hm3"]),
            valor_cobre=_numero_ou_none(reservoir.get("max_storage_hm3")),
            tolerancia=tolerancias["volume_hm3"],
        )

        expansao_usina = expansao.get(normalizar_nome(nome))
        potencia_expansao = (
            sum(float(m["potencia_mw"]) for m in expansao_usina["maquinas"])
            if expansao_usina is not None
            else 0.0
        )
        potencia_esperada = float(row["pinst_base_mw"]) + potencia_expansao
        detalhe_potencia = (
            f"PINST base {float(row['pinst_base_mw']):g} MW + "
            f"{potencia_expansao:g} MW do cronograma de expansão."
            if potencia_expansao
            else "PINST da tabela de cadastro do PMO."
        )
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Potência",
            variavel="Potência instalada final",
            periodo="",
            unidade="MW",
            valor_newave=potencia_esperada,
            valor_cobre=_numero_ou_none(generation.get("max_generation_mw")),
            tolerancia=tolerancias["potencia_mw"],
            detalhe=detalhe_potencia,
        )
        grupos = hydro.get("unit_groups")
        potencia_grupos: float | None = None
        if isinstance(grupos, list) and all(isinstance(g, dict) for g in grupos):
            valores_grupos = [
                _numero_ou_none(g.get("max_generation_mw")) for g in grupos
            ]
            if all(valor is not None for valor in valores_grupos):
                potencia_grupos = sum(float(valor) for valor in valores_grupos)
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Potência",
            variavel="Soma da potência dos grupos",
            periodo="",
            unidade="MW",
            valor_newave=potencia_esperada,
            valor_cobre=potencia_grupos,
            tolerancia=tolerancias["potencia_mw"],
            detalhe=detalhe_potencia,
        )

        prod_cobre = _numero_ou_none(
            hydro.get("specific_productivity_mw_per_m3s_per_m")
        )
        if float(row["prod_esp"]) == 0.0 and prod_cobre is None:
            prod_cobre = 0.0
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Produtividade e perdas",
            variavel="PROD.ESP.",
            periodo="",
            unidade="MW/(m³/s·m)",
            valor_newave=float(row["prod_esp"]),
            valor_cobre=prod_cobre,
            tolerancia=tolerancias["produtividade"],
            detalhe="Valor nulo no COBRE é tratado como zero quando o PMO informa zero.",
        )

        perda_pmo = float(row["perda_hidraulica"])
        unidade_perda = str(row["unidade_perda"])
        perda_objeto = hydro.get("hydraulic_losses")
        tipo_esperado = (
            "ausente"
            if perda_pmo == 0.0
            else "constant" if unidade_perda == "m" else "factor"
        )
        tipo_cobre = "ausente"
        perda_cobre: float | None = 0.0 if perda_pmo == 0.0 else None
        if isinstance(perda_objeto, dict):
            tipo_cobre = str(perda_objeto.get("type", ""))
            if tipo_cobre == "constant":
                perda_cobre = _numero_ou_none(perda_objeto.get("value_m"))
            elif tipo_cobre == "factor":
                fator = _numero_ou_none(perda_objeto.get("value"))
                perda_cobre = fator * 100.0 if fator is not None else None
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Produtividade e perdas",
            variavel="P.HID.",
            periodo="",
            unidade="m" if unidade_perda == "m" else "%",
            valor_newave=perda_pmo,
            valor_cobre=perda_cobre,
            tolerancia=tolerancias["perda_cfmed"],
            tipo_esperado=tipo_esperado,
            tipo_cobre=tipo_cobre,
        )

        tailrace = hydro.get("tailrace")
        cfmed_cobre: float | None = None
        if isinstance(tailrace, dict):
            coeficientes = tailrace.get("coefficients")
            if isinstance(coeficientes, list) and coeficientes:
                cfmed_cobre = _numero_ou_none(coeficientes[0])
        if float(row["cfmed_m"]) == 0.0 and cfmed_cobre is None:
            cfmed_cobre = 0.0
        _adicionar_comparacao_numerica(
            linhas,
            codigo=codigo,
            id_cobre=id_cobre,
            nome=nome,
            grupo="Canal de fuga",
            variavel="CFMED",
            periodo="",
            unidade="m",
            valor_newave=float(row["cfmed_m"]),
            valor_cobre=cfmed_cobre,
            tolerancia=tolerancias["perda_cfmed"],
        )
        coef_pmo = evaporacao.get(codigo)
        if coef_pmo is None:
            raise ErroLeitura(
                f"A usina {nome} (código {codigo}) não foi localizada na tabela "
                "de coeficientes de evaporação do PMO."
            )
        evap_objeto = hydro.get("evaporation")
        coef_cobre: list[float]
        if evap_objeto is None:
            coef_cobre = [0.0] * 12
        elif isinstance(evap_objeto, dict) and isinstance(
            evap_objeto.get("coefficients_mm"), list
        ):
            coef_cobre = [float(valor) for valor in evap_objeto["coefficients_mm"]]
            if len(coef_cobre) != 12:
                raise ErroLeitura(
                    f"A usina {nome} possui {len(coef_cobre)} coeficientes de "
                    "evaporação no hydros.json; eram esperados 12."
                )
        else:
            raise ErroLeitura(
                f"O bloco evaporation da usina {nome} possui estrutura inválida."
            )
        for indice, mes in enumerate(MESES):
            _adicionar_comparacao_numerica(
                linhas,
                codigo=codigo,
                id_cobre=id_cobre,
                nome=nome,
                grupo="Evaporação",
                variavel="Coeficiente de evaporação",
                periodo=mes,
                unidade="mm/mês",
                valor_newave=float(coef_pmo[indice]),
                valor_cobre=float(coef_cobre[indice]),
                tolerancia=tolerancias["evaporacao_mm"],
                detalhe=(
                    "Bloco ausente no COBRE é tratado como doze valores zero."
                    if evap_objeto is None
                    else ""
                ),
            )

        if expansao_usina is not None:
            maquinas = expansao_usina["maquinas"]
            if float(row["pinst_base_mw"]) == 0.0 and maquinas:
                primeira = min(maquinas, key=lambda item: (item["ano"], item["mes"]))
                data_entrada = date(int(primeira["ano"]), int(primeira["mes"]), 1)
                data_cobre = str(hydro.get("operational_start_date") or "")
                _adicionar_comparacao_texto(
                    linhas,
                    codigo=codigo,
                    id_cobre=id_cobre,
                    nome=nome,
                    grupo="Expansão",
                    variavel="Data de entrada em operação",
                    periodo="",
                    valor_newave=data_entrada.isoformat(),
                    valor_cobre=data_cobre,
                )
                entrada_esperada = _id_estagio(
                    data_entrada.month,
                    data_entrada.year,
                    mes_inicial,
                    ano_inicial,
                )
                _adicionar_comparacao_numerica(
                    linhas,
                    codigo=codigo,
                    id_cobre=id_cobre,
                    nome=nome,
                    grupo="Expansão",
                    variavel="Estágio de entrada",
                    periodo="",
                    unidade="ID de estágio",
                    valor_newave=float(entrada_esperada),
                    valor_cobre=_numero_ou_none(hydro.get("entry_stage_id")),
                    tolerancia=0.0,
                )

            mes_enchimento = int(expansao_usina["mes_inicio_enchimento"])
            ano_enchimento = int(expansao_usina["ano_inicio_enchimento"])
            duracao = int(expansao_usina["duracao_enchimento_meses"])
            if mes_enchimento > 0 and ano_enchimento > 0 and duracao > 0:
                filling = hydro.get("filling")
                filling_obj = filling if isinstance(filling, dict) else {}
                inicio_enchimento = date(ano_enchimento, mes_enchimento, 1)
                fim_enchimento = _somar_meses(inicio_enchimento, duracao)
                estagio_enchimento = _id_estagio(
                    mes_enchimento,
                    ano_enchimento,
                    mes_inicial,
                    ano_inicial,
                )
                _adicionar_comparacao_numerica(
                    linhas,
                    codigo=codigo,
                    id_cobre=id_cobre,
                    nome=nome,
                    grupo="Expansão",
                    variavel="Estágio inicial do enchimento",
                    periodo="",
                    unidade="ID de estágio",
                    valor_newave=float(estagio_enchimento),
                    valor_cobre=_numero_ou_none(filling_obj.get("start_stage_id")),
                    tolerancia=0.0,
                )
                segundos = (fim_enchimento - inicio_enchimento).days * 86400.0
                vazao_enchimento = float(row["vmin_hm3"]) * 1_000_000.0 / segundos
                _adicionar_comparacao_numerica(
                    linhas,
                    codigo=codigo,
                    id_cobre=id_cobre,
                    nome=nome,
                    grupo="Expansão",
                    variavel="Vazão mínima de enchimento",
                    periodo="",
                    unidade="m³/s",
                    valor_newave=vazao_enchimento,
                    valor_cobre=_numero_ou_none(
                        filling_obj.get("filling_min_rate_m3s")
                    ),
                    tolerancia=tolerancias["vazao_enchimento_m3s"],
                    detalhe=(
                        "Calculada como VMIN dividido pela duração exata do "
                        "enchimento em segundos."
                    ),
                )

        if volref is not None:
            referencia_cobre: list[float] | None = None
            if isinstance(evap_objeto, dict) and isinstance(
                evap_objeto.get("reference_volumes_hm3"), list
            ):
                referencia_cobre = [
                    float(valor) for valor in evap_objeto["reference_volumes_hm3"]
                ]
                if len(referencia_cobre) != 12:
                    raise ErroLeitura(
                        f"A usina {nome} possui {len(referencia_cobre)} volumes "
                        "sazonais no hydros.json; eram esperados 12."
                    )

            registro_volref = volref.get(codigo)
            if registro_volref is None:
                if referencia_cobre is not None:
                    _adicionar_comparacao_texto(
                        linhas,
                        codigo=codigo,
                        id_cobre=id_cobre,
                        nome=nome,
                        grupo="Volume de referência sazonal",
                        variavel="Presença do volume sazonal",
                        periodo="",
                        valor_newave="Ausente",
                        valor_cobre="Presente",
                        detalhe="A usina não possui linha no VOLREF_SAZ.DAT.",
                    )
            else:
                uteis = [float(valor) for valor in registro_volref["volumes_uteis_hm3"]]
                linha_zerada = all(valor == 0.0 for valor in uteis)
                tem_evaporacao = any(float(valor) != 0.0 for valor in coef_pmo)
                if linha_zerada or not tem_evaporacao:
                    _adicionar_comparacao_texto(
                        linhas,
                        codigo=codigo,
                        id_cobre=id_cobre,
                        nome=nome,
                        grupo="Volume de referência sazonal",
                        variavel="Representação do volume sazonal",
                        periodo="",
                        valor_newave="Campo omitido",
                        valor_cobre=(
                            "Campo omitido"
                            if referencia_cobre is None
                            else "Campo presente"
                        ),
                        detalhe=(
                            "Os 12 volumes úteis são zero; nessa situação o "
                            "conversor deve omitir reference_volumes_hm3."
                            if linha_zerada
                            else "A usina não possui coeficientes de evaporação ativos."
                        ),
                    )
                else:
                    for indice, mes in enumerate(MESES):
                        esperado = max(
                            float(row["vmin_hm3"]),
                            min(
                                float(row["vmax_hm3"]),
                                float(row["vmin_hm3"]) + uteis[indice],
                            ),
                        )
                        encontrado = (
                            referencia_cobre[indice]
                            if referencia_cobre is not None
                            else None
                        )
                        _adicionar_comparacao_numerica(
                            linhas,
                            codigo=codigo,
                            id_cobre=id_cobre,
                            nome=nome,
                            grupo="Volume de referência sazonal",
                            variavel="Volume de referência sazonal",
                            periodo=mes,
                            unidade="hm³",
                            valor_newave=esperado,
                            valor_cobre=encontrado,
                            tolerancia=tolerancias["volref_hm3"],
                            detalhe=(
                                "Esperado = VMIN + volume útil mensal, limitado "
                                "ao intervalo [VMIN, VMAX]."
                            ),
                        )

    comparacao = pd.DataFrame(linhas)
    if comparacao.empty:
        raise ErroLeitura("Nenhuma comparação de parâmetros foi produzida.")
    comparacao = comparacao.sort_values(
        ["codigo_newave", "grupo", "variavel", "periodo"], ignore_index=True
    )
    divergencias = comparacao[comparacao["resultado"] == "Divergente"].copy()
    resumo = (
        comparacao.assign(
            coincidentes=comparacao["resultado"].eq("Coincidente").astype(int),
            divergencias=comparacao["resultado"].eq("Divergente").astype(int),
        )
        .groupby(["grupo", "variavel"], as_index=False)
        .agg(
            comparacoes=("resultado", "size"),
            coincidentes=("coincidentes", "sum"),
            divergencias=("divergencias", "sum"),
            maior_diferenca=("diferenca_absoluta", "max"),
        )
    )
    total_divergencias = len(divergencias)
    return ResultadoParametros(
        comparacao=comparacao,
        divergencias=divergencias,
        resumo_variaveis=resumo,
        total_usinas_pmo=total_pmo,
        total_usinas_comparadas=parametros["codigo_newave"].nunique(),
        total_comparacoes=len(comparacao),
        total_coincidentes=int((comparacao["resultado"] == "Coincidente").sum()),
        total_divergencias=total_divergencias,
        total_usinas_divergentes=divergencias["codigo_newave"].nunique(),
        volref_validado=volref is not None,
        aprovada=total_divergencias == 0,
        avisos=tuple(avisos),
    )


def mostrar_aba() -> None:
    """Renderiza a Aba 3 no Streamlit."""
    import streamlit as st

    st.subheader("Parâmetros hidráulicos")
    st.markdown(
        "Esta aba compara os dados de cadastro, evaporação e expansão do "
        "`pmo.dat` com os campos correspondentes do `system/hydros.json`."
    )
    st.caption(
        "O VOLREF_SAZ.DAT é opcional. Quando enviado, também são validados "
        "os 12 volumes mensais de referência da evaporação."
    )

    coluna_pmo, coluna_cobre, coluna_volref = st.columns(3)
    with coluna_pmo:
        arquivo_pmo = st.file_uploader(
            "Arquivo NEWAVE — pmo.dat",
            type=["dat"],
            key="aba03_pmo",
        )
    with coluna_cobre:
        arquivo_hydros = st.file_uploader(
            "Arquivo COBRE — hydros.json",
            type=["json"],
            key="aba03_hydros",
        )
    with coluna_volref:
        arquivo_volref = st.file_uploader(
            "Arquivo NEWAVE — VOLREF_SAZ.DAT (opcional)",
            type=["dat"],
            key="aba03_volref",
        )

    with st.expander("Tolerâncias da comparação", expanded=False):
        st.caption(
            "Uma diferença menor ou igual à tolerância é considerada coincidente."
        )
        linha_1 = st.columns(4)
        volume_hm3 = linha_1[0].number_input(
            "Volumes de cadastro (hm³)",
            min_value=0.0,
            value=0.51,
            step=0.01,
            format="%.4f",
            key="aba03_tol_volume",
        )
        potencia_mw = linha_1[1].number_input(
            "Potência (MW)",
            min_value=0.0,
            value=0.15,
            step=0.01,
            format="%.4f",
            key="aba03_tol_potencia",
        )
        produtividade = linha_1[2].number_input(
            "Produtividade específica",
            min_value=0.0,
            value=0.000001,
            step=0.000001,
            format="%.8f",
            key="aba03_tol_prod",
        )
        perda_cfmed = linha_1[3].number_input(
            "P.HID. e CFMED",
            min_value=0.0,
            value=0.051,
            step=0.001,
            format="%.4f",
            key="aba03_tol_perda",
        )
        linha_2 = st.columns(4)
        evaporacao_mm = linha_2[0].number_input(
            "Evaporação (mm/mês)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.4f",
            key="aba03_tol_evap",
        )
        volref_hm3 = linha_2[1].number_input(
            "Volume sazonal (hm³)",
            min_value=0.0,
            value=0.51,
            step=0.01,
            format="%.4f",
            key="aba03_tol_volref",
        )
        vazao_enchimento = linha_2[2].number_input(
            "Vazão de enchimento (m³/s)",
            min_value=0.0,
            value=0.10,
            step=0.01,
            format="%.4f",
            key="aba03_tol_enchimento",
        )

    if arquivo_pmo is None or arquivo_hydros is None:
        st.info("Envie o pmo.dat e o hydros.json para iniciar a validação.")
        return

    tolerancias = {
        "volume_hm3": float(volume_hm3),
        "potencia_mw": float(potencia_mw),
        "produtividade": float(produtividade),
        "perda_cfmed": float(perda_cfmed),
        "evaporacao_mm": float(evaporacao_mm),
        "volref_hm3": float(volref_hm3),
        "vazao_enchimento_m3s": float(vazao_enchimento),
    }
    pmo_bytes = arquivo_pmo.getvalue()
    hydros_bytes = arquivo_hydros.getvalue()
    volref_bytes = arquivo_volref.getvalue() if arquivo_volref is not None else None

    hash_arquivos = hashlib.sha256()
    hash_arquivos.update(pmo_bytes)
    hash_arquivos.update(b"\0PARAMETROS_HYDROS\0")
    hash_arquivos.update(hydros_bytes)
    if volref_bytes is not None:
        hash_arquivos.update(b"\0VOLREF\0")
        hash_arquivos.update(volref_bytes)
    hash_arquivos.update(repr(sorted(tolerancias.items())).encode("utf-8"))
    assinatura = hash_arquivos.hexdigest()
    chave_cache = "aba03_resultado_cache"

    executar = st.button(
        "Executar validação dos parâmetros",
        type="primary",
        key="aba03_executar",
    )
    if executar:
        try:
            with st.spinner("Lendo e comparando os parâmetros hidráulicos..."):
                resultado_calculado = validar_parametros(
                    pmo_bytes,
                    hydros_bytes,
                    volref_bytes,
                    tolerancias,
                )
            st.session_state[chave_cache] = {
                "assinatura": assinatura,
                "resultado": resultado_calculado,
            }
        except ErroLeitura as exc:
            st.session_state.pop(chave_cache, None)
            st.error(f"Não foi possível executar a validação: {exc}")
            return
        except Exception as exc:
            st.session_state.pop(chave_cache, None)
            st.error(f"Ocorreu um erro inesperado durante a validação: {exc}")
            return

    cache = st.session_state.get(chave_cache)
    if not cache or cache.get("assinatura") != assinatura:
        st.info("Clique em Executar validação dos parâmetros para comparar os arquivos.")
        return
    resultado: ResultadoParametros = cache["resultado"]

    for aviso in resultado.avisos:
        st.warning(aviso)
    if not resultado.volref_validado:
        st.info(
            "Volume de referência sazonal não validado: envie o "
            "VOLREF_SAZ.DAT para habilitar essa comparação."
        )

    metricas = st.columns(6)
    metricas[0].metric("Usinas no PMO", resultado.total_usinas_pmo)
    metricas[1].metric("Usinas comparadas", resultado.total_usinas_comparadas)
    metricas[2].metric("Comparações", resultado.total_comparacoes)
    metricas[3].metric("Coincidentes", resultado.total_coincidentes)
    metricas[4].metric("Divergências", resultado.total_divergencias)
    metricas[5].metric("Usinas divergentes", resultado.total_usinas_divergentes)

    if resultado.aprovada:
        st.success(
            "Validação aprovada: todos os parâmetros comparados coincidem "
            "dentro das tolerâncias escolhidas."
        )
    else:
        st.error(
            "Validação reprovada: existem parâmetros diferentes entre o "
            "NEWAVE e o hydros.json."
        )

    st.markdown("#### Resumo por variável")
    st.dataframe(
        resultado.resumo_variaveis,
        hide_index=True,
        use_container_width=True,
    )

    if not resultado.divergencias.empty:
        st.markdown("#### Diferenças identificadas")
        st.dataframe(
            resultado.divergencias,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Comparação completa")
    filtro_resultado, filtro_grupo = st.columns(2)
    escolha_resultado = filtro_resultado.selectbox(
        "Resultado",
        ["Somente divergências", "Todas", "Somente coincidentes"],
        key="aba03_filtro_resultado",
    )
    grupos = ["Todos"] + sorted(resultado.comparacao["grupo"].unique().tolist())
    escolha_grupo = filtro_grupo.selectbox(
        "Grupo de variáveis",
        grupos,
        key="aba03_filtro_grupo",
    )
    exibicao = resultado.comparacao
    if escolha_resultado == "Somente divergências":
        exibicao = exibicao[exibicao["resultado"] == "Divergente"]
    elif escolha_resultado == "Somente coincidentes":
        exibicao = exibicao[exibicao["resultado"] == "Coincidente"]
    if escolha_grupo != "Todos":
        exibicao = exibicao[exibicao["grupo"] == escolha_grupo]
    st.dataframe(exibicao, hide_index=True, use_container_width=True)

    st.markdown("#### Exportações")
    download_1, download_2, download_3 = st.columns(3)
    with download_1:
        st.download_button(
            "Baixar comparação completa",
            data=gerar_csv(resultado.comparacao),
            file_name="validacao_parametros_hidraulicos.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_2:
        st.download_button(
            "Baixar somente divergências",
            data=gerar_csv(resultado.divergencias),
            file_name="divergencias_parametros_hidraulicos.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_3:
        st.download_button(
            "Baixar resumo por variável",
            data=gerar_csv(resultado.resumo_variaveis),
            file_name="resumo_parametros_hidraulicos.csv",
            mime="text/csv",
            use_container_width=True,
        )
