"""Aba 1 — presença e identificação das usinas hidráulicas.

Compara a relação de usinas interpretada pelo NEWAVE no pmo.dat com as
entidades efetivamente criadas no system/hydros.json do COBRE.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
from typing import Any

import pandas as pd


STATUS_CONSIDERADOS = {"EX", "EE", "NE"}
STATUS_VALIDOS = STATUS_CONSIDERADOS | {"NC"}

RE_CONFIGURACAO = re.compile(
    r"^\s*(?P<codigo>\d+)\s+"
    r"(?P<nome>.+?)\s+"
    r"(?P<posto>\d+)\s+"
    r"(?P<jusante>\d+)\s+"
    r"(?P<ree>\S+)\s+"
    r"(?P<volume_inicial>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<status>EX|EE|NE|NC)\s+"
    r"(?P<altera>SIM|NAO)\s+"
    r"(?P<inicio_historico>\d{4})\s+"
    r"(?P<fim_historico>\d{4})\s+"
    r"(?P<tec>\d+)\s*$"
)

NUMERO = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
RE_DADOS_HIDRAULICOS = re.compile(
    rf"^\s*(?P<nome>.+?)\s+"
    rf"(?P<posto>\d+)\s+"
    rf"(?P<vmin>{NUMERO})\s+"
    rf"(?P<vmax>{NUMERO})\s+"
    rf"(?P<vref>{NUMERO})\s+"
    rf"(?P<pinst>{NUMERO})\s+"
    rf"(?P<rho>{NUMERO})(?:\s+|$)"
)

RE_TOTAL_USINAS = re.compile(r"TOTAL\s+DE\s+USINAS\s+(?P<total>\d+)")
RE_LINHA_EXPANSAO = re.compile(
    r"^\s+(?P<nome>[A-ZÀ-Ü][A-ZÀ-Ü0-9 ._-]{0,11}?)\s+"
    r"(?P<mes>\d{1,2})\s+(?P<ano>\d{1,4})\s+"
    r"(?P<duracao>\d+)\s*$"
)


class ErroLeitura(ValueError):
    """Erro apresentado quando um arquivo não possui a estrutura necessária."""


@dataclass(frozen=True)
class ResultadoValidacao:
    """Conjunto de resultados produzido pela comparação."""

    tabela_completa: pd.DataFrame
    de_para: pd.DataFrame
    ausentes: pd.DataFrame
    inesperadas: pd.DataFrame
    total_pmo: int
    total_esperado: int
    total_cobre: int
    total_ficticias: int
    total_ausentes: int
    total_ee_ausentes: int
    total_ids_deslocados: int
    ids_cobre_validos: bool
    aprovada: bool
    avisos: tuple[str, ...]


def normalizar_nome(valor: Any) -> str:
    """Normaliza espaços e caixa sem retirar os caracteres significativos."""
    return " ".join(str(valor).strip().upper().split())


def decodificar_texto(conteudo: bytes) -> str:
    """Decodifica o pmo.dat aceitando os encodings usuais dos decks."""
    for encoding in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return conteudo.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ErroLeitura("Não foi possível identificar a codificação do pmo.dat.")


def ler_configuracao_pmo(linhas: list[str]) -> tuple[pd.DataFrame, int]:
    """Lê a tabela de configuração hidráulica do pmo.dat."""
    dentro = False
    registros: list[dict[str, Any]] = []
    total_declarado: int | None = None

    for linha in linhas:
        if "NUM" in linha and "NOME" in linha and "POSTO" in linha and "JUSANTE" in linha:
            dentro = True
            continue

        if not dentro:
            continue

        total_match = RE_TOTAL_USINAS.search(linha)
        if total_match:
            total_declarado = int(total_match.group("total"))
            break

        match = RE_CONFIGURACAO.match(linha)
        if not match:
            continue

        item = match.groupdict()
        registros.append(
            {
                "codigo_newave": int(item["codigo"]),
                "nome_usina": item["nome"].strip(),
                "posto": int(item["posto"]),
                "codigo_jusante": int(item["jusante"]),
                "ree": item["ree"].strip(),
                "status_newave": item["status"],
            }
        )

    if not registros:
        raise ErroLeitura(
            "Não foi encontrada a tabela de configuração das usinas "
            "hidráulicas no pmo.dat."
        )
    if total_declarado is None:
        raise ErroLeitura(
            "A tabela hidráulica foi localizada, mas o TOTAL DE USINAS não foi encontrado."
        )

    df = pd.DataFrame(registros)
    if df["codigo_newave"].duplicated().any():
        codigos = sorted(
            df.loc[df["codigo_newave"].duplicated(False), "codigo_newave"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ErroLeitura(f"O pmo.dat possui códigos de usina repetidos: {codigos}.")
    if len(df) != total_declarado:
        raise ErroLeitura(
            f"O pmo.dat declara {total_declarado} usinas, mas foram lidas {len(df)}."
        )
    if not set(df["status_newave"]).issubset(STATUS_VALIDOS):
        desconhecidos = sorted(set(df["status_newave"]) - STATUS_VALIDOS)
        raise ErroLeitura(f"Foram encontrados status hidráulicos desconhecidos: {desconhecidos}.")
    return df, total_declarado


def ler_dados_hidraulicos_pmo(linhas: list[str]) -> pd.DataFrame:
    """Lê posto, potência instalada e produtividade específica do pmo.dat."""
    dentro = False
    registros: list[dict[str, Any]] = []

    for linha in linhas:
        if "NOME" in linha and "POSTO" in linha and "PROD.ESP." in linha:
            dentro = True
            continue
        if dentro and "POLINOMIO VOLUME-COTA" in linha:
            break
        if not dentro:
            continue

        match = RE_DADOS_HIDRAULICOS.match(linha)
        if not match:
            continue
        item = match.groupdict()
        registros.append(
            {
                "nome_normalizado": normalizar_nome(item["nome"]),
                "posto": int(item["posto"]),
                "potencia_inicial_mw": float(item["pinst"]),
                "produtibilidade_especifica": float(item["rho"]),
            }
        )

    if not registros:
        raise ErroLeitura(
            "Não foi encontrada a tabela DADOS DAS USINAS HIDROELETRICAS "
            "com a coluna PROD.ESP. no pmo.dat."
        )

    df = pd.DataFrame(registros).drop_duplicates(
        subset=["nome_normalizado", "posto"], keep="first"
    )
    return df


def ler_usinas_com_expansao(linhas: list[str]) -> set[str]:
    """Retorna os nomes apresentados no cronograma de expansão do pmo.dat."""
    dentro = False
    nomes: set[str] = set()

    for linha in linhas:
        if "C R O N O G R A M A  DE  E X P A N S A O" in linha:
            dentro = True
            continue
        if dentro and "SAZONALIZACAO" in linha:
            break
        if not dentro:
            continue

        match = RE_LINHA_EXPANSAO.match(linha)
        if match:
            nomes.add(normalizar_nome(match.group("nome")))
    return nomes


def ler_pmo(conteudo: bytes) -> tuple[pd.DataFrame, int, tuple[str, ...]]:
    """Consolida cadastro, situação e produtividade das usinas do PMO."""
    texto = decodificar_texto(conteudo)
    linhas = texto.splitlines()
    configuracao, total = ler_configuracao_pmo(linhas)
    dados = ler_dados_hidraulicos_pmo(linhas)

    configuracao["nome_normalizado"] = configuracao["nome_usina"].map(normalizar_nome)
    consolidado = configuracao.merge(
        dados,
        on=["nome_normalizado", "posto"],
        how="left",
        validate="one_to_one",
    )

    sem_dados = consolidado["produtibilidade_especifica"].isna()
    if sem_dados.any():
        faltantes = consolidado.loc[
            sem_dados, ["codigo_newave", "nome_usina", "posto"]
        ].to_dict("records")
        raise ErroLeitura(
            "Não foi possível localizar os dados hidráulicos destas usinas "
            f"na tabela de produtividade do pmo.dat: {faltantes}."
        )

    existentes = consolidado[consolidado["status_newave"].isin({"EX", "EE"})]
    postos_com_geracao = set(
        existentes.loc[
            existentes["produtibilidade_especifica"] > 0.0, "posto"
        ].astype(int)
    )
    consolidado["eh_ficticia"] = (
        consolidado["status_newave"].isin({"EX", "EE"})
        & (consolidado["produtibilidade_especifica"] == 0.0)
        & consolidado["posto"].isin(postos_com_geracao)
    )

    avisos: list[str] = []
    expansoes = ler_usinas_com_expansao(linhas)
    declaradas_expansao = consolidado[
        consolidado["status_newave"].isin({"EE", "NE"})
    ]
    sem_cronograma = declaradas_expansao[
        ~declaradas_expansao["nome_normalizado"].isin(expansoes)
    ]
    if not sem_cronograma.empty:
        nomes = ", ".join(sem_cronograma["nome_usina"].astype(str))
        avisos.append(
            "Estas usinas possuem status EE/NE, mas não foram localizadas no "
            f"cronograma de expansão do PMO: {nomes}."
        )

    return consolidado, total, tuple(avisos)


def ler_hydros_json(conteudo: bytes) -> tuple[pd.DataFrame, bool]:
    """Lê e valida a estrutura básica do system/hydros.json."""
    try:
        objeto = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErroLeitura(f"Não foi possível ler o hydros.json: {exc}") from exc

    hydros = objeto.get("hydros") if isinstance(objeto, dict) else None
    if not isinstance(hydros, list):
        raise ErroLeitura("O hydros.json não possui uma lista válida no campo 'hydros'.")

    registros: list[dict[str, Any]] = []
    for posicao, hydro in enumerate(hydros):
        if not isinstance(hydro, dict) or "id" not in hydro or "name" not in hydro:
            raise ErroLeitura(
                f"O registro {posicao} do hydros.json não possui os campos id e name."
            )
        registros.append(
            {
                "id_cobre": int(hydro["id"]),
                "nome_cobre": str(hydro["name"]).strip(),
                "nome_normalizado": normalizar_nome(hydro["name"]),
            }
        )

    df = pd.DataFrame(
        registros, columns=["id_cobre", "nome_cobre", "nome_normalizado"]
    ).sort_values("id_cobre", ignore_index=True)

    if df["id_cobre"].duplicated().any():
        ids = sorted(
            df.loc[df["id_cobre"].duplicated(False), "id_cobre"].unique().tolist()
        )
        raise ErroLeitura(f"O hydros.json possui IDs repetidos: {ids}.")

    ids_validos = df["id_cobre"].tolist() == list(range(len(df)))
    return df, ids_validos


def alinhar_usinas(
    esperadas: pd.DataFrame, cobre: pd.DataFrame
) -> tuple[dict[int, int], list[int], list[int]]:
    """Alinha listas ordenadas e localiza omissões sem usar o nome como ID."""
    nomes_newave = esperadas["nome_normalizado"].tolist()
    nomes_cobre = cobre["nome_normalizado"].tolist()
    comparador = SequenceMatcher(
        a=nomes_newave, b=nomes_cobre, autojunk=False
    )

    correspondencias: dict[int, int] = {}
    ausentes: list[int] = []
    inesperadas: list[int] = []

    for tag, i1, i2, j1, j2 in comparador.get_opcodes():
        if tag == "equal":
            for indice_newave, indice_cobre in zip(range(i1, i2), range(j1, j2)):
                correspondencias[indice_newave] = indice_cobre
        elif tag == "delete":
            ausentes.extend(range(i1, i2))
        elif tag == "insert":
            inesperadas.extend(range(j1, j2))
        else:  # replace: não há uma chave técnica confiável para casar os blocos
            ausentes.extend(range(i1, i2))
            inesperadas.extend(range(j1, j2))

    return correspondencias, ausentes, inesperadas


def validar_usinas(pmo_bytes: bytes, hydros_bytes: bytes) -> ResultadoValidacao:
    """Executa a validação completa entre o PMO e o deck COBRE."""
    pmo, total_pmo, avisos = ler_pmo(pmo_bytes)
    cobre, ids_cobre_validos = ler_hydros_json(hydros_bytes)

    consideradas = pmo[
        pmo["status_newave"].isin(STATUS_CONSIDERADOS) & ~pmo["eh_ficticia"]
    ].copy()
    consideradas = consideradas.sort_values("codigo_newave", ignore_index=True)
    consideradas["id_esperado_cobre"] = range(len(consideradas))

    correspondencias, ausentes_idx, inesperadas_idx = alinhar_usinas(
        consideradas, cobre
    )

    linhas_resultado: list[dict[str, Any]] = []
    linhas_de_para: list[dict[str, Any]] = []

    for indice, row in consideradas.iterrows():
        indice_cobre = correspondencias.get(indice)
        if indice_cobre is None:
            id_cobre: int | None = None
            nome_cobre: str | None = None
            resultado = "Ausente no COBRE"
        else:
            registro_cobre = cobre.iloc[indice_cobre]
            id_cobre = int(registro_cobre["id_cobre"])
            nome_cobre = str(registro_cobre["nome_cobre"])
            if id_cobre == int(row["id_esperado_cobre"]):
                resultado = "Convertida corretamente"
            else:
                resultado = "Convertida, mas com ID deslocado"
            linhas_de_para.append(
                {
                    "codigo_newave": int(row["codigo_newave"]),
                    "id_cobre": id_cobre,
                    "nome_usina": str(row["nome_usina"]),
                }
            )

        linhas_resultado.append(
            {
                "codigo_newave": int(row["codigo_newave"]),
                "id_esperado_cobre": int(row["id_esperado_cobre"]),
                "id_cobre_encontrado": id_cobre,
                "nome_newave": str(row["nome_usina"]),
                "nome_cobre": nome_cobre,
                "status_newave": str(row["status_newave"]),
                "resultado": resultado,
            }
        )

    codigos_considerados = set(consideradas["codigo_newave"].astype(int))
    excluidas = pmo[~pmo["codigo_newave"].isin(codigos_considerados)].copy()
    for _, row in excluidas.iterrows():
        if bool(row["eh_ficticia"]):
            resultado = "Exclusão justificada: usina fictícia"
        elif row["status_newave"] == "NC":
            resultado = "Não considerada pelo NEWAVE (NC)"
        else:
            resultado = "Exclusão não classificada"
        linhas_resultado.append(
            {
                "codigo_newave": int(row["codigo_newave"]),
                "id_esperado_cobre": None,
                "id_cobre_encontrado": None,
                "nome_newave": str(row["nome_usina"]),
                "nome_cobre": None,
                "status_newave": str(row["status_newave"]),
                "resultado": resultado,
            }
        )

    inesperadas_registros: list[dict[str, Any]] = []
    for indice in inesperadas_idx:
        row = cobre.iloc[indice]
        inesperadas_registros.append(
            {
                "id_cobre": int(row["id_cobre"]),
                "nome_cobre": str(row["nome_cobre"]),
                "resultado": "Usina do COBRE sem correspondência no PMO",
            }
        )

    tabela = pd.DataFrame(linhas_resultado)
    tabela["_ordem"] = tabela["codigo_newave"]
    tabela = tabela.sort_values("_ordem").drop(columns="_ordem").reset_index(drop=True)
    tabela["id_esperado_cobre"] = tabela["id_esperado_cobre"].astype("Int64")
    tabela["id_cobre_encontrado"] = tabela["id_cobre_encontrado"].astype("Int64")

    ausentes = tabela[tabela["resultado"] == "Ausente no COBRE"].copy()
    inesperadas = pd.DataFrame(
        inesperadas_registros,
        columns=["id_cobre", "nome_cobre", "resultado"],
    )
    de_para = pd.DataFrame(
        linhas_de_para,
        columns=["codigo_newave", "id_cobre", "nome_usina"],
    ).sort_values("codigo_newave", ignore_index=True)

    total_ids_deslocados = int(
        (tabela["resultado"] == "Convertida, mas com ID deslocado").sum()
    )
    total_ee_ausentes = int(
        ((ausentes["status_newave"] == "EE")).sum()
    )
    aprovada = (
        ausentes.empty
        and inesperadas.empty
        and ids_cobre_validos
        and total_ids_deslocados == 0
        and len(consideradas) == len(cobre)
    )

    return ResultadoValidacao(
        tabela_completa=tabela,
        de_para=de_para,
        ausentes=ausentes,
        inesperadas=inesperadas,
        total_pmo=total_pmo,
        total_esperado=len(consideradas),
        total_cobre=len(cobre),
        total_ficticias=int(pmo["eh_ficticia"].sum()),
        total_ausentes=len(ausentes),
        total_ee_ausentes=total_ee_ausentes,
        total_ids_deslocados=total_ids_deslocados,
        ids_cobre_validos=ids_cobre_validos,
        aprovada=aprovada,
        avisos=avisos,
    )


def gerar_csv(df: pd.DataFrame) -> bytes:
    """Gera CSV amigável ao Excel em português, sem salvar arquivo temporário."""
    buffer = StringIO()
    df.to_csv(buffer, index=False, sep=";", lineterminator="\n")
    return buffer.getvalue().encode("utf-8-sig")


def mostrar_aba() -> None:
    """Renderiza a Aba 1 no Streamlit."""
    import hashlib

    import streamlit as st

    st.subheader("Presença e identificação das usinas hidráulicas")
    st.markdown(
        "Esta aba verifica se todas as usinas consideradas pelo NEWAVE no "
        "`pmo.dat` foram criadas no `system/hydros.json` do COBRE. Usinas "
        "`EE` são tratadas como existentes com expansão e, portanto, precisam "
        "estar presentes desde o início."
    )

    coluna_pmo, coluna_cobre = st.columns(2)
    with coluna_pmo:
        arquivo_pmo = st.file_uploader(
            "Arquivo NEWAVE — pmo.dat",
            type=["dat"],
            key="aba01_pmo",
        )
    with coluna_cobre:
        arquivo_hydros = st.file_uploader(
            "Arquivo COBRE — hydros.json",
            type=["json"],
            key="aba01_hydros",
        )

    if arquivo_pmo is None or arquivo_hydros is None:
        st.info("Envie os dois arquivos para iniciar a validação.")
        return

    pmo_bytes = arquivo_pmo.getvalue()
    hydros_bytes = arquivo_hydros.getvalue()
    hash_arquivos = hashlib.sha256()
    hash_arquivos.update(pmo_bytes)
    hash_arquivos.update(b"\0HYDROS\0")
    hash_arquivos.update(hydros_bytes)
    assinatura = hash_arquivos.hexdigest()
    chave_cache = "aba01_resultado_cache"

    executar = st.button(
        "Executar validação", type="primary", key="aba01_executar"
    )
    if executar:
        try:
            with st.spinner("Lendo o PMO e comparando as usinas..."):
                resultado_calculado = validar_usinas(pmo_bytes, hydros_bytes)
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
        st.info("Clique em Executar validação para comparar os arquivos enviados.")
        return
    resultado: ResultadoValidacao = cache["resultado"]

    for aviso in resultado.avisos:
        st.warning(aviso)

    colunas = st.columns(6)
    colunas[0].metric("Usinas no PMO", resultado.total_pmo)
    colunas[1].metric("Esperadas no COBRE", resultado.total_esperado)
    colunas[2].metric("Encontradas no COBRE", resultado.total_cobre)
    colunas[3].metric("Ausentes", resultado.total_ausentes)
    colunas[4].metric("EE ausentes", resultado.total_ee_ausentes)
    colunas[5].metric("Fictícias excluídas", resultado.total_ficticias)

    if resultado.total_ee_ausentes:
        st.error(
            "ERRO CRÍTICO: "
            f"{resultado.total_ee_ausentes} usina(s) existente(s) em expansão "
            "não foram consideradas na conversão."
        )
    elif resultado.aprovada:
        st.success(
            "Validação aprovada: todas as usinas esperadas foram criadas "
            "com os IDs corretos."
        )
    else:
        st.error(
            "Validação reprovada: existem ausências, usinas inesperadas ou "
            "divergências na atribuição dos IDs."
        )

    if not resultado.ids_cobre_validos:
        st.error("Os IDs do hydros.json não são únicos e contínuos a partir de zero.")

    if not resultado.ausentes.empty:
        st.markdown("#### Usinas esperadas que não foram criadas no COBRE")
        st.dataframe(
            resultado.ausentes[
                [
                    "codigo_newave",
                    "nome_newave",
                    "status_newave",
                    "id_esperado_cobre",
                    "resultado",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    if not resultado.inesperadas.empty:
        st.markdown("#### Usinas inesperadas no COBRE")
        st.dataframe(
            resultado.inesperadas,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Resultado completo")
    filtro = st.selectbox(
        "Mostrar",
        [
            "Somente problemas",
            "Todas as usinas",
            "Somente usinas EE",
            "Somente exclusões justificadas",
        ],
        key="aba01_filtro",
    )
    exibicao = resultado.tabela_completa
    if filtro == "Somente problemas":
        exibicao = exibicao[
            ~exibicao["resultado"].isin(
                [
                    "Convertida corretamente",
                    "Exclusão justificada: usina fictícia",
                    "Não considerada pelo NEWAVE (NC)",
                ]
            )
        ]
    elif filtro == "Somente usinas EE":
        exibicao = exibicao[exibicao["status_newave"] == "EE"]
    elif filtro == "Somente exclusões justificadas":
        exibicao = exibicao[
            exibicao["resultado"].str.startswith("Exclusão justificada")
        ]

    st.dataframe(exibicao, hide_index=True, use_container_width=True)

    if resultado.total_ids_deslocados:
        st.warning(
            f"{resultado.total_ids_deslocados} usina(s) foram encontradas com "
            "ID deslocado porque uma ou mais usinas anteriores na ordenação "
            "crescente dos códigos não foram criadas."
        )

    st.markdown("#### Exportações")
    col_download_1, col_download_2 = st.columns(2)
    with col_download_1:
        st.download_button(
            "Baixar resultado completo",
            data=gerar_csv(resultado.tabela_completa),
            file_name="validacao_usinas_hidraulicas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_download_2:
        st.download_button(
            "Baixar de-para código NEWAVE × ID COBRE",
            data=gerar_csv(resultado.de_para),
            file_name="de_para_codigo_newave_id_cobre.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.caption(
        "O de-para adicional contém somente codigo_newave, id_cobre e "
        "nome_usina. Usinas ausentes não recebem um ID inventado."
    )
