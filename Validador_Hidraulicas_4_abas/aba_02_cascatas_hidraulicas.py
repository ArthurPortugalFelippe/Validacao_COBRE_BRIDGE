"""Aba 2 — formação das cascatas hidráulicas.

Reconstrói a topologia do system/hydros.json e compara cada ligação com a
tabela de cascatas já apresentada pelo NEWAVE no pmo.dat.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from aba_01_usinas_hidraulicas import (
    ErroLeitura,
    decodificar_texto,
    gerar_csv,
    validar_usinas,
)


TITULO_CASCATA_PMO = (
    "CASCATA DAS USINAS HIDROELETRICAS PARA PERIODOS INDIDUALIZADOS"
)

RESULTADOS_CORRETOS = {
    "Ligação de cascata correta",
    "Saída da cascata representada corretamente",
}


@dataclass(frozen=True)
class ResultadoCascatas:
    """Conjunto de tabelas e indicadores da validação de topologia."""

    cascata_pmo: pd.DataFrame
    cascata_cobre: pd.DataFrame
    comparacao: pd.DataFrame
    divergencias: pd.DataFrame
    usinas_nao_representadas: pd.DataFrame
    usinas_inesperadas_cobre: pd.DataFrame
    total_pmo: int
    total_cobre: int
    total_corretas: int
    total_divergencias: int
    total_nao_representadas: int
    total_ligacoes_desviadas: int
    total_divergencias_canal: int
    considera_ficticias_pmo: bool | None
    ids_cobre_validos: bool
    aprovada: bool
    avisos: tuple[str, ...]


def _inteiro_ou_none(valor: Any, campo: str, contexto: str) -> int | None:
    """Converte um campo JSON opcional em inteiro, com mensagem clara de erro."""
    if valor is None:
        return None
    if isinstance(valor, bool):
        raise ErroLeitura(
            f"O campo {campo} de {contexto} não pode ser verdadeiro/falso."
        )
    try:
        convertido = int(valor)
    except (TypeError, ValueError) as exc:
        raise ErroLeitura(
            f"O campo {campo} de {contexto} não contém um ID inteiro válido."
        ) from exc
    if isinstance(valor, float) and not valor.is_integer():
        raise ErroLeitura(
            f"O campo {campo} de {contexto} não contém um ID inteiro válido."
        )
    return convertido


def ler_cascata_pmo(
    conteudo: bytes,
) -> tuple[pd.DataFrame, bool | None]:
    """Lê a tabela pronta de cascatas hidráulicas do pmo.dat."""
    linhas = decodificar_texto(conteudo).splitlines()
    inicio: int | None = None
    considera_ficticias: bool | None = None

    for indice, linha in enumerate(linhas):
        if TITULO_CASCATA_PMO in linha.upper():
            inicio = indice
            break

    if inicio is None:
        raise ErroLeitura(
            "Não foi encontrada no pmo.dat a tabela 'CASCATA DAS USINAS "
            "HIDROELETRICAS PARA PERIODOS INDIDUALIZADOS'."
        )

    for linha in linhas[inicio : inicio + 12]:
        match = re.search(
            r"CONSIDERA\s+AS\s+USINAS\s+FICTICIAS\s*:\s*(SIM|NAO)",
            linha.upper(),
        )
        if match:
            considera_ficticias = match.group(1) == "SIM"
            break

    separador: int | None = None
    for indice in range(inicio + 1, min(len(linhas), inicio + 20)):
        linha = linhas[indice].upper()
        if "X----X" in linha and "X-------------X" in linha:
            separador = indice
            break
    if separador is None:
        raise ErroLeitura(
            "A tabela de cascatas foi localizada, mas seu cabeçalho não pôde ser lido."
        )

    registros: list[dict[str, Any]] = []
    for linha in linhas[separador + 1 :]:
        codigo_montante_txt = linha[0:6].strip() if len(linha) >= 6 else ""
        if not codigo_montante_txt.isdigit():
            if registros and linha.strip():
                break
            continue

        codigo_jusante_txt = linha[20:25].strip() if len(linha) >= 25 else ""
        if not codigo_jusante_txt.isdigit():
            raise ErroLeitura(
                "Foi encontrada uma linha sem código de jusante válido na "
                f"tabela de cascatas: {linha!r}."
            )

        codigo_canal_txt = linha[41:46].strip() if len(linha) >= 46 else ""
        if codigo_canal_txt and not codigo_canal_txt.isdigit():
            raise ErroLeitura(
                "Foi encontrado um código inválido de jusante do canal de desvio "
                f"na tabela de cascatas: {linha!r}."
            )
        codigo_canal = int(codigo_canal_txt) if codigo_canal_txt else 0
        nome_canal = linha[46:59].strip() if len(linha) > 46 else ""

        registros.append(
            {
                "codigo_montante": int(codigo_montante_txt),
                "nome_montante": linha[6:20].strip(),
                "codigo_jusante": int(codigo_jusante_txt),
                "nome_jusante": linha[25:41].strip() or "-",
                "codigo_jusante_canal_desvio": codigo_canal,
                "nome_jusante_canal_desvio": nome_canal or "-",
            }
        )

    if not registros:
        raise ErroLeitura(
            "A tabela de cascatas foi localizada no pmo.dat, mas nenhuma relação "
            "montante–jusante foi lida."
        )

    df = pd.DataFrame(registros)
    if df["codigo_montante"].duplicated().any():
        repetidos = sorted(
            df.loc[df["codigo_montante"].duplicated(False), "codigo_montante"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ErroLeitura(
            "A tabela de cascatas do PMO possui usinas montantes repetidas: "
            f"{repetidos}."
        )
    return df, considera_ficticias


def ler_cascata_cobre(
    conteudo: bytes,
) -> tuple[pd.DataFrame, bool]:
    """Reconstrói a cascata a partir dos IDs e downstream_id do hydros.json."""
    try:
        objeto = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErroLeitura(f"Não foi possível ler o hydros.json: {exc}") from exc

    hydros = objeto.get("hydros") if isinstance(objeto, dict) else None
    if not isinstance(hydros, list):
        raise ErroLeitura("O hydros.json não possui uma lista válida no campo 'hydros'.")

    registros_brutos: list[dict[str, Any]] = []
    for posicao, hydro in enumerate(hydros):
        contexto = f"registro {posicao} do hydros.json"
        if not isinstance(hydro, dict) or "id" not in hydro or "name" not in hydro:
            raise ErroLeitura(
                f"O {contexto} não possui os campos obrigatórios id e name."
            )
        hydro_id = _inteiro_ou_none(hydro["id"], "id", contexto)
        if hydro_id is None:
            raise ErroLeitura(f"O campo id do {contexto} não pode ser nulo.")
        downstream_id = _inteiro_ou_none(
            hydro.get("downstream_id"), "downstream_id", contexto
        )

        diversion = hydro.get("diversion")
        diversion_id: int | None = None
        if diversion is not None:
            if not isinstance(diversion, dict):
                raise ErroLeitura(
                    f"O campo diversion do {contexto} deve ser um objeto ou nulo."
                )
            if "downstream_id" not in diversion:
                raise ErroLeitura(
                    f"O campo diversion do {contexto} não possui downstream_id."
                )
            diversion_id = _inteiro_ou_none(
                diversion.get("downstream_id"),
                "diversion.downstream_id",
                contexto,
            )

        registros_brutos.append(
            {
                "id_montante": hydro_id,
                "nome_montante": str(hydro["name"]).strip(),
                "downstream_id": downstream_id,
                "diversion_downstream_id": diversion_id,
            }
        )

    ids = [r["id_montante"] for r in registros_brutos]
    if len(ids) != len(set(ids)):
        repetidos = sorted({item for item in ids if ids.count(item) > 1})
        raise ErroLeitura(f"O hydros.json possui IDs repetidos: {repetidos}.")
    ids_ordenados = sorted(ids)
    ids_validos = ids_ordenados == list(range(len(ids_ordenados)))
    nomes_por_id = {r["id_montante"]: r["nome_montante"] for r in registros_brutos}

    for registro in registros_brutos:
        for campo in ("downstream_id", "diversion_downstream_id"):
            destino = registro[campo]
            if destino is not None and destino not in nomes_por_id:
                raise ErroLeitura(
                    f"A usina COBRE ID {registro['id_montante']} aponta em {campo} "
                    f"para o ID inexistente {destino}."
                )

    registros: list[dict[str, Any]] = []
    for registro in sorted(registros_brutos, key=lambda item: item["id_montante"]):
        downstream_id = registro["downstream_id"]
        diversion_id = registro["diversion_downstream_id"]
        registros.append(
            {
                "id_montante": registro["id_montante"],
                "nome_montante": registro["nome_montante"],
                "id_jusante": downstream_id,
                "nome_jusante": nomes_por_id.get(downstream_id, "-"),
                "id_jusante_canal_desvio": diversion_id,
                "nome_jusante_canal_desvio": nomes_por_id.get(diversion_id, "-"),
            }
        )
    df = pd.DataFrame(registros)
    for coluna in ("id_jusante", "id_jusante_canal_desvio"):
        df[coluna] = df[coluna].astype("Int64")
    return df, ids_validos


def _nome(valor: Any, padrao: str = "-") -> str:
    """Converte nomes opcionais para texto de exibição."""
    if valor is None or pd.isna(valor) or not str(valor).strip():
        return padrao
    return str(valor).strip()


def _proxima_usina_representada(
    codigo_inicial: int,
    jusante_pmo: dict[int, int],
    codigos_representados: set[int],
) -> int | None:
    """Percorre a cascata PMO até a próxima usina presente no COBRE ou a saída."""
    codigo = codigo_inicial
    visitados: set[int] = set()
    while codigo != 0 and codigo not in codigos_representados:
        if codigo in visitados:
            return None
        visitados.add(codigo)
        if codigo not in jusante_pmo:
            return None
        codigo = jusante_pmo[codigo]
    return codigo


def _comparar_canal_desvio(
    codigo_esperado: int,
    id_encontrado: int | None,
    codigo_montante_representado: bool,
    codigo_para_id: dict[int, int],
) -> str:
    """Classifica somente a ligação do canal de desvio."""
    if not codigo_montante_representado:
        return "Não comparado: usina montante não representada"
    if codigo_esperado == 0 and id_encontrado is None:
        return "Sem canal de desvio nos dois modelos"
    if codigo_esperado == 0 and id_encontrado is not None:
        return "Canal de desvio criado apenas no COBRE"
    id_esperado = codigo_para_id.get(codigo_esperado)
    if id_esperado is None:
        return "Jusante do canal de desvio não representada no COBRE"
    if id_encontrado == id_esperado:
        return "Canal de desvio correto"
    return "Divergência no canal de desvio"


def comparar_cascatas(
    cascata_pmo: pd.DataFrame,
    cascata_cobre: pd.DataFrame,
    codigo_para_id: dict[int, int],
    nomes_pmo: dict[int, str],
) -> pd.DataFrame:
    """Compara cada montante do PMO com a ligação reconstruída no COBRE."""
    cobre_por_id = cascata_cobre.set_index("id_montante").to_dict("index")
    id_para_codigo = {identificador: codigo for codigo, identificador in codigo_para_id.items()}
    jusante_pmo = dict(
        zip(cascata_pmo["codigo_montante"], cascata_pmo["codigo_jusante"])
    )
    codigos_representados = set(codigo_para_id)
    linhas: list[dict[str, Any]] = []

    for _, row in cascata_pmo.iterrows():
        codigo_montante = int(row["codigo_montante"])
        codigo_jusante = int(row["codigo_jusante"])
        codigo_canal = int(row["codigo_jusante_canal_desvio"])
        id_montante = codigo_para_id.get(codigo_montante)
        registro_cobre = cobre_por_id.get(id_montante) if id_montante is not None else None

        id_jusante_encontrado: int | None = None
        nome_jusante_encontrado = "-"
        codigo_jusante_encontrado: int | None = None
        id_canal_encontrado: int | None = None
        nome_canal_encontrado = "-"

        if registro_cobre is None:
            resultado_cascata = "Usina montante do NEWAVE não representada no COBRE"
            detalhe = (
                f"A usina {_nome(row['nome_montante'])} existe na cascata do PMO, "
                "mas não foi representada no deck COBRE."
            )
        else:
            id_jusante_valor = registro_cobre["id_jusante"]
            id_jusante_encontrado = (
                None if pd.isna(id_jusante_valor) else int(id_jusante_valor)
            )
            nome_jusante_encontrado = _nome(registro_cobre["nome_jusante"])
            codigo_jusante_encontrado = (
                id_para_codigo.get(id_jusante_encontrado)
                if id_jusante_encontrado is not None
                else 0
            )
            id_canal_valor = registro_cobre["id_jusante_canal_desvio"]
            id_canal_encontrado = (
                None if pd.isna(id_canal_valor) else int(id_canal_valor)
            )
            nome_canal_encontrado = _nome(
                registro_cobre["nome_jusante_canal_desvio"]
            )

            if codigo_jusante == 0 and id_jusante_encontrado is None:
                resultado_cascata = "Saída da cascata representada corretamente"
                detalhe = "A saída da cascata foi representada corretamente nos dois modelos."
            elif codigo_jusante in codigo_para_id:
                id_jusante_esperado = codigo_para_id[codigo_jusante]
                if id_jusante_encontrado == id_jusante_esperado:
                    resultado_cascata = "Ligação de cascata correta"
                    detalhe = (
                        f"A ligação {_nome(row['nome_montante'])} → "
                        f"{_nome(row['nome_jusante'])} coincide nos dois modelos."
                    )
                else:
                    resultado_cascata = "Usina ligada a um jusante diferente"
                    detalhe = (
                        f"O PMO liga {_nome(row['nome_montante'])} a "
                        f"{_nome(row['nome_jusante'])}; o COBRE liga a "
                        f"{nome_jusante_encontrado}."
                    )
            else:
                proxima = _proxima_usina_representada(
                    codigo_jusante, jusante_pmo, codigos_representados
                )
                if codigo_jusante_encontrado == proxima and proxima is not None:
                    resultado_cascata = (
                        "Ligação desviada em razão de usina intermediária não representada"
                    )
                    nome_proxima = (
                        nomes_pmo.get(proxima, "-") if proxima != 0 else "a saída"
                    )
                    detalhe = (
                        f"A usina jusante esperada, {_nome(row['nome_jusante'])}, "
                        "não foi representada no COBRE. A ligação foi direcionada "
                        f"para {nome_proxima}, próxima referência representada na cascata."
                    )
                else:
                    resultado_cascata = (
                        "Usina jusante esperada não representada no COBRE"
                    )
                    detalhe = (
                        f"A usina jusante esperada, {_nome(row['nome_jusante'])}, "
                        "não foi representada no COBRE. A ligação encontrada aponta "
                        f"para {nome_jusante_encontrado}."
                    )

        resultado_canal = _comparar_canal_desvio(
            codigo_canal,
            id_canal_encontrado,
            registro_cobre is not None,
            codigo_para_id,
        )
        if (
            resultado_cascata in RESULTADOS_CORRETOS
            and resultado_canal
            not in {"Sem canal de desvio nos dois modelos", "Canal de desvio correto"}
        ):
            resultado_geral = "Divergência no canal de desvio"
            detalhe_geral = (
                f"{detalhe} O canal de desvio apresenta a classificação: "
                f"{resultado_canal}."
            )
        else:
            resultado_geral = resultado_cascata
            detalhe_geral = detalhe

        linhas.append(
            {
                "codigo_montante_newave": codigo_montante,
                "nome_montante_newave": _nome(row["nome_montante"]),
                "id_montante_cobre": id_montante,
                "nome_montante_cobre": (
                    _nome(registro_cobre["nome_montante"])
                    if registro_cobre is not None
                    else None
                ),
                "codigo_jusante_newave": codigo_jusante,
                "nome_jusante_newave": _nome(row["nome_jusante"]),
                "id_jusante_cobre_esperado": codigo_para_id.get(codigo_jusante),
                "id_jusante_cobre_encontrado": id_jusante_encontrado,
                "nome_jusante_cobre_encontrado": nome_jusante_encontrado,
                "codigo_canal_desvio_newave": codigo_canal,
                "nome_canal_desvio_newave": _nome(
                    row["nome_jusante_canal_desvio"]
                ),
                "id_canal_desvio_cobre_esperado": codigo_para_id.get(codigo_canal),
                "id_canal_desvio_cobre_encontrado": id_canal_encontrado,
                "nome_canal_desvio_cobre_encontrado": nome_canal_encontrado,
                "resultado_cascata": resultado_cascata,
                "resultado_canal_desvio": resultado_canal,
                "resultado": resultado_geral,
                "detalhe": detalhe_geral,
            }
        )

    df = pd.DataFrame(linhas)
    for coluna in (
        "id_montante_cobre",
        "id_jusante_cobre_esperado",
        "id_jusante_cobre_encontrado",
        "id_canal_desvio_cobre_esperado",
        "id_canal_desvio_cobre_encontrado",
    ):
        df[coluna] = df[coluna].astype("Int64")
    return df


def validar_cascatas(pmo_bytes: bytes, hydros_bytes: bytes) -> ResultadoCascatas:
    """Executa a validação completa da formação das cascatas."""
    cascata_pmo, considera_ficticias = ler_cascata_pmo(pmo_bytes)
    cascata_cobre, ids_cobre_validos = ler_cascata_cobre(hydros_bytes)
    identificacao = validar_usinas(pmo_bytes, hydros_bytes)

    codigo_para_id = {
        int(row["codigo_newave"]): int(row["id_cobre"])
        for _, row in identificacao.de_para.iterrows()
    }
    nomes_pmo = {
        int(row["codigo_montante"]): str(row["nome_montante"])
        for _, row in cascata_pmo.iterrows()
    }
    comparacao = comparar_cascatas(
        cascata_pmo, cascata_cobre, codigo_para_id, nomes_pmo
    )

    ids_mapeados = set(codigo_para_id.values())
    inesperadas = cascata_cobre[
        ~cascata_cobre["id_montante"].isin(ids_mapeados)
    ].copy()
    if not inesperadas.empty:
        inesperadas["resultado"] = (
            "Usina do COBRE sem correspondência na cascata do PMO"
        )

    divergencias = comparacao[
        ~comparacao["resultado"].isin(RESULTADOS_CORRETOS)
    ].copy()

    codigos_nao_representados: set[int] = set(
        comparacao.loc[
            comparacao["id_montante_cobre"].isna(), "codigo_montante_newave"
        ].astype(int)
    )
    jusantes_ausentes = comparacao[
        (comparacao["codigo_jusante_newave"] != 0)
        & comparacao["id_jusante_cobre_esperado"].isna()
    ]
    codigos_nao_representados.update(
        jusantes_ausentes["codigo_jusante_newave"].astype(int).tolist()
    )
    nomes_por_codigo = {
        int(row["codigo_montante"]): str(row["nome_montante"])
        for _, row in cascata_pmo.iterrows()
    }
    usinas_nao_representadas = pd.DataFrame(
        [
            {"codigo_newave": codigo, "nome_usina": nomes_por_codigo.get(codigo, "-")}
            for codigo in sorted(codigos_nao_representados)
        ],
        columns=["codigo_newave", "nome_usina"],
    )

    total_corretas = int(comparacao["resultado"].isin(RESULTADOS_CORRETOS).sum())
    total_ligacoes_desviadas = int(
        (
            comparacao["resultado"]
            == "Ligação desviada em razão de usina intermediária não representada"
        ).sum()
    )
    total_divergencias_canal = int(
        (~comparacao["resultado_canal_desvio"].isin(
            {
                "Sem canal de desvio nos dois modelos",
                "Canal de desvio correto",
                "Não comparado: usina montante não representada",
            }
        )).sum()
    )

    avisos: list[str] = []
    if len(cascata_pmo) != identificacao.total_esperado:
        avisos.append(
            "A tabela de cascatas do PMO possui "
            f"{len(cascata_pmo)} relações, enquanto a identificação hidráulica "
            f"indicou {identificacao.total_esperado} usinas esperadas."
        )
    if considera_ficticias is True:
        avisos.append(
            "O PMO informa que a tabela de cascatas considera usinas fictícias. "
            "A comparação deve ser interpretada com atenção porque o COBRE pode "
            "não criar essas entidades."
        )
    if not ids_cobre_validos:
        avisos.append(
            "Os IDs do hydros.json não são contínuos a partir de zero. As ligações "
            "foram comparadas pelos valores declarados, mas a estrutura deve ser revisada."
        )

    aprovada = (
        divergencias.empty
        and inesperadas.empty
        and ids_cobre_validos
        and len(cascata_pmo) == len(cascata_cobre)
    )
    return ResultadoCascatas(
        cascata_pmo=cascata_pmo,
        cascata_cobre=cascata_cobre,
        comparacao=comparacao,
        divergencias=divergencias,
        usinas_nao_representadas=usinas_nao_representadas,
        usinas_inesperadas_cobre=inesperadas,
        total_pmo=len(cascata_pmo),
        total_cobre=len(cascata_cobre),
        total_corretas=total_corretas,
        total_divergencias=len(divergencias) + len(inesperadas),
        total_nao_representadas=len(usinas_nao_representadas),
        total_ligacoes_desviadas=total_ligacoes_desviadas,
        total_divergencias_canal=total_divergencias_canal,
        considera_ficticias_pmo=considera_ficticias,
        ids_cobre_validos=ids_cobre_validos,
        aprovada=aprovada,
        avisos=tuple(avisos),
    )


def _tabela_diferencas(resultado: ResultadoCascatas) -> pd.DataFrame:
    """Seleciona as colunas essenciais para leitura das divergências."""
    return resultado.divergencias[
        [
            "codigo_montante_newave",
            "nome_montante_newave",
            "id_montante_cobre",
            "codigo_jusante_newave",
            "nome_jusante_newave",
            "id_jusante_cobre_encontrado",
            "nome_jusante_cobre_encontrado",
            "resultado",
            "detalhe",
        ]
    ]


def mostrar_aba() -> None:
    """Renderiza a Aba 2 no Streamlit."""
    import streamlit as st

    st.subheader("Formação das cascatas hidráulicas")
    st.markdown(
        "Esta aba reconstrói a cascata do COBRE a partir de `id` e "
        "`downstream_id` do `system/hydros.json` e compara cada ligação com a "
        "tabela de cascatas apresentada pelo NEWAVE no `pmo.dat`."
    )
    st.caption(
        "As diferenças são descritas somente a partir do que os arquivos "
        "comprovam, inclusive quando uma usina do NEWAVE não foi representada no COBRE."
    )

    coluna_pmo, coluna_cobre = st.columns(2)
    with coluna_pmo:
        arquivo_pmo = st.file_uploader(
            "Arquivo NEWAVE — pmo.dat",
            type=["dat"],
            key="aba02_pmo",
        )
    with coluna_cobre:
        arquivo_hydros = st.file_uploader(
            "Arquivo COBRE — hydros.json",
            type=["json"],
            key="aba02_hydros",
        )

    if arquivo_pmo is None or arquivo_hydros is None:
        st.info("Envie os dois arquivos para iniciar a validação da cascata.")
        return

    pmo_bytes = arquivo_pmo.getvalue()
    hydros_bytes = arquivo_hydros.getvalue()
    hash_arquivos = hashlib.sha256()
    hash_arquivos.update(pmo_bytes)
    hash_arquivos.update(b"\0CASCATA_HYDROS\0")
    hash_arquivos.update(hydros_bytes)
    assinatura = hash_arquivos.hexdigest()
    chave_cache = "aba02_resultado_cache"

    executar = st.button(
        "Executar validação da cascata",
        type="primary",
        key="aba02_executar",
    )
    if executar:
        try:
            with st.spinner("Reconstruindo e comparando as cascatas..."):
                resultado_calculado = validar_cascatas(pmo_bytes, hydros_bytes)
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
        st.info("Clique em Executar validação da cascata para comparar os arquivos.")
        return
    resultado: ResultadoCascatas = cache["resultado"]

    for aviso in resultado.avisos:
        st.warning(aviso)

    metricas = st.columns(6)
    metricas[0].metric("Relações no PMO", resultado.total_pmo)
    metricas[1].metric("Usinas no COBRE", resultado.total_cobre)
    metricas[2].metric("Relações corretas", resultado.total_corretas)
    metricas[3].metric("Diferenças", resultado.total_divergencias)
    metricas[4].metric("Usinas não representadas", resultado.total_nao_representadas)
    metricas[5].metric("Ligações desviadas", resultado.total_ligacoes_desviadas)

    if resultado.aprovada:
        st.success(
            "Validação aprovada: a cascata reconstruída no COBRE coincide com "
            "a cascata apresentada pelo NEWAVE."
        )
    else:
        st.error(
            "Validação reprovada: foram identificadas diferenças entre a "
            "cascata do NEWAVE e a topologia representada no COBRE."
        )

    if not resultado.usinas_nao_representadas.empty:
        st.markdown("#### Usinas do NEWAVE não representadas no COBRE")
        st.dataframe(
            resultado.usinas_nao_representadas,
            hide_index=True,
            use_container_width=True,
        )

    if not resultado.divergencias.empty:
        st.markdown("#### Diferenças identificadas")
        st.dataframe(
            _tabela_diferencas(resultado),
            hide_index=True,
            use_container_width=True,
        )

    if not resultado.usinas_inesperadas_cobre.empty:
        st.markdown("#### Usinas do COBRE sem correspondência no PMO")
        st.dataframe(
            resultado.usinas_inesperadas_cobre,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Cascatas no formato montante–jusante")
    tabela_pmo, tabela_cobre = st.tabs(
        ["Cascata apresentada no PMO", "Cascata reconstruída do COBRE"]
    )
    with tabela_pmo:
        st.dataframe(
            resultado.cascata_pmo,
            hide_index=True,
            use_container_width=True,
        )
    with tabela_cobre:
        st.dataframe(
            resultado.cascata_cobre,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Comparação completa")
    filtro = st.selectbox(
        "Mostrar",
        [
            "Somente diferenças",
            "Todas as relações",
            "Somente ligações corretas",
            "Somente usinas não representadas",
            "Somente canais de desvio divergentes",
        ],
        key="aba02_filtro",
    )
    exibicao = resultado.comparacao
    if filtro == "Somente diferenças":
        exibicao = resultado.divergencias
    elif filtro == "Somente ligações corretas":
        exibicao = exibicao[exibicao["resultado"].isin(RESULTADOS_CORRETOS)]
    elif filtro == "Somente usinas não representadas":
        exibicao = exibicao[
            exibicao["resultado"].str.contains("não representada", case=False)
        ]
    elif filtro == "Somente canais de desvio divergentes":
        exibicao = exibicao[
            ~exibicao["resultado_canal_desvio"].isin(
                {
                    "Sem canal de desvio nos dois modelos",
                    "Canal de desvio correto",
                    "Não comparado: usina montante não representada",
                }
            )
        ]
    st.dataframe(exibicao, hide_index=True, use_container_width=True)

    st.markdown("#### Exportações")
    download_1, download_2, download_3 = st.columns(3)
    with download_1:
        st.download_button(
            "Baixar comparação completa",
            data=gerar_csv(resultado.comparacao),
            file_name="validacao_cascatas_hidraulicas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_2:
        st.download_button(
            "Baixar somente diferenças",
            data=gerar_csv(resultado.divergencias),
            file_name="diferencas_cascatas_hidraulicas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_3:
        st.download_button(
            "Baixar cascata reconstruída do COBRE",
            data=gerar_csv(resultado.cascata_cobre),
            file_name="cascata_reconstruida_cobre.csv",
            mime="text/csv",
            use_container_width=True,
        )
