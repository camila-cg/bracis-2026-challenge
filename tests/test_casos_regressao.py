"""Reexecuta os casos registrados em data/validation/casos_regressao.csv.

Cada linha é um exemplo que já gerou erro, ou um controle contra excesso de
correção. As colunas legíveis por máquina dizem o que o pipeline deve produzir:

- classe_esperada: real, inventada ou incompleta (vazio = não verificar);
- id_esperado: id_canonico esperado (vazio = não verificar);
- n_citacoes: quantas citações devem ser detectadas (vazio = não verificar);
- trecho_contem: texto que o span detectado deve conter (vazio = não verificar);
- trecho_nao_contem: texto que o span detectado **não** deve conter (vazio = não verificar);
  pega o span que engole a frase seguinte, que `trecho_contem` não vê;
- modo_teste: `regressao` (deve passar hoje), `aberto` (problema conhecido, deve
  continuar falhando) ou `pendente` (sem critério definido, não é executado).

Um caso `aberto` que passa a funcionar faz o teste falhar de propósito: é o aviso
para mudar a linha para `regressao` e atualizar a coluna `status` com o commit.
"""

import csv
import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor

CASOS = RAIZ / "data" / "validation" / "casos_regressao.csv"
COLUNAS = (
    "exemplo", "resultado esperado", "resultado obtido", "motivo", "status",
    "classe_esperada", "id_esperado", "n_citacoes", "trecho_contem", "trecho_nao_contem", "modo_teste",
)
MODOS = {"regressao", "aberto", "pendente"}
CLASSES = {"", "real", "inventada", "incompleta"}


def carregar_casos() -> list[dict]:
    with open(CASOS, encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def divergencias(caso: dict, citacoes: list[dict]) -> list[str]:
    """Diferenças entre o que o pipeline produziu e o que a linha espera."""
    problemas = []
    esperado_n = caso["n_citacoes"]
    if esperado_n != "" and len(citacoes) != int(esperado_n):
        problemas.append(f"citações: esperado {esperado_n}, obtido {len(citacoes)}")

    if not citacoes:
        if caso["classe_esperada"] or caso["id_esperado"] or caso["trecho_contem"]:
            problemas.append("nenhuma citação detectada")
        return problemas

    principal = max(citacoes, key=lambda c: c["fim"] - c["inicio"])
    if caso["classe_esperada"] and principal["classificacao"] != caso["classe_esperada"]:
        problemas.append(
            f"classe: esperado {caso['classe_esperada']}, obtido {principal['classificacao']}"
        )
    if caso["id_esperado"]:
        obtido = str((principal["resolucao"] or {}).get("id_canonico"))
        if obtido != caso["id_esperado"]:
            problemas.append(f"id: esperado {caso['id_esperado']}, obtido {obtido}")
    if caso["trecho_contem"] and caso["trecho_contem"] not in principal["trecho"]:
        problemas.append(
            f"span: esperado conter {caso['trecho_contem']!r}, obtido {principal['trecho']!r}"
        )
    if caso["trecho_nao_contem"] and caso["trecho_nao_contem"] in principal["trecho"]:
        problemas.append(
            f"span: não devia conter {caso['trecho_nao_contem']!r}, obtido {principal['trecho']!r}"
        )
    return problemas


class FormatoDoRegistro(unittest.TestCase):
    def test_colunas_e_valores_sao_validos(self):
        if not CASOS.is_file():
            self.skipTest("data/validation/casos_regressao.csv ausente")
        casos = carregar_casos()
        self.assertGreater(len(casos), 0)
        self.assertEqual(tuple(casos[0].keys()), COLUNAS)
        exemplos = [caso["exemplo"] for caso in casos]
        self.assertEqual(len(exemplos), len(set(exemplos)), "exemplo repetido no registro")
        for caso in casos:
            with self.subTest(exemplo=caso["exemplo"][:70]):
                for coluna in ("exemplo", "resultado esperado", "resultado obtido", "motivo", "status"):
                    self.assertTrue(caso[coluna].strip(), f"coluna {coluna} vazia")
                self.assertIn(caso["modo_teste"], MODOS)
                self.assertIn(caso["classe_esperada"], CLASSES)
                self.assertTrue(caso["id_esperado"] == "" or caso["id_esperado"].isdigit())
                self.assertTrue(caso["n_citacoes"] == "" or caso["n_citacoes"].isdigit())
                if caso["modo_teste"] != "pendente":
                    criterios = (
                        caso["classe_esperada"], caso["id_esperado"],
                        caso["n_citacoes"], caso["trecho_contem"], caso["trecho_nao_contem"],
                    )
                    self.assertTrue(any(criterios), "caso executável sem nenhum critério")


class CasosDeRegressao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        if not CASOS.is_file():
            raise unittest.SkipTest("data/validation/casos_regressao.csv ausente")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.resolvedor = Resolvedor(cls.con)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def test_casos_registrados(self):
        for caso in carregar_casos():
            modo = caso["modo_teste"]
            if modo == "pendente":
                continue
            with self.subTest(exemplo=caso["exemplo"][:70], modo=modo):
                citacoes = processar_documento("regressao", caso["exemplo"], self.resolvedor)["citacoes"]
                problemas = divergencias(caso, citacoes)
                if modo == "regressao":
                    self.assertEqual(problemas, [], f"regressão em: {caso['exemplo']}")
                else:
                    self.assertTrue(
                        problemas,
                        f"o caso aberto passou a funcionar: mude modo_teste para 'regressao' "
                        f"e atualize o status de: {caso['exemplo']}",
                    )


if __name__ == "__main__":
    unittest.main()
