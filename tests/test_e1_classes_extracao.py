"""E1: classes e modificadores que o normalizador conhecia e o extrator não iniciava.

`AIRR`, `Ag-AIRR`, `RRAg` (TST), `Embargos Infringentes e de Nulidade` (STM) e o
modificador `PExt no` (STJ) não abriam um candidato: a citação nem chegava ao
resolvedor (`nao_extraiu`) ou chegava sem o modificador e resolvia o acórdão
errado. Os exemplos vêm de acórdãos reais da base, sem duplicata.
"""

import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor

MOLDURA = "Nesse sentido, confira-se o que decidiu esta Corte no {}, cujo entendimento se aplica ao caso.\n"

# (classe como se escreve, número, id do acórdão na base)
AIRR = ("AIRR", "0000517-84.2013.5.09.0127", 1975071157)
AG_AIRR = ("Ag-AIRR", "0001366-63.2011.5.02.0081", 945032914)
RRAG = ("RRAg", "0001751-46.2017.5.09.0003", 1935164504)
EI_NULIDADE = ("Embargos Infringentes e de Nulidade", "7000380-08.2023.7.00.0000/DF", 2382649285)


def formas(classe: str, numero: str) -> list[str]:
    """Formas de citar: `CLASSE nº N` e, nas siglas do TST, `CLASSE-N`."""
    base = [f"{classe} nº {numero}"]
    if classe in {"AIRR", "Ag-AIRR", "RRAg"}:
        base.append(f"{classe}-{numero}")
    return base


class BaseCanonica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.resolvedor = Resolvedor(cls.con)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def citacoes(self, citacao: str) -> list[dict]:
        return processar_documento("sintetico", MOLDURA.format(citacao), self.resolvedor)["citacoes"]


class ClassesQueOExtratorNaoIniciava(BaseCanonica):
    def test_cada_classe_e_extraida_com_a_classe_no_span_e_resolve_o_acordao(self):
        for classe, numero, id_esperado in (AIRR, AG_AIRR, RRAG, EI_NULIDADE):
            for forma in formas(classe, numero):
                with self.subTest(forma=forma):
                    citacoes = self.citacoes(forma)
                    self.assertEqual(len(citacoes), 1, f"citações detectadas: {[c['trecho'] for c in citacoes]}")
                    c = citacoes[0]
                    self.assertTrue(c["trecho"].lower().startswith(classe.lower()),
                                    f"o span deve começar na classe: {c['trecho']!r}")
                    self.assertEqual(c["classificacao"], "real")
                    self.assertEqual(c["resolucao"], {"id_canonico": str(id_esperado)})

    def test_numero_inexistente_dessas_classes_continua_inventada(self):
        # Extrair a classe não pode afrouxar a consulta: número que não existe é inventada.
        for forma in ("AIRR nº 9999999-99.2013.5.09.0127", "Ag-AIRR-9999999-99.2011.5.02.0081",
                      "RRAg nº 9999999-99.2017.5.09.0003",
                      "Embargos Infringentes e de Nulidade nº 9999999-99.2023.7.00.0000/DF"):
            with self.subTest(forma=forma):
                citacoes = self.citacoes(forma)
                self.assertEqual(len(citacoes), 1)
                self.assertEqual(citacoes[0]["classificacao"], "inventada")

    def test_siglas_novas_nao_casam_com_palavras_parecidas(self):
        # `airr`, `rrag` e `pext` são marcadores exatos: sem aproximação por similaridade.
        for texto in (
            "A companhia aérea Air France cobrou R$ 1.234,56 pela passagem.\n",
            "O ar condicionado do fórum e o arr do contrato não são citações.\n",
            "Os embargos infringentes são cabíveis quando o acórdão não for unânime.\n",
            "A sigla rrag aparece em planilhas internas, sem número de processo.\n",
        ):
            with self.subTest(texto=texto.strip()):
                citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
                self.assertEqual([c["trecho"] for c in citacoes], [])


class ModificadorPExt(BaseCanonica):
    """A6: o extrator cortava `PExt no` do span e resolvia o acórdão errado."""

    def test_pext_entra_no_span_e_resolve_o_acordao_do_pext(self):
        citacoes = self.citacoes("PExt no Recurso em Habeas Corpus nº 90.861/RS do STJ")
        self.assertEqual(len(citacoes), 1)
        c = citacoes[0]
        self.assertTrue(c["trecho"].startswith("PExt"), f"o span deve começar em PExt: {c['trecho']!r}")
        self.assertEqual(c["classificacao"], "real")
        # 2679473072 é o "PExt no RHC"; 2566482355 é o RHC simples, de mesmo número.
        self.assertEqual(c["resolucao"], {"id_canonico": "2679473072"})

    def test_pext_escrito_por_extenso_ou_com_ocr_nao_e_exigido(self):
        # Só o modificador explícito `PExt` é reconhecido; não inventamos sinônimos.
        citacoes = self.citacoes("Recurso em Habeas Corpus nº 90.861/RS do STJ")
        self.assertEqual(len(citacoes), 1)
        self.assertFalse(citacoes[0]["trecho"].startswith("PExt"))


if __name__ == "__main__":
    unittest.main()
