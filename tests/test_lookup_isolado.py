"""Gate do lookup isolado (src/validacao/lookup_isolado.py)."""

import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import eda_utils as u
from validacao import lookup_isolado as li

# Medidos em 30/09/2026 sobre os 994 registros do índice. Melhorou: mova o piso
# na mesma mudança. Regrediu: o índice ou a regra de resolução piorou.
PISO_REGISTROS_NO_INDICE = 994
PISO_ID_UNICO_CORRETO = 842
PISO_AMBIGUIDADE_COM_ESPERADO = 152


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (RAIZ / "data" / "desafio1_bracis.db").is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.medicao = li.executar(u.conectar_base())
        cls.resultado = cls.medicao["resultado"]

    def test_nenhum_registro_do_indice_some(self):
        self.assertGreaterEqual(sum(self.resultado.values()), PISO_REGISTROS_NO_INDICE)

    def test_nenhuma_resposta_errada(self):
        for desfecho in ("real_errado", "ambiguo_sem_esperado", "inventada"):
            with self.subTest(desfecho=desfecho):
                self.assertEqual(self.resultado[desfecho], 0, self.medicao["exemplos"])

    def test_todo_registro_e_correto_ou_ambiguo_com_o_esperado(self):
        aceitaveis = sum(self.resultado[d] for d in li.ACEITAVEIS)
        self.assertEqual(aceitaveis, sum(self.resultado.values()))

    def test_id_unico_e_ambiguidades_nao_regridem(self):
        self.assertGreaterEqual(self.resultado["ok"], PISO_ID_UNICO_CORRETO)
        self.assertGreaterEqual(self.resultado["ambiguo_com_esperado"], PISO_AMBIGUIDADE_COM_ESPERADO)


if __name__ == "__main__":
    unittest.main()
