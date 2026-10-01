"""Gate do estresse (src/validacao/estresse.py): ruído e troca de dígito nas citações do dev.

O estresse perturba as citações reais de acórdão do dev com a mesma seed, então o
texto ruidoso não depende do pipeline: uma queda aqui é do código. O mínimo do
ruído já caiu de 1,0865 para 1,0811 sem que nenhum teste acusasse (A11); este teste
existe para isso não se repetir.

Pisos medidos em 30/09/2026. Melhorou: suba o piso na mesma mudança.
"""

import random
import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

PISO_ORIGINAL = 1.099
PISO_RUIDO_MINIMO = 1.086   # mínimo entre as seeds do ruído abaixo
PISO_DIGITO_MINIMO = 1.098
# Só as seeds que já mostraram regressão: o estresse inteiro (3 seeds por cenário)
# leva cerca de 1 min; estas 4 execuções, cerca de metade. A seed 0 do ruído nunca mudou.
SEEDS_RUIDO = (1, 2)
SEEDS_DIGITO = (0,)


class Estresse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dados = RAIZ / "data"
        if not (dados / "desafio1_bracis.db").is_file() or not (dados / "goldenset_offsets.csv").is_file():
            raise unittest.SkipTest("base canônica ou gabarito do dev ausentes em data/")
        import eda_utils as u
        from validacao import estresse as e
        import pipeline as p

        cls.e = e
        metrica = e.carregar_modulo(dados / "kaggle_metric.py", "kaggle_metric")
        gabarito = e.carregar_gabarito()
        textos = {f.stem: p.ler_texto(f) for f in sorted((dados / "txt").glob("*.txt"))}
        con = sqlite3.connect(u._uri_sqlite((dados / "desafio1_bracis.db").resolve()) + "?immutable=1", uri=True)
        resolvedor = p.Resolvedor(con)

        def pontuar(cenario: str, seed: int) -> float:
            documentos, solucao = e.montar_cenario(cenario, textos, gabarito, random.Random(seed))
            return metrica.avaliar(solucao, e.predizer(documentos, resolvedor), row_id="documento_id")["score_final"]

        cls.original = pontuar("original", 0)
        cls.ruido = {seed: pontuar("ruido", seed) for seed in SEEDS_RUIDO}
        cls.digito = {seed: pontuar("digito", seed) for seed in SEEDS_DIGITO}

    def test_cenario_original_bate_com_o_dev(self):
        self.assertGreaterEqual(self.original, PISO_ORIGINAL)

    def test_ruido_nao_regride_em_nenhuma_seed(self):
        for seed, score in self.ruido.items():
            with self.subTest(seed=seed):
                self.assertGreaterEqual(score, PISO_RUIDO_MINIMO, f"ruído seed {seed}: {score:.4f}")

    def test_troca_de_digito_nao_regride_em_nenhuma_seed(self):
        for seed, score in self.digito.items():
            with self.subTest(seed=seed):
                self.assertGreaterEqual(score, PISO_DIGITO_MINIMO, f"dígito seed {seed}: {score:.4f}")


if __name__ == "__main__":
    unittest.main()
