"""Contrato do gerador opcional de ZIP."""

from zipfile import ZipFile

from empacotar_submissao import gerar_zip


def test_gerar_zip_contem_somente_o_csv(tmp_path):
    csv = tmp_path / "resultado.csv"
    csv.write_text("documento_id,citacoes\ndoc,-\n", encoding="utf-8")
    destino = tmp_path / "artefatos" / "submission.zip"

    gerar_zip(csv, destino)

    with ZipFile(destino) as arquivo:
        assert arquivo.namelist() == ["submission.csv"]
        assert arquivo.read("submission.csv") == csv.read_bytes()
