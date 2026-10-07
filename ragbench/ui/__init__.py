"""Painel local do ragbench: uma página no navegador para rodar os testes, ver as falhas de cada método,
cruzar as falhas entre métodos e explorar os dados (perguntas, gabaritos, corpus e o RAG-Error Bench).

    python -m ragbench ui                         # abre http://127.0.0.1:8765
    python -m ragbench ui --export painel.html    # um HTML único com os resultados (sem servidor)

Só a biblioteca padrão do Python: ``server`` (HTTP + controle da rodada), ``data`` (o que a página mostra) e
``static/`` (HTML, CSS e JavaScript puros, sem nada da internet).
"""
