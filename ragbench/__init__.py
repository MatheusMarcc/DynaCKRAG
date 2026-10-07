"""ragbench — RAG-Critic e DynaKRAG implementados e avaliados isoladamente.

Cada artigo tem o seu pacote (``ragbench.ragcritic`` e ``ragbench.dynakrag``), com os seus baselines, ablações
e tabelas no formato do artigo. O que é infraestrutura comum (cliente do Ollama com cache, retrieval, datasets,
métricas, relatórios) fica em ``ragbench.core``. Por cima dos dois:

* ``ragbench.analysis`` — em quais perguntas cada método falha, em que etapa, e se falham nas mesmas;
* ``ragbench.ui``       — o painel no navegador (``python -m ragbench ui``).

Uso: ``python -m ragbench --help``
"""

__version__ = "1.3.0"
