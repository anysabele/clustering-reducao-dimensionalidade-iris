# Aprendizado não supervisionado na base Iris: 7 algoritmos de agrupamento e 4 métodos de redução de dimensionalidade

Projeto da disciplina de **Aprendizado de Máquina Não Supervisionado** (CIn/UFPE, Prof. Francisco de A. T. de Carvalho), realizado pela equipe 6 (dupla). O notebook compara **sete algoritmos de agrupamento** (Questão 1) e **quatro métodos de redução de dimensionalidade** (Questão 2) sobre a base Iris, com protocolo experimental rigoroso e verificações independentes de cada implementação.

> **Pergunta central:** que estrutura os algoritmos encontram na Iris **sem usar os rótulos** e como eles concordam entre si?

## Dados

**Iris (UCI)**: 150 objetos, 4 atributos numéricos (comprimento e largura de sépala e pétala, em cm) e 3 espécies de 50 objetos. O notebook usa o `iris.data` original fornecido pelo professor e confere o arquivo por **SHA-256**. Se o arquivo não estiver na pasta de dados, ele é baixado da UCI.

Dois pontos tratados com cuidado:
- O `iris.data` difere da versão do scikit-learn em **2 objetos** (35 e 38), por erros documentados no `iris.names`. A versão original foi mantida.
- Há **5 objetos duplicados** (trio 10/35/38 e par 102/143). Eles exigem sorteio entre pontos distintos na inicialização, atenção às fusões de altura zero no single linkage e cuidado na contagem de vizinhos do DBSCAN.

Os rótulos (espécies) servem **apenas para avaliar** os resultados e colorir os gráficos. Nenhuma escolha de hiperparâmetro ou de pré-processamento usa os rótulos.

## Decisões de projeto

| Decisão | Escolha |
|---|---|
| Pré-processamento | **Atributos brutos, em cm, sem padronização.** Os quatro atributos têm a mesma unidade, e a padronização elevaria o peso da largura da sépala de 4,1% para 25% na distância Euclidiana², mascarando a estrutura das pétalas |
| Silhueta | Distância Euclidiana para os 7 algoritmos (comparabilidade). Para o single linkage e o KFCM, a silhueta na métrica própria é calculada como complemento |
| Fuzzy/EM → partição crisp | Maior pertinência ou probabilidade |
| Ruído do DBSCAN | Silhueta e ARI calculados sem o ruído, com a convenção alternativa (ruído = 0) reportada como complemento |
| Reprodutibilidade | Semente global 2026, com `SeedSequence` por par (k, execução), igual entre algoritmos |

## Questão 1: Agrupamento

| Algoritmo | Particularidades |
|---|---|
| **K-means** | Lloyd, inicialização por sorteio de objetos distintos, `tol = 0` |
| **K-medoids** | FasterPAM sobre matriz de distâncias Euclidianas explícita |
| **FCM** (fuzzy k-means) | Bezdek (1981), `m = 2`, via `scikit-fuzzy` |
| **KFCM** | **Implementação própria** da formulação no espaço de características (KFCM-F), kernel gaussiano com largura pela heurística da mediana |
| **EM** (mistura gaussiana) | Covariância completa, regularização `h²/12` ligada à precisão dos dados |
| **Single linkage** | Distância city-block em aritmética exata (empates exatos) |
| **DBSCAN** | `min_samples` = 8, ε escolhido por busca |

### Protocolo

1. **Particionais** (k-means, k-medoids, FCM, KFCM, EM): 10 execuções para cada `k` de 1 a 5, e a **melhor execução** de cada `k` é escolhida pela função objetivo.
2. **Single linkage:** uma execução, com cortes do dendrograma em `k` de 1 a 5.
3. **DBSCAN:** uma execução (o número de grupos é definido pelo algoritmo).
4. **Avaliação comum:** silhueta de cada partição, `k* = argmax Sil(k)`, e depois ARI e tabela de contingência em `k*`.
5. **Busca de hiperparâmetros sem rótulos**, antes das execuções oficiais: configurações dentro de 1 erro-padrão do melhor escore são equivalentes, e a configuração *a priori* é mantida quando possível. As buscas ficam separadas dos resultados oficiais.

### Resultados

| Algoritmo | k* | Sil(k*) | ARI (espécies) | Partição |
|---|---|---|---|---|
| K-means | 2 | 0,681 | 0,540 | 97 / 53 |
| FCM | 2 | 0,681 | 0,540 | 97 / 53 |
| K-medoids | 2 | 0,685 | 0,558 | 99 / 51 |
| KFCM | 2 | 0,685 | 0,558 | 99 / 51 |
| EM (gaussianas) | 2 | 0,686 | **0,568** | 100 / 50 |
| Single linkage | 2 | 0,686 | **0,568** | 100 / 50 |
| DBSCAN (ε = 0,65) | 2 grupos | 0,724* | 0,595* | 50 / 90 + 10 de ruído |

\* Calculados só nos 140 objetos agrupados. Com ruído = 0 na silhueta e ruído como rótulo −1 no ARI, os valores são 0,676 e 0,547. **Não são diretamente comparáveis** às demais linhas.

**Principais observações**
- **Os seis algoritmos que escolhem `k` pela silhueta escolheram `k* = 2`**, e o DBSCAN também encontrou dois grupos densos. A estrutura dominante é a das *setosas* isoladas, e versicolor e virginica reunidas (a lacuna de 1,9 a 3,0 cm no comprimento da pétala já indicava isso na análise exploratória).
- **O ARI máximo possível com 2 grupos e 3 espécies é 0,568.** EM e single linkage atingem esse teto, e k-medoids e KFCM ficam a um objeto, k-means e FCM a três. Portanto, a distância até 1 vem sobretudo de `k* = 2`, e não de erros de atribuição.
- **As seis partições se reduzem a três** e diferem em no máximo 3 objetos de fronteira (ARI entre elas ≥ 0,920). Nenhum método se mostrou superior, e as diferenças refletem como cada um define a fronteira.
- **Busca de hiperparâmetros:** cinco algoritmos mantiveram a configuração *a priori*, com a mesma partição. Só o DBSCAN mudou (ε de 0,548 para 0,65 cm, reduzindo o ruído de 15 para 10 objetos).
- **Estabilidade:** as 250 execuções convergiram. FCM e KFCM foram os mais estáveis até `k = 3`, e o EM já apresentou instabilidade em `k = 2`. Para `k ≥ 4`, todos perdem estabilidade, o que justifica as 10 execuções.

### Verificações das implementações

Cada algoritmo é conferido de forma independente: objetivo recalculado, ponto fixo de Lloyd, ótimo local (todas as trocas) e global (busca exaustiva para `k ≤ 3`) no k-medoids, condições de otimalidade e monotonicidade no FCM e KFCM, etapas E e M do EM, equivalência com a árvore geradora mínima no single linkage e reconstrução da definição do DBSCAN. O KFCM é validado contra o FCM do `scikit-fuzzy` (kernel linear, diferença máxima de 2×10⁻¹²). Há ainda testes da infraestrutura com dados sintéticos e algoritmo fictício.

## Questão 2: Redução de dimensionalidade

| Método | Configuração | Saída própria |
|---|---|---|
| **PCA** | Matriz de covariâncias (dados brutos), 4 componentes, SVD exata | CP1 explica 92,5% e CP2 5,3% (**97,8% juntas**) |
| **Kernel PCA** | Gaussiano, mesma largura do KFCM (σ² = 2,785 cm²) | 146 autovalores positivos; 78,9% nas duas primeiras (no espaço do kernel) |
| **t-SNE** | Perplexidade 30, inicialização PCA, método exato | KL final = 0,122 |
| **UMAP** | `n_neighbors` = 15, `min_dist` = 0,1 | Grafo fuzzy com 2.678 entradas não nulas |

Comparação sem usar rótulos (confiabilidade das vizinhanças × correlação de Spearman entre distâncias):

| Método | Preservação local | Spearman global |
|---|---|---|
| PCA | 0,983 | **0,996** |
| Kernel PCA | 0,940 | 0,939 |
| t-SNE | **0,989** | 0,973 |
| UMAP | 0,980 | 0,956 |

Os quatro métodos mostram a mesma estrutura: setosas isoladas e versicolor e virginica sobrepostas, o que confirma visualmente o `k* = 2` da Questão 1. O notebook inclui também a **análise de sensibilidade** do t-SNE à perplexidade (5, 30, 50) e do UMAP a `n_neighbors` (5, 15, 50). A escolha do método depende do objetivo: PCA para interpretar os atributos, Kernel PCA para estruturas não lineares, t-SNE e UMAP para vizinhanças locais.

## Limitações

- Base pequena (150 objetos) e bem estudada, em que a estrutura de 2 grupos é muito marcada.
- Os resultados dependem de decisões explícitas: atributos brutos, silhueta Euclidiana e conversão crisp.
- A faixa de tolerância de 1 erro-padrão na busca de hiperparâmetros é uma heurística de parcimônia, e não prova de equivalência estatística.
- t-SNE e UMAP: só as vizinhanças locais são confiáveis, e distâncias entre grupos e tamanhos aparentes não têm interpretação.

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy pandas scipy scikit-learn matplotlib seaborn kmedoids scikit-fuzzy umap-learn jupyter
jupyter notebook "amns_iris.ipynb"
```

- O notebook foi feito para o **Google Colab** (instala `kmedoids`, `scikit-fuzzy` e `umap-learn` com versões fixadas) e também roda localmente.
- Coloque `iris.data` e `iris.names` na pasta `dados/` (ou deixe o notebook baixar da UCI).

## Estrutura

```
.
├── amns_iris.ipynb
├── dados/
│   ├── iris.data
│   └── iris.names
├── amns_iris_relatorio.pdf  
└── README.md
```

## Tecnologias

Python · NumPy · pandas · SciPy · scikit-learn · scikit-fuzzy · kmedoids · umap-learn · Matplotlib · Seaborn

## Autoria

- Anysabele de Paula Barbosa Santos
- Millena Ferreira Marçal das Neves
