# Finance RAG : RAG et agent Azure AI mesurés sur FinanceBench

Un système de question-réponse sur 84 rapports financiers (10-K, 10-Q, 8-K, communiqués de
résultats ; 12 013 pages), construit sur **Azure AI Search**, **Azure OpenAI** et
**Microsoft Foundry Agent Service**, puis **évalué sur les 150 questions** du jeu
[FinanceBench](https://github.com/patronus-ai/financebench) open source.

Chaque choix d'architecture a été comparé à l'alternative sur les mêmes questions, avec des tests
statistiques appariés. Les décisions et leurs preuves sont consignées dans les
[ADR](adr/README.md).

## Résultats

| Configuration | Taux de réponse | Completeness (1-5) | Chiffres exacts (sur 52) | Coût par question |
|---|---|---|---|---|
| C1 : vectoriel, 5 passages | 53 % | 3,12 | 24 | 0,016 USD |
| C3 : hybride, 10 passages | 65 % | 3,37 | 31 | 0,025 USD |
| C5 : hybride + semantic ranker | 69 % | 3,69 | 37 | 0,026 USD |
| **Agent : C5 en outil, plusieurs recherches, calculatrice** | **91 %** | **4,25** | **46** | 0,039 USD |

- **L'agent répond à 91 % des questions contre 69 % pour le meilleur RAG, sans perdre en
  précision** : 46 chiffres exacts sur 48 donnés. Gain significatif sur la completeness
  (+0,56, IC 95 % [+0,31 ; +0,83]) et sur les chiffres (+10 / −1, McNemar p = 0,012).
- **La trajectoire de l'agent est jugée bonne** par les évaluateurs d'agents : Tool Call Accuracy
  4,66 sur 5, Intent Resolution 4,81 sur 5, Task Adherence 84 % (échecs surtout de forme).
- **Le semantic ranker améliore la réponse à lot de candidats égal** : +0,32 de completeness,
  IC 95 % [+0,15 ; +0,51] ([ADR-0003](adr/0003-hybride-semantic-ranker.md)).
- **Les filtres extraits de la question** font passer la recherche lexicale de 49 % à 92 % de bons
  documents dans le top 5 ([ADR-0002](adr/0002-filtres-metadonnees-question.md)).
- **L'analyse de mise en page (Layout) ne gagne pas** face à l'extraction native sur ce corpus
  ([ADR-0001](adr/0001-index-extraction-native.md)).
- **Le système est précis quand il répond** : 90 à 94 % de chiffres exacts pour les configurations
  RAG, 96 % pour l'agent. La principale faiblesse mesurée est l'abstention, liée au retrieval.

Completeness : évaluateur Response Completeness d'Azure AI Evaluation, avec un juge distinct du
générateur. Chiffres exacts : contrôle déterministe sur les questions dont la réponse est un
nombre. Détails dans l'[ADR-0005](adr/0005-protocole-evaluation.md).

## Architecture

```
question
   │
   ├─► filtre entreprise / exercice ± 1 (règles, texte de la question seul)
   │
   ├─► Azure AI Search : BM25 + vecteur (fusion RRF) → 20 candidats → semantic ranker → 10 passages
   │       index de chunks (4 000 caractères), embeddings text-embedding-3-small (1 024 dimensions)
   │
   ├─► gpt-5.2 (Responses API, sortie structurée) : réponse, citations, extraits mot pour mot
   │
   └─► contrôles du code : extraits vérifiés dans les passages, statut de la réponse
                           (verified, partially_verified, unsupported, insufficient_evidence…)

Agent (Foundry Agent Service, prompt agent gpt-5.2)
   list_filings ─┐
   search_filings├─ outils fonction exécutés par l'application, budgets par question
   calculate ────┘   (6 recherches, 12 appels, 10 tours, 0,10 USD)
```

**Sécurité et fiabilité** : authentification Microsoft Entra sans clé ; délimiteurs de documents,
balises neutralisées et garde-fou de déploiement contre l'injection indirecte de prompt ;
calculatrice sans `eval` ; aucune donnée d'évaluation dans les prompts
([ADR-0004](adr/0004-generation-ancree-citations-verifiees.md),
[ADR-0006](adr/0006-agent-foundry-outils-fonction.md)).

**Observabilité** : traces OpenTelemetry selon les conventions sémantiques GenAI (tokens, tokens de
raisonnement, coût) pour chaque étape, chaque tour d'agent et chaque appel d'outil
([ADR-0007](adr/0007-observabilite-opentelemetry.md)).

## Installation

Python 3.12+ et [`uv`](https://docs.astral.sh/uv/) sont nécessaires.

```bash
uv sync --locked
cp .env.example .env
az login
```

Les outils utilisent `DefaultAzureCredential` ; `AZURE_TOKEN_CREDENTIALS=AzureCliCredential`
sélectionne la session Azure CLI. Le fichier `.env` reste local et n'est jamais versionné.

| Variable | Rôle |
|---|---|
| `AZURE_SEARCH_SERVICE_ENDPOINT` | Service Azure AI Search |
| `AZURE_SEARCH_SEMANTIC_CONFIGURATION` | Configuration sémantique de l'index |
| `AZURE_OPENAI_ENDPOINT` | Ressource Azure OpenAI / Foundry (sans `/openai/v1`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Déploiement d'embeddings |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Déploiement du modèle de génération |
| `AZURE_OPENAI_JUDGE_DEPLOYMENT` | Déploiement du juge d'évaluation, distinct du générateur |
| `FOUNDRY_PROJECT_ENDPOINT` | Projet Foundry (agents) |
| `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_STORAGE_CONTAINER_NAME` | Stockage des PDF |

Rôles nécessaires pour l'identité utilisée : lecture des index Search, inférence sur la ressource
Foundry, et Foundry User sur le projet pour les agents.

## Utilisation

```bash
# Diagnostic de la configuration et des accès
uv run ai103 doctor --quick

# Une question au RAG (configuration C5)
uv run python scripts/ask.py "What is the FY2018 capital expenditure amount for 3M?" \
    --mode semantic_hybrid --k 10 --pool 20

# Agent : simulation de la définition, création dans Foundry, question
uv run python scripts/agent.py create
uv run python scripts/agent.py create --apply
uv run python scripts/agent.py ask "Does 3M have a healthy quick ratio for Q2 of FY2023?"

# Campagnes sur les 150 questions, avec plafond de coût, journal et traces
uv run python scripts/rag_run.py --limit 150 --mode semantic_hybrid --k 10 --pool 20 \
    --output runs/C5
uv run python scripts/agent.py run --output runs/agent
uv run python scripts/agent_monitor.py --run runs/agent

# Évaluation (Azure AI Evaluation + chiffres) et comparaison appariée
uv run python scripts/evaluate_rag.py --run runs/C5
uv run python scripts/evaluate_rag.py --run runs/agent
uv run python scripts/compare_rag_configs.py runs/C5 runs/agent --output runs/comparison.json

# Évaluateurs d'agents (Tool Call Accuracy, Task Adherence, Intent Resolution)
uv run python scripts/evaluate_agent.py --run runs/agent
```

Les commandes qui écrivent dans Azure sont en simulation par défaut et demandent `--apply`.
Les campagnes s'arrêtent d'elles-mêmes au-delà d'un plafond de coût (`--max-cost-usd`).

Autres outils : benchmark du retrieval (`scripts/evaluate_retrieval.py`), filtres de métadonnées
(`scripts/evaluate_query_filters.py`), labels de pertinence des chunks
(`scripts/prepare_retrieval_qrels.py`, `scripts/score_retrieval_qrels.py`), revérification hors
ligne des citations (`scripts/recheck_citations.py`), diagnostic de Prompt Shields
(`scripts/check_prompt_shields.py`). Chaque script documente ses options avec `--help`.

### Ingestion et enrichissement des index

- `ai103.storage_inventory` : inventaire des PDF dans Blob Storage, sans téléchargement.
- `ai103.pdf_page_inventory` : comptage local des pages.
- `ai103.metadata_enrichment` : ajout de `company` et `fiscal_year` aux chunks existants par
  `merge`, sans réindexation, avec contrôle des schémas, lots, reprises sur HTTP 429 et
  vérification complète après écriture.
- `ai103.chunk_snapshot`, `ai103.page_localization`, `ai103.page_enrichment` : localisation des
  pages physiques de chaque chunk dans les PDF originaux, puis application contrôlée aux index.

## Organisation

| Dossier | Contenu |
|---|---|
| `src/ai103/retriever/` | Recherche lexicale, vectorielle, hybride et sémantique ; filtres ; benchmark ; qrels |
| `src/ai103/rag/` | Contexte, génération Azure OpenAI, garde-fous, pipeline, campagnes |
| `src/ai103/agent/` | Outils, définition du prompt agent, boucle d'exécution, moniteur |
| `src/ai103/evaluation/` | Évaluateurs Azure AI Evaluation, contrôle des chiffres, comparaisons |
| `src/ai103/observability.py` | Traces OpenTelemetry |
| `adr/` | Décisions d'architecture et leurs preuves |
| `scripts/` | Points d'entrée en ligne de commande |
| `tests/` | Tests unitaires, avec des clients Azure simulés et sans appel réseau |

Les PDF, les rapports de campagne, les secrets et la configuration locale ne sont pas versionnés.

## Vérifications

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Limites

- Une seule exécution par configuration et un seul juge, de la même famille que le générateur :
  la variabilité du modèle n'est pas mesurée.
- Quelques réponses de référence FinanceBench sont ambiguës ou discutables ; elles sont relues à
  part, sans modifier le jeu d'origine.
- Le garde-fou de déploiement détecte une injection isolée, mais pas une injection noyée dans un
  long contexte : les défenses applicatives restent nécessaires.
- L'agent fonctionne avec une application qui exécute ses outils : il ne répond pas seul dans le
  playground du portail.
