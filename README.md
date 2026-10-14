# Finance RAG

Outils Python pour préparer et vérifier un environnement de recherche sur des PDF financiers,
avec Azure Blob Storage, Microsoft Foundry et Ollama.

Le dépôt contient les diagnostics de connexion, un inventaire des propriétés Blob et un compteur
de pages PDF locales. Les configurations de ressources Azure et les données du corpus sont externes
au dépôt.

## Installation

Python 3.12+ et `uv` sont nécessaires.

```bash
uv sync --locked
cp .env.example .env
```

Renseigner les variables de `.env` avec les endpoints, les identifiants Azure et les noms des modèles
Ollama disponibles sur le poste. Le fichier `.env` reste local. Les variables exportées dans le shell
ont priorité sur ce fichier.

Pour l'authentification Azure :

```bash
az login
```

Les outils utilisent `DefaultAzureCredential`. `AZURE_TOKEN_CREDENTIALS=AzureCliCredential`
sélectionne la session Azure CLI ; `dev` permet la chaîne des outils de développement.

## Diagnostics

```bash
uv run ai103 doctor --quick
uv run ai103 doctor
uv run ai103 doctor --json
```

Les contrôles vérifient la configuration, le serveur Ollama et l'accès Azure. Le diagnostic complet
effectue aussi de petites requêtes de génération et d'embeddings Ollama. Foundry est facultatif :
son contrôle lit les connexions du projet et ne lance pas d'inférence Azure.

Les logs vont sur stderr et le rapport sur stdout. Codes de sortie : `0` pour aucun échec,
`1` pour un contrôle en échec et `2` pour une configuration invalide.

## Inventaire Blob

Configurer `AZURE_STORAGE_ACCOUNT_URL` et `AZURE_STORAGE_CONTAINER_NAME`, puis indiquer le nombre
de PDF attendu :

```bash
uv run python -m ai103.storage_inventory --expected-count 84
```

L'inventaire lit les noms et tailles des blobs sans télécharger les PDF. L'identité utilisée doit
disposer des droits de lecture sur le conteneur.

## Pages PDF locales

```bash
uv run python -m ai103.pdf_page_inventory /chemin/vers/corpus --expected-count 84 --output /chemin/vers/rapport.json
```

Le compteur parcourt les PDF locaux en lecture seule et enregistre les pages par fichier,
les erreurs et le total dans un rapport JSON. Un corpus incomplet n'est pas déclaré complet.

## Métadonnées documentaires

Configurer `AZURE_SEARCH_SERVICE_ENDPOINT` dans `.env` et préparer un référentiel JSONL
avec les champs `doc_name`, `company` et `fiscal_year` (entier ou `null`). Le nom documentaire
correspond au nom du PDF sans son extension, en conservant sa casse.

Préparer les mises à jour sans écrire dans Azure :

```bash
uv run python -m ai103.metadata_enrichment --dry-run
```

Le client vérifie le schéma et la couverture complète des documents dans les deux index,
puis enregistre un bilan dans `docs/reports/metadata-enrichment.json`.
Le référentiel par défaut est `docs/corpus-metadata.jsonl`, avec 84 documents attendus.
`--metadata`, `--expected-count`, `--endpoint`, `--report` et `--index` permettent de configurer
le traitement ; `--index` peut être répété pour sélectionner plusieurs index.

Appliquer les métadonnées après examen du bilan :

```bash
uv run python -m ai103.metadata_enrichment --apply
```

Les mises à jour utilisent uniquement `merge` avec `chunk_id`, `company` et `fiscal_year`.
Le contenu, les pages, les sources et les vecteurs ne sont pas envoyés. Aucun indexer,
appel Layout ou calcul d'embeddings n'est déclenché. Les champs vectoriels doivent être stockés
avec `stored=true`. La vérification relit toutes les clés connues par lots filtrés avec
`search.in`, sans pagination par position : les merge peuvent changer l'ordre des résultats.
Elle réessaie brièvement les lectures si les mises à jour ne sont pas encore visibles.
Les envois sont séquentiels, avec une seconde entre lots. Un HTTP 429 sans indication
explicite de quota déclenche au plus quatre reprises du même lot avec attente croissante,
en respectant `Retry-After`. Un quota signalé ou une attente demandée supérieure à 60 secondes
arrête le traitement. Les erreurs individuelles d'un lot entraînent également un arrêt.
`--batch-size 100` réduit la taille des lots (500 par défaut).
Les lectures et écritures consomment les ressources du service Search existant.

Le traitement n'est pas une transaction globale : un échec peut laisser des mises à jour partielles.
Le rapport indique les opérations acquittées ; une nouvelle simulation identifie les chunks encore
à modifier. Les chunks déjà conformes sont ignorés.
La lecture initiale exige un index sans ingestion concurrente et s'arrête si elle détecte
des clés répétées ou un changement de comptage. Une reprise utilise la même commande `--apply` ;
aucun Reset ni lancement d'indexer n'est nécessaire. Utiliser `--report` avec un autre chemin
pour conserver le rapport précédent pendant un diagnostic en simulation.

## Localisation des pages

Un export en lecture seule permet de préparer les pages physiques des chunks existants :

```bash
uv run python -m ai103.chunk_snapshot --endpoint https://your-search-service.search.windows.net
uv run python -m ai103.page_localization
uv run python -m ai103.page_localization_validation
```

La localisation utilise les PDF originaux locaux et `pdftotext` (Poppler requis).
Les sauts de page sont conservés ; la numérotation commence à 1, couverture comprise.
Les propositions `chunk_id`, `page_start`, `page_end` et leurs audits sont enregistrés sous
`docs/reports/page-localization/`. Les passages ambigus restent à `null`.
Les scores d'alignement ne garantissent pas l'exactitude de chaque proposition.
Ces commandes ne modifient aucun index Azure et ne déclenchent pas d'analyse documentaire.

Après examen des propositions, le client de pages vérifie leur provenance et prépare les merge :

```bash
uv run python -m ai103.page_enrichment --dry-run --endpoint https://your-search-service.search.windows.net
```

`--apply` autorise explicitement leur application, par lots de 100. Seuls `chunk_id`,
`page_start` et `page_end` sont envoyés. Les propositions non résolues sont ignorées ;
une page existante contradictoire ou un contenu modifié provoque un arrêt. Le client relit
les deux index avant toute écriture, recontrôle chaque lot et vérifie les pages ainsi que
le contenu, la source et les métadonnées préservées après application. Le rapport se trouve
dans `docs/reports/page-localization/enrichment.json`.

## Vérifications

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Les tests simulent les appels Azure et Ollama et utilisent des PDF générés localement.

## Organisation

- `src/ai103/` : configuration, diagnostics et outils d'inventaire.
- `tests/` : tests automatisés.
- `.env.example` : modèle de configuration générique.
- `pyproject.toml` et `uv.lock` : métadonnées et dépendances.

Les PDF, les secrets, les rapports et la documentation locale sont exclus de Git.
