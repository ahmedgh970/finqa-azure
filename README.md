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
