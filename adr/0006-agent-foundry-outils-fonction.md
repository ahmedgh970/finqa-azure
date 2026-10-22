# ADR-0006 : Agent Foundry avec outils fonction exécutés par l'application

- **Statut** : Accepté
- **Date** : 2026-10-22

## Contexte

Le meilleur RAG (C5, ADR-0003) s'abstenait encore sur 31 % des questions, surtout quand il
fallait plusieurs informations (un ratio de liquidité, une moyenne sur trois ans) ou une
recherche mieux formulée que la question brute. Un agent peut chercher plusieurs fois,
reformuler et calculer avant de répondre.

## Options

- **Hosted agent** : notre propre code d'orchestration dans un conteneur exécuté par Foundry.
- **Prompt agent** (définition déclarative exécutée par Foundry Agent Service) avec :
  - les **outils intégrés** de Foundry (Azure AI Search, Code Interpreter, File Search, Bing) ;
  - **Foundry IQ** (agentic retrieval d'Azure AI Search, via MCP) ;
  - des **outils fonction** exécutés par notre application.

## Décision

Un **prompt agent** `gpt-5.2` créé par SDK dans Foundry Agent Service, avec trois **outils
fonction** en mode strict, exécutés par notre application :

- `list_filings` : catalogue des rapports d'une entreprise ;
- `search_filings` : le retrieval de C5, avec des filtres entreprise et exercice choisis par l'agent ;
- `calculate` : arithmétique évaluée par arbre syntaxique, sans `eval`.

La boucle (conversation, `function_call`, `function_call_output`) est pilotée par le code, avec
des **budgets séparés** (6 recherches, 12 appels d'outils, 10 tours, 0,10 USD par question). La
réponse finale est en JSON strict et passe par les mêmes contrôles qu'à l'ADR-0004.

## Conséquences

- Positives : comparaison propre avec C5 (même retrieval, seule l'orchestration change) ; contrôle
  total des outils, des budgets et des données accessibles ; l'agent ne voit jamais de secret.
- Négatives : l'agent ne fonctionne pas seul dans le playground du portail, qui ne peut pas
  exécuter les outils fonction ; plus de tours, donc plus de latence et de coût.
- Écartés : l'outil Azure AI Search intégré exigerait un vectorizer sur l'index pour la recherche
  vectorielle (l'outil envoie le texte de la requête) ; File Search imposerait une réingestion ;
  la recherche web trouverait les réponses publiées et fausserait l'évaluation ; un hosted agent
  ajouterait conteneur et calcul provisionné sans bénéfice ici. Foundry IQ reste une piste.
- Deux pièges rencontrés : un budget unique d'appels épuisé par des calculs parallèles, et des
  exemples de prompt qui reprenaient des chiffres du jeu d'évaluation (fuite de données, retirée
  et contrôlée automatiquement).

## Preuves

| | C5 (RAG) | Agent |
|---|---|---|
| Taux de réponse | 69 % | **91 %** |
| Completeness (1-5) | 3,69 | **4,25** |
| Chiffres exacts (sur 52) | 37 | **46** |
| Chiffres exacts quand il répond | 37/41 | **46/48** |
| Coût par question | 0,026 USD | 0,039 USD |
| Latence médiane | 8,3 s | 15,8 s |

- Comparaison appariée : completeness **+0,56, IC 95 % [+0,31 ; +0,83]**, 40 questions améliorées
  et 10 dégradées ; chiffres **+10 / −1, McNemar p = 0,012**.
- Comportement : 1,6 recherche par question en moyenne (jusqu'à 6), outil de calcul utilisé dans
  87 questions, 4,2 tours en moyenne, aucun appel refusé, 5 arrêts sur limite.
- Le cache de prompt couvre 1,6 million des 4,1 millions de tokens d'entrée, ce qui limite le coût
  des tours successifs.
