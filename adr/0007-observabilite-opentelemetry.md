# ADR-0007 : Observabilité OpenTelemetry, conventions GenAI, exporteur interchangeable

- **Statut** : Accepté
- **Date** : 2026-10-19

## Contexte

Les premières questions RAG prenaient de 2 à 13 secondes sans explication, et le coût par question
devait être suivi pour respecter un budget limité. L'agent ajoute des tours et des appels d'outils
qu'il faut pouvoir inspecter.

## Options

- Des chronomètres maison dans le code.
- Application Insights directement (nouvelle ressource, coût d'ingestion).
- OpenTelemetry avec un exporteur local, prêt à basculer vers Azure Monitor ou un autre collecteur.

## Décision

Instrumenter avec **OpenTelemetry** : une trace par question, avec un span par étape
(`filter`, `embedding`, `search`, `generation`) pour le RAG, et par tour et par outil
(`agent.round`, `tool.*`) pour l'agent. Les attributs suivent les **conventions sémantiques GenAI**
(`gen_ai.usage.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.response.model`…),
complétées du coût estimé. L'identité Entra est enveloppée pour chronométrer chaque obtention de
jeton. L'exporteur est local (JSON) ; seul lui change pour viser Application Insights et le tracing
de Foundry, ou un outil comme Arize Phoenix.

## Conséquences

- Les campagnes produisent un journal par question, des traces et une synthèse : statuts, coût,
  latence médiane et p95 par étape.
- Pas de tableau de bord tant qu'un collecteur n'est pas branché ; pour Python, l'ancien SDK
  OpenCensus d'Application Insights est retiré et la voie supportée est le Distro OpenTelemetry de
  Microsoft.

## Preuves

- Les traces ont montré que le client d'embeddings redemandait un jeton Entra à chaque question :
  10 obtentions et 7,9 s perdues sur 10 questions, alors que les SDK Search et OpenAI gardent le
  leur en cache. Après correction : 3 jetons pour toute une campagne de 150 questions, et
  l'embedding est passé de 1,5 s à 0,7 s en médiane.
- La latence de génération suit les tokens de sortie : la réponse la plus longue (1 869 tokens)
  a pris 22 s.
