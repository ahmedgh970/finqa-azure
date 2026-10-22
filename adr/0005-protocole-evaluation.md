# ADR-0005 : Juge distinct du générateur, chiffres déterministes, statistiques appariées

- **Statut** : Accepté
- **Date** : 2026-10-19

## Contexte

Il fallait comparer plusieurs configurations RAG, puis un agent, sur les 150 questions FinanceBench,
dont 52 attendent un nombre et 98 une analyse rédigée. Les mesures devaient être reproductibles,
peu coûteuses, et distinguer « répondre souvent » de « répondre juste ». Le jeu de référence
contient par ailleurs quelques réponses discutables.

## Options

- Métriques de recouvrement de texte (F1, BLEU…) : inadaptées aux réponses analytiques.
- Un juge LLM seul : couverture large, mais coût, variabilité et biais.
- Le même modèle pour générer et juger : biais d'auto-préférence.
- Ragas ou le SDK Azure AI Evaluation pour les juges.

## Décision

- **SDK Azure AI Evaluation** (`evaluate()`), exécuté localement, avec les évaluateurs
  **Groundedness**, **Relevance** et **Response Completeness**. Ce SDK ne dépend pas de Ragas.
- **Juge `gpt-5-mini`, distinct du générateur `gpt-5.2`**, configuré comme modèle de raisonnement.
- Un **évaluateur personnalisé déterministe** branché dans `evaluate()` : le nombre de référence
  doit figurer dans le premier paragraphe de la réponse, avec une tolérance d'arrondi et les
  changements d'échelle usuels ; les renvois de citation et les années sont exclus.
- **Comparaisons appariées** : différence de completeness avec un intervalle bootstrap à 95 %
  (10 000 tirages, graine fixe), test exact de McNemar sur les chiffres, transitions entre réponse
  et abstention.
- Résultats séparés entre **quand le système répond** et **taux de réponse**.
- Seul le texte des questions est envoyé au système évalué ; les réponses de référence ne sont lues
  qu'après la génération, et aucune donnée d'évaluation n'apparaît dans les prompts.

## Conséquences

- Groundedness et Relevance ne départagent pas les configurations (environ 4,9 et 4,3 partout) :
  une abstention honnête est « ancrée » et « pertinente ». Les décisions reposent sur la
  completeness et les chiffres exacts.
- Un seul juge, de la même famille que le générateur, et une seule exécution par configuration :
  la variabilité du modèle n'est pas mesurée.
- Les erreurs restantes sont relues à la main ; plusieurs viennent de réponses de référence
  ambiguës ou erronées, consignées à part sans modifier le jeu d'origine.

## Preuves

- Test du juge sur trois réponses : une réponse juste notée 5 en completeness, deux abstentions
  notées 5 en Groundedness mais 1 en completeness.
- Coût de l'évaluation de quatre configurations : 13,05 USD de génération et 6,66 USD de juge
  (borne haute), pour 600 réponses et 1 800 jugements.
- Ce protocole a permis de détecter un gain significatif (ADR-0003, ADR-0006) et une absence de
  gain (ADR-0001) là où les moyennes seules auraient été ambiguës.
