# ADR-0002 : Filtres entreprise et exercice extraits de la question

- **Statut** : Accepté
- **Date** : 2026-10-17

## Contexte

Les 84 rapports partagent le même vocabulaire (revenue, net income, cash flow…). Sans filtre, la
recherche lexicale confond les entreprises et les exercices. Chaque chunk porte les métadonnées
`company` et `fiscal_year`, ajoutées après l'ingestion à partir d'un référentiel documentaire
vérifié, et ces champs sont filtrables. Azure AI Search applique un filtre OData mais ne le déduit
pas de la question : l'application doit l'extraire elle-même.

## Options

- **S0** : aucun filtre.
- **S1** : filtre sur l'entreprise.
- **S2** : entreprise et années citées dans la question.
- **S3** : entreprise et années citées ± 1.
- Extraction par règles (catalogue des 32 entreprises et alias) ou par un LLM en sortie structurée.

## Décision

Extraction **par règles**, déterministe et gratuite, à partir du **texte de la question seul**
(jamais des réponses de référence), et stratégie **S3** :
`company eq '…' and (fiscal_year eq N-1 or … or fiscal_year eq N+1 or fiscal_year eq null)`.
Sans entreprise détectée, ou si plusieurs entreprises sont détectées, aucun filtre n'est appliqué.
Le filtrage vectoriel est en `preFilter`.

## Conséquences

- La tolérance d'un an reflète le contenu des rapports : un rapport de l'exercice N contient les
  comparatifs de N-1 et les perspectives de N+1, et certaines entreprises nomment leur exercice
  selon l'année de clôture.
- Les documents sans exercice unique (8-K) restent admissibles grâce à `fiscal_year eq null`.
- Limite : quatre alias ont été ajoutés après lecture des questions, donc la précision de
  l'extracteur est légèrement optimiste pour des questions inédites.

## Preuves

- Extracteur évalué hors ligne : entreprise correcte pour 147 questions sur 150, aucune erreur,
  3 abstentions (questions qui ne nomment pas l'entreprise).
- Documents de référence exclus par le filtre : S1 aucun, S2 7, S3 1. Un filtre trop strict
  détruit le rappel sans le signaler.
- Campagne de retrieval sur les 150 questions (bon document dans le top 5) :

| Mode, index natif | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| Lexical | 0,49 | 0,87 | 0,91 | 0,92 |
| Vectoriel | 0,85 | 0,87 | 0,92 | 0,92 |
| Hybride | 0,83 | 0,90 | 0,91 | 0,95 |

- S3 donne les meilleures preuves vérifiées en vectoriel (0,39 de questions avec toutes les preuves
  retrouvées contre 0,34 sans filtre). S2 classe mieux (MRR 0,85 contre 0,80), mais au prix de
  7 documents exclus.
