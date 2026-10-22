# ADR-0003 : Recherche hybride puis semantic ranker sur un lot fixe de 20 candidats

- **Statut** : Accepté
- **Date** : 2026-10-21

## Contexte

Azure AI Search propose la recherche lexicale (BM25), vectorielle (HNSW), hybride (fusion RRF des
deux) et un reclassement de niveau 2, le **semantic ranker**, qui réordonne au plus les
50 premiers résultats avec un modèle de compréhension du langage. Le premier RAG utilisait le
vectoriel seul ; il fallait choisir le mode, le nombre de passages (k) et mesurer l'apport du
reclassement indépendamment du reste.

## Options

- C1 : vectoriel, k = 5 ; C2 : vectoriel, k = 10 ; C3 : hybride, k = 10.
- C5 : hybride puis semantic ranker, k = 10.

Pour isoler le reclassement, C5 utilise **le même lot de 20 candidats hybrides** que C3 : une
première requête fixe le lot, puis la requête sémantique est filtrée par
`search.in(chunk_id, …)`. C3 garde les 10 premiers dans l'ordre RRF (identique à un top 10, car
l'ordre RRF ne dépend pas du nombre de résultats demandés), C5 les 10 meilleurs après reclassement.

## Décision

Retenir **C5** : hybride (BM25 et vecteur, fusion RRF), filtre de l'ADR-0002, reclassement
sémantique d'un lot fixe de 20 candidats, 10 passages transmis au modèle.

## Conséquences

- Le semantic ranker exige une configuration sémantique sur l'index (champ de contenu) ; elle
  s'ajoute sans réindexation.
- Deux requêtes Search par question au lieu d'une ; latence médiane 8,3 s contre 8,0 s.
- Le plan sémantique gratuit couvre l'usage du projet ; le plan payant n'est pas activé.

## Preuves

| | C1 | C2 | C3 | C5 |
|---|---|---|---|---|
| Taux de réponse | 53 % | 58 % | 65 % | **69 %** |
| Completeness (1-5) | 3,12 | 3,29 | 3,37 | **3,69** |
| Chiffres exacts (sur 52) | 24 | 27 | 31 | **37** |

- C3 face à C1 : completeness +0,26, IC 95 % [+0,03 ; +0,49]. Le lexical, une fois l'entreprise
  filtrée, retrouve les termes exacts (« net PP&E », « Section 12(b) »).
- C5 face à C3, sur le même lot : completeness **+0,32, IC 95 % [+0,15 ; +0,51]**, 26 questions
  améliorées et 8 dégradées ; chiffres +8 / −2 (McNemar p = 0,11).
- Mécanisme : en moyenne 3,7 des 10 passages retenus viennent des rangs 11 à 20 du lot hybride,
  et 17 des 26 questions améliorées citent un passage ainsi remonté. Le reclassement améliore la
  précision du haut du classement sans changer le rappel du lot.
