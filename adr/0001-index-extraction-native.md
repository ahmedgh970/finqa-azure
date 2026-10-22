# ADR-0001 : Index d'extraction native comme référence, Layout en comparaison

- **Statut** : Accepté
- **Date** : 2026-10-13, confirmé par l'évaluation du 2026-10-19

## Contexte

Le corpus compte 84 rapports financiers en PDF (10-K, 10-Q, 8-K, communiqués de résultats),
soit 12 013 pages riches en tableaux. Azure AI Search peut extraire leur texte de deux façons :

- l'**extraction native** de l'indexer, rapide et sans coût d'analyse ;
- la compétence **Document Layout** (Azure AI Document Intelligence), qui restitue la structure
  (titres en Markdown, tableaux en HTML), facturée à la page.

Le budget disponible était limité, et chaque passe Layout complète coûtait environ 106 € pour les
12 013 pages.

## Options

1. Extraction native seule.
2. Layout seul.
3. Les deux, dans deux index au même schéma, pour les comparer sur des questions identiques.

## Décision

Construire **les deux index** (option 3) avec le même découpage (4 000 caractères, recouvrement
de 400) et les mêmes embeddings (`text-embedding-3-small`, 1 024 dimensions), puis retenir
l'**index natif comme référence**, l'index Layout restant une configuration de comparaison.

## Conséquences

- Positives : coût d'ingestion minimal ; index plus compact (11 866 chunks contre 19 386) ;
  décision fondée sur une mesure plutôt que sur une intuition.
- Négatives : les tableaux natifs sont aplatis ; les titres de section ne sont pas disponibles
  comme métadonnées.
- L'index Layout a atteint la limite de 300 Mo vectoriels du niveau Serverless (environ 102 %) :
  ses enrichissements ultérieurs ont été bloqués.

## Preuves

- Une inspection du même tableau dans les deux index a montré une structure fidèle côté Layout,
  mais un montant altéré par l'extraction (`$ 5,854.4` devenu `$ 3 5,854.4`) : une meilleure
  structure ne garantit pas des chiffres justes.
- Évaluation RAG du 19 octobre (vectoriel, k = 10, mêmes réglages) : Layout ne gagne pas
  globalement face à l'index natif ; différence de completeness +0,05 avec un IC 95 %
  [−0,24 ; +0,33], et 7 questions chiffrées gagnées pour 7 perdues.
- Layout est meilleur sur les questions qualitatives (completeness 3,47 contre 3,06 à 3,42) et moins
  bon sur les calculs (2,92 contre 3,02 à 3,34), ce qui correspond aux montants abîmés dans les
  tableaux HTML. Il a aussi produit plus d'extraits non vérifiables et un refus du modèle.
