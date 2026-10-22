# Architecture Decision Records

Ce dossier consigne les décisions structurantes du projet : le contexte, les options étudiées,
le choix retenu, ses conséquences et **les mesures qui l'ont justifié**. Chaque décision a été
prise après une expérience reproductible sur les 150 questions du jeu FinanceBench open source.

| ADR | Décision | Statut |
|---|---|---|
| [0001](0001-index-extraction-native.md) | Index d'extraction native comme référence, Layout en comparaison | Accepté |
| [0002](0002-filtres-metadonnees-question.md) | Filtres entreprise et exercice extraits de la question, avec une tolérance d'un an | Accepté |
| [0003](0003-hybride-semantic-ranker.md) | Recherche hybride puis semantic ranker sur un lot fixe de 20 candidats | Accepté |
| [0004](0004-generation-ancree-citations-verifiees.md) | Génération ancrée : extraits cités vérifiés par le code, défense en profondeur | Accepté |
| [0005](0005-protocole-evaluation.md) | Évaluation : juge distinct du générateur, chiffres déterministes, statistiques appariées | Accepté |
| [0006](0006-agent-foundry-outils-fonction.md) | Agent Foundry (prompt agent) avec outils fonction exécutés par l'application | Accepté |
| [0007](0007-observabilite-opentelemetry.md) | Observabilité OpenTelemetry, conventions GenAI, exporteur interchangeable | Accepté |

## Format

Chaque ADR suit le même plan : **Contexte**, **Options**, **Décision**, **Conséquences**,
**Preuves**. Une décision n'est pas modifiée après coup : si elle change, un nouvel ADR la remplace
et l'ancien passe au statut « Remplacé par ».

## Conventions des mesures

- **Taux de réponse** : part des 150 questions auxquelles le système répond, au lieu de s'abstenir
  ou de s'arrêter sur une limite.
- **Completeness** : note de 1 à 5 donnée par l'évaluateur Response Completeness d'Azure AI
  Evaluation, face à la réponse de référence.
- **Chiffres exacts** : contrôle déterministe sur les 52 questions dont la réponse de référence est
  un nombre (tolérance d'arrondi, changements d'échelle).
- **Comparaisons appariées** : mêmes questions, intervalle de confiance bootstrap à 95 % sur la
  différence de completeness, test exact de McNemar sur les chiffres.
