# Résultats

Cette page synthétise les sorties finales du notebook 05 : sélection des modèles, gain du vote doux, résultats out-of-distribution et fichiers produits.

## Objectif De La Sélection

Le projet ne choisit pas une architecture unique à l'avance. Plusieurs familles de modèles TensorFlow/Keras sont benchmarkées, puis le notebook 05 sélectionne trois modèles par tâche.

La politique finale est :

```text
top3_max2_family
```

Cela signifie :

- choisir les meilleurs modèles selon un score multicritère ;
- éviter qu'une même famille occupe les trois places ;
- autoriser deux modèles de la même famille si les résultats le justifient ;
- conserver une architecture homogène avec trois modèles et vote doux pour chaque tâche.

Le vote doux est utilisé en production :

```text
probabilité finale = moyenne(probabilités modèle 1, modèle 2, modèle 3)
classe finale = argmax(probabilité finale)
```

## Modèles Retenus

| Tâche | Classes | Modèles retenus | F1 test moyen | F1 test min |
| --- | --- | --- | --- | --- |
| species | 7 | EfficientNetB0, ConvNeXtTiny, EfficientNetB1 | 0.9996 | 0.9994 |
| tomato | 10 | ConvNeXtTiny, MobileNetV3Large, EfficientNetB0 | 0.9877 | 0.9849 |
| apple | 4 | EfficientNetB0, MobileNetV3Large, MobileNetV3Small | 0.9993 | 0.9987 |
| grape | 4 | EfficientNetB1, EfficientNetB0, ConvNeXtTiny | 0.9993 | 0.9985 |
| corn | 4 | MobileNetV3Large, EfficientNetB1, EfficientNetB0 | 0.9909 | 0.9902 |
| potato | 3 | MobileNetV3Large, EfficientNetB1, MobileNetV3Small | 0.9994 | 0.9991 |
| pepper | 2 | MobileNetV3Large, MobileNetV3Small, ConvNeXtTiny | 0.9991 | 0.9973 |
| strawberry | 2 | MobileNetV3Small, EfficientNetB0, MobileNetV3Large | 1.0000 | 1.0000 |

Lecture : les performances in-distribution sont très élevées, ce qui est cohérent avec PlantVillage, dont les images sont relativement propres et homogènes.

## Gain Du Vote Doux Sur Le Test In-Distribution

| Tâche | F1 meilleur modèle | F1 ensemble | Gain F1 | Log loss meilleur | Log loss ensemble | Gain log loss |
| --- | --- | --- | --- | --- | --- | --- |
| apple | 1.0000 | 0.9993 | -0.000662 | 0.002130 | 0.002939 | -0.000809 |
| corn | 0.9917 | 0.9917 | 0.000049 | 0.022140 | 0.020067 | 0.002073 |
| grape | 1.0000 | 1.0000 | 0.000000 | 0.001113 | 0.001888 | -0.000775 |
| pepper | 1.0000 | 1.0000 | 0.000000 | 0.002800 | 0.003818 | -0.001018 |
| potato | 1.0000 | 1.0000 | 0.000000 | 0.000751 | 0.002336 | -0.001585 |
| species | 0.9998 | 0.9998 | 0.000070 | 0.001299 | 0.001284 | 0.000014 |
| strawberry | 1.0000 | 1.0000 | 0.000000 | 0.000316 | 0.000364 | -0.000048 |
| tomato | 0.9898 | 0.9938 | 0.004042 | 0.025737 | 0.028634 | -0.002896 |

Interprétation :

- l'ensemble améliore nettement la tomate sur le F1 macro test ;
- il est quasiment neutre sur les tâches déjà proches de 1.0 ;
- il peut légèrement dégrader la log loss sur certaines tâches, mais l'écart de F1 reste très faible ;
- l'architecture à trois modèles est conservée pour garder un système homogène et explicable.

## Résultats Out-Of-Distribution

PlantDoc sert uniquement à l'évaluation OOD. Il n'est jamais utilisé pour l'entraînement.

| Tâche | F1 OOD meilleur | F1 OOD ensemble | Gain F1 OOD | Accuracy OOD meilleur | Accuracy OOD ensemble | Gain accuracy OOD |
| --- | --- | --- | --- | --- | --- | --- |
| apple | 0.3315 | 0.3146 | -0.016886 | 0.3763 | 0.3659 | -0.010453 |
| corn | 0.2508 | 0.2075 | -0.043265 | 0.3148 | 0.2751 | -0.039683 |
| grape | 0.3291 | 0.3554 | 0.026357 | 0.5974 | 0.6299 | 0.032468 |
| pepper | 0.7958 | 0.7842 | -0.011590 | 0.8000 | 0.8160 | 0.016000 |
| potato | 0.4116 | 0.3891 | -0.022430 | 0.6332 | 0.5858 | -0.047493 |
| species | 0.7055 | 0.7081 | 0.002607 | 0.7399 | 0.7562 | 0.016365 |
| strawberry | 0.4110 | 0.3766 | -0.034420 | 0.6979 | 0.6042 | -0.093750 |
| tomato | 0.1759 | 0.1625 | -0.013459 | 0.2658 | 0.2525 | -0.013289 |

Interprétation :

- les résultats OOD sont nettement plus faibles que les résultats PlantVillage ;
- l'écart confirme que les images terrain sont plus difficiles : fonds naturels, éclairages variables, feuilles moins centrées ;
- PlantDoc ne couvre pas toutes les classes de façon fiable pour chaque espèce ;
- les métriques OOD sont donc utilisées comme diagnostic de robustesse, pas comme critère principal de sélection.

## Fichiers Produits Par Le Notebook 05

| Fichier | Usage |
|---|---|
| `models/ensemble_config.json` | Configuration finale lue par l'API locale |
| `models/ensemble/ensemble_config_hf.json` | Configuration enrichie avec chemins Hugging Face Hub |
| `models/ensemble/selection_strategy_comparison.csv` | Comparaison des stratégies de sélection |
| `models/ensemble/selection_summary.csv` | Sélection candidate initiale |
| `models/ensemble/final_selection_summary.csv` | Sélection finale des 24 modèles |
| `models/ensemble/ensemble_evaluation.csv` | Évaluation single models + ensembles |
| `models/ensemble/ensemble_gain_summary.csv` | Gain du vote doux par tâche |
| `models/ensemble/final_decisions.csv` | Décision finale : vote doux conservé partout |

## Limites

- PlantVillage est un dataset propre, donc les performances in-distribution peuvent surestimer la robustesse en conditions réelles ;
- PlantDoc ne couvre pas uniformément toutes les classes ;
- les images terrain demanderaient idéalement plus de données, d'augmentation réaliste et une validation métier ;
- le premier appel API peut être lent sur Hugging Face Spaces car les modèles sont chargés en lazy loading.
