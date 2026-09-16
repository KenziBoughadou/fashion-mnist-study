# Vérifications et reproductibilité numérique

Cette note détaille les contrôles techniques de l’étude et un incident observé
avant les expériences complètes. Le [README](../README.md) présente la question
expérimentale, les résultats et leur interprétation.

## Conservation des tentatives

Une exception d'entraînement laisse le journal déjà écrit, son statut et la cause
observée. Les autres entraînements sont tentés, mais aucune évaluation finale
n'a lieu si l'un d'eux échoue. Une interruption interceptable est enregistrée.
Après un arrêt brutal non interceptable, un statut resté actif ne doit pas être
interprété comme une réussite. Le rapport liste aussi les exécutions échouées
ou non démarrées et indique les comparaisons incomplètes.

## Contrôles réalisés

Les tests vérifient les partitions disjointes et stratifiées, le mélange commun,
les dimensions, la mise à jour des poids, le rechargement d'un checkpoint,
la pondération des pertes et l'absence d'accès au test avant la fin des entraînements.
Ils couvrent aussi la conservation des échecs, le refus d'écrasement, la sélection
par validation et l'agrégation de toutes les graines.
Les 12 tests ont été exécutés avec succès. Deux essais courts indépendants avec
le réglage final ont produit des poids et des métriques identiques pour les deux
modèles ; les durées ne font pas partie de cette comparaison d'identité.
Le rapport final a été régénéré à l'identique sans modifier les mesures ni les
checkpoints. Les métriques ont aussi été recalculées depuis les six fichiers de
prédictions et les empreintes du code vérifiées contre celles enregistrées au lancement.

## Incident de reproductibilité numérique

Une première vérification a échoué sur l'identité bit à bit de deux entraînements
synthétiques du MLP à graine identique. Un contrôle a mesuré un écart absolu
maximal de `3,1851 × 10⁻⁷` sur la première matrice de poids ; une tolérance
absolue de `10⁻⁷` ne suffisait pas. Les contrôles isolés ne reproduisaient pas
systématiquement l'écart. Le mode MKL `COMPATIBLE` a été introduit avant les
expériences complètes après des contrôles réussis avec ce réglage. Le mécanisme
précis de l'écart initial n'est pas établi. Le test conserve une comparaison
stricte des poids ; sa réussite locale ne garantit pas l'identité numérique
sur une autre machine ou une trajectoire complète de quinze époques.

## Réglage CPU

`experiments.py` fixe `MKL_CBWR=COMPATIBLE` avant les premières opérations
numériques, puis active les opérations déterministes de PyTorch. Le nombre
de threads est fixé à deux. Ces réglages sont enregistrés avec chaque étude.

Références : [reproductibilité PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html)
et [reproductibilité numérique conditionnelle de MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/getting-started-with-conditional-numerical.html).
