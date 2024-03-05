# Plan Atelier

## Présentation de la problématique de la segmentation 
 * Basé sur des règles : limité pour tout expliciter
 * Basé sur la CV (watershed) : necessité d'adapter le seuil/pre processing
 * Solution : faire "apprendre" un modèle en donnant des exemples
 * Problème : pas de base d'app suffisamment grande

## Utilisation de modèles fondations
 * Evaluation de différents modeles pré entrainés pour voir les possibilités sur leurs images (entre 2 et 5 trucs à tester et comparer)
    * Présentation de YOLO, SAM, etc
    * Présentation du code pour faire tourner ces modèles
        (insister sur la récupération des poids du modèle qui est au coeur du pb)


## Apprendre un modèle
 * Petit layus sur l'apprentissage et le fine tuning
### Apprendre un modèle simple : prédire la classe d'une image
 * MNIST
 * Un CNN basique (et léger)
 * Notions d'epoch et de protocole d'évaluation (train/valid/test)

### Fine tuning d'un modele de segmentation (à tester)
 * Basé sur le dataset emps
 * Quel modele ? quelle ressource ? SAM sur colab ? Hugging faces ?
 * Ressources : 
    * https://huggingface.co/blog/fine-tune-segformer
    * https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/
    * https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb

## Conclusion pour aller plus loin
 * des données annotées et de la puissance de calcul