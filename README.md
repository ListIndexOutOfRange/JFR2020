<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                            BANNER & SHIELD                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


![](./sfr.jpg)

<p align="center">
    <!-- Last Master Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/MicroNet?label=last%20master%20commit&style=flat-square"
         alt="GitHub last commit">
        <!-- Last Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/MicroNet/master?style=flat-square"
         alt="GitHub last commit">
    <!-- Commit Status -->
    <img src="https://img.shields.io/github/commit-status/the-dharma-bum/MicroNet/improve_logging/0c8c2d6e5363b479344983c564c6dcc27834390a?style=flat-square"
         alt="GitHub commit status">
</p>

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                               MAIN TITLE                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# DATA CHALLENGE JFR 2020

Implémentation Pytorch du Data Challenge 2020 pour les Journées Françaises de la Radiologie:

Détermination automatique du score calcique sur Scanner 3D


Dataloader basé sur [fastaiv2](https://github.com/fastai/fastai)

Modèle et entrainement basé sur [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                          TABLE OF CONTENTS                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SOMMAIRE

- [A Faire](#to-do-for-next-release)
     - [New Features](#new-features)
     - [Bugfixes](#bugfixes)
- [Last Commit Changes Log](#last-commit-changes-log)
- [Installation](#installation)
- [Usage](#usage)


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                  TO DO                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# A faire
[(Back to top)](#table-of-contents)

## New features:

| Features                                                 |      Status      |      Type    |
|----------------------------------------------------------|:----------------:|:------------:|
| Dataloader                                               |  TO DO           |   Feature    |
| Dataloader: load dcm par dossier (ie par patient)        |  TO DO           |   Feature    |
| Dataloader: rescale correct                              |  TO DO           |   Feature    |
| Dataloader: normalisation basée sur l'histogramme        |  TO DO           |   Feature    |
| Dataloader: suppression des coupes inutiles              |  TO DO           |   Feature    |
| Dataloader: crop centré sur le coeur                     |  TO DO           |   Feature    |


## Bugfixes:

- Rien pour l'instant.

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              CHANGES LOG                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# Last Commit Changes Log

- initialisation du github


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              INSTALLATION                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Installation
[(Back to top)](#table-of-contents)

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/the-dharma-bum/jfr2020```

Ce projet nécessite fastai et pytorch lightning. 
Pour s'assurer que tout se déroule bien, essayer: 

```apt install gcc git pip```

```pip install fastai```

```pip install pytorch-lightning```


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                 USAGE                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Usage
[(Back to top)](#table-of-contents)

Tous les hyperparamètres sont configurables dans le fichier config.py.
Autrement, les hyperparamètres sont configurables en instanciant les dataclass de configuration (se référer à config.py pour les instancier correctement).
Une fois les dataclass de configuration instanciées, une dataclass config peut être instanciée avec comme attribut les dataclass de configuration.
Ensuite, un objet model (voir model.py) peut être instancié avec cette dataclass config.
Enfin, ce model est donné à un objet trainer. 

En résumé, l'initilisation se fait en 3 étapes: 
- 1. Instanciation d'une configuration
- 2. Instanciation d'un model à l'aide de l'objet de configuration
- 3. Instanciation d'un trainer avec un model. 

Une fois tout paramétré, il suffit de lancer:

```python main.py ```

Cette commande est compatible avec un grand nombre d'arguments. Tapez

```python main.py -h ```

pour les voir tous, ou référez vous à la [documentation de Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Quelques paramètres utiles:

- ```--gpus n``` : lancer l'entraînement sur n gpus
- ```--distributed_backend ddp``` : utiliser DistributedDataParallel comme backend pour entraîner sur plusieurs gpus.
- ```--fast_dev_run True``` : lance une boucle d'entraînement complète sur un batch (train, eval, test). A utiliser pour débuguer.

Si vous voulez débuguez mais que fast_dev_run ne vous convient pas (par exemple si vous voulez voir ce qu'il se passe entre deux époques), vous pouvez utiliser:

- ```--limit_train_batches i --limit_val_batches j --max_epochs k```

i,j,k étant bien sûr trois entiers de votre choix.





