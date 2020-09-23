<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                            BANNER & SHIELD                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


![](./img/sfr.jpg)

<p align="center">
    <!-- Last Master Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/JFR2020?label=last%20master%20commit&style=flat-square"
         alt="GitHub last commit">
        <!-- Last Commit-->
    <img src="https://img.shields.io/github/last-commit/the-dharma-bum/JFR2020/master?style=flat-square"
         alt="GitHub last commit">
    <!-- Commit Status -->
    <img src="https://img.shields.io/github/commit-status/the-dharma-bum/JFR2020/master/eca2cb36cf952f86dcd3fc9112890c92913f8b60?style=flat-square"
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


Dataloader fait main gérant des couples (json, nifti).

Modèle et entrainement basé sur [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                DEVNOTE                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# NOTE POUR LES CONTRIBUTEURS

Lorsque vous participez à ce projet, penser à: 

1. Développer en dehors de la branche master
2. Faire des commits petits et réguliers
3. Mettre à jour les badges, notamment la branche du dernier commit
4. Dès qu'on développe une feature ou un bugfix: mettre à jour le tableau correspondant
5. Dès qu'une feature est prête, merge sur master
6. Dès qu'on merge sur master, mettre à jour le tableau correspondant


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                          TABLE OF CONTENTS                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SOMMAIRE

- [A Faire](#a-faire)
     - [New Features](#new-features)
     - [Bugfixes](#bugfixes)
- [Last Commit Changes Log](#last-commit-changes-log)
- [Installation](#installation)
- [Usage](#usage)
- [Intégration à fastai](#integration-a-fastai)


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                  TO DO                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# A faire
[(Back to top)](#sommaire)

## New features:

| Features                                                 |      Status      |     Owner    |
|----------------------------------------------------------|:----------------:|:------------:|
| Dataloader                                               |  TO DO           |              |
| Dataloader: load bons couples (jsons,nifti)              |  DONE            |              |
| Dataloader: rescale correct                              |  DONE            |              |
| Dataloader: normalisation basée sur l'histogramme        |  TO DO           |              |
| Dataloader: suppression des coupes inutiles              |  DONE            |              |
| Dataloader: crop centré sur le coeur                     |  DONE            |              |
| Augmentation                                             |  TO DO           |              |


## Bugfixes:

| Bugfixes                                                 |      Status      |     Owner    |
|----------------------------------------------------------|:----------------:|:------------:|
| Rien pour l'instant                                      |                  |              |

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              CHANGES LOG                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# Last Commit Changes Log

- Finalisation du preprocessing: wrapping en une fonction agissant sur tout le dataset.


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              INSTALLATION                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Installation
[(Back to top)](#sommaire)

Clonez le repo:

```git clone https://github.com/the-dharma-bum/jfr2020```

Installer toutes les dépendances (attention ça peut être plutôt long):

``` pip install requirements.txt ```


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                 USAGE                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Usage
[(Back to top)](#sommaire)

![](./img/arch.png)

Tous les hyperparamètres sont configurables dans le fichier config.py.
Autrement, les hyperparamètres sont configurables en instanciant les dataclass de configuration (se référer à config.py pour les instancier correctement).
Une fois les dataclass de configuration instanciées, une dataclass config peut être instanciée avec comme attribut les dataclass de configuration.
Ensuite, un objet model (voir model.py) peut être instancié avec cette dataclass config.
Enfin, ce model est donné à un objet trainer. 

En résumé, l'initialisation se fait en 3 étapes: 
- 1. Instanciation d'une configuration
- 2. Instanciation d'un model à l'aide de l'objet de configuration
- 3. Instanciation d'un trainer avec un model.


Cette architecture a plusieurs avantages, notamment:

1. Tous les hyperparamètres sont accessibles depuis un même fichier. 
2. Tous les hyperparamètres sont découplés du code qui les utilise.
3. La partie modèle est découplée de la partie data (!).
4. Un datamodule est partageable et réutilisable.

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




# Intégration à fastai
[(Back to top)](#sommaire)

Il est est très facile d'utiliser la procédure d'entraînement de fastai à partir de ce code. 

En ayant instancié 

- un modèle (définit dans model.py) comme ceci:
```python
from model import LightningModel

model = LightningModel(config)
```
(où config est une dataclass Model définit dans config.py)

- un datamodule (définit dans datamodule.py) comme ceci:
```python
from datamodule import DicomDataModule

dm = DicomDataModule(config)
```
(où config est une dataclass Dataloader définit dans config.py)

Il suffit de créer un object Dataloaders:
```python
from fastai.vision.all import DataLoaders

data = Dataloaders(dm.train_dataloader(), dm.val_dataloader()).cuda()
```

On peut alors définir un Learner et le fit, par exemple:
```python
learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam, metrics=accuracy)
learn.fit_one_cycle(1, 0.001)
```

Ceci permet alors d'utiliser toutes les fonctionnalités de fastai (callbacks, transforms, visualizations ...).