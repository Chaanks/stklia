
# Introduction

Ce dossier permet la paramétrisation de n'importe quel dataset avec Kaldi. Pour se faire, 4 étapes sont nécéssaires :

- Étape 1 : Installer kaldi
- Étape 2 : Copier ce dossier dans kaldi/egs/any_dataset/fbank/
- Étape 3 : Génerer les fichiers wav.scp, utt2spk, spk2utt
- Étape 4 : Lancer la paramétrisation

# Étape 1

Dans cette premiere étape, nous installons Kaldi.
cf `kaldi/INSTALL` pour plus d'informations.

## Installation des ressources

Kaldi nécéssite certaines ressources pouvant être installé avec les commandes :
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git bc g++ zlib1g-dev make automake autoconf bzip2 libtool subversion libatlas3-base
```

## Installation de Kaldi:

Pour installer Kaldi, il nous faut cloner le dépot officiel dans l'emplacement de notre choix, puis *build* le toolkit :
```
git clone https://github.com/kaldi-asr/kaldi.git kaldi –origin upstream
cd kaldi/tools
extras/check_dependencies.sh
make
```

# Étape 2 (Optionelle)

Kaldi est livré avec de nombreux exemples situés dans `kaldi/egs/`. Pour respecter la structure des dossiers de kaldi, nous pouvons copier le dossier de any_dataset dans `kaldi/egs/`. 
C'est le moment de donner le nom de votre dataset à ce dossier :
```
cp -r any_dataset/ path/to/kaldi/egs/<my_dataset_name>
```
Ainsi, la structure `kaldi/egs/<my_dataset_name>/fbank/feature-extraction.sh` devrait être respectée.

# Étape 3

Pour la acceder aux données, Kaldi a besoin de 3 fichiers : `wav.scp`, `utt2spk` et `spk2utt`. Chaque dataset se présentent différement, il vous revient de créer ces fichiers. Vous trouverez ci dessous des conseil et aide pour cette tâche.
Ces fichiers devront être déposé dans un dossier. Nous conseillons de les places dans `data/<corpus>` (exemple : `data/train` ou `data/test`).

## wav.scp

wav.scp contient les `uttID` et les chemins vers les fichiers wav de la session, sous la forme :
```
uttID1 /full/path/to/uttID1.wav
uttID2 /full/path/to/uttID2.wav
...
```

Le script python suivant peut servir de base pour la géneration d'un tel fichier.
```python
import os

path = "/full/path/to/wavs/"
with open("wav.scp", "w") as fout:
    for wav in os.listdir(path):
        fout.write(f"{wav.replace('.wav', '')} {path}{wav}\n")
```

## utt2spk

utt2spk contient les session et leur locuteur respectif sous la forme :
```
uttID1 spkIDX
uttID2 spkIDY
```

Le script python suivant peut servir de base pour la géneration d'un tel fichier.
```python
import os

path = "/full/path/to/wavs/"
split_char = "#" 
with open("utt2spk", "w") as fout:
    for wav in os.listdir(path):
        uttID = wav.replace('.wav', '')
        spkID = uttID.split(split_char)[0] # souvent, les nom de fichier wav de la forme spkID#uttID.wav
        fout.write(f"{uttID} {pkID}\n")
```

## spk2utt

spk2utt contient les ID locuteurs et toutes les sessions leurs appartenant : 
```
spkIDX uttID1 uttID3 uttID4
spkIDY uttID2 uttID5  
```

Vous pouvez generer ce fichier en utilisant l'outil `kaldi/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl` :
```
./path/to/utt2spk_to_spk2utt.pl utt2pk > spk2utt
```

# Étape 4

## Paramétrisation

La paramétrisation des données se fait dans le script `feature-extraction.sh`. Ce cript génère deux dossiers :
- le dossier spécifié dans `feature-out` contenant les features sous format binaire (format kaldi `.ark`). Ce dossier peut être lourd et sera fortement utilisé en lecture/écriture : nous déconseillons l'utilisation d'un SSD.
- le dossier spécifié dans `data-out` contenant les metadonnées (`wav.scp`, `utt2spk` et `spk2utt` ...) utilisé pour lire les features (cf [section sur fichiers ark et scp](#ark-&-scp)).

Ce script comporte les arguments suivant :
- `--help` affiche l'aide
- `--kaldi-path` le path COMPLET vers l'installation de kaldi
- `--data-in` permet de spécifier le dossier contenant les fichiers `wav.scp`, `utt2spk` et `spk2utt`.
- `--data-out` permet de spécifier le dossier de sortie des metadonnées.
- `--feature-out` permet de spécifier le dossier de sortie de features (peut être lourd)
- `--nj` permet de spécifier le nombre de process lancé en parallel. Attention, un nombre de process supérieur aux nombre de thread du CPU peut entrainer un rallentissement significatif.

Exemple :

```
./feature-extraction.sh --nj 8 --data-in data/train --data-out data/train-no-sil --features-out features/ --kaldi-path /home/me/kaldi/
```

## Data Augmentation

La paramétrisation des données peut parfois nécéssiter de la data augmentation (rajout de bruit et réverbération). Dans ce cas, la paramétrisation des données avec data augmentaion se fait dans le script `data-augmentation.sh`.

Ce script fonctionne de manière similaire à `feature-extraction.sh` et comporte des arguments similaires :
- `--help` affiche l'aide
- `--kaldi-path` le path COMPLET vers l'installation de kaldi
- `--data-in` permet de spécifier le dossier contenant les fichiers `wav.scp`, `utt2spk` et `spk2utt`.
- `--data-out` permet de spécifier le dossier de sortie des metadonnées.
- `--feature-out` permet de spécifier le dossier de sortie de features (peut être lourd)
- `--nj` permet de spécifier le nombre de process lancé en parallel. Attention, un nombre de process supérieur aux nombre de thread du CPU peut entrainer un rallentissement significatif.
- `--rirs-root` permet de spécifier le dossier contenant le dataset rirs noises (https://www.openslr.org/28/)
- `-musan-root` permet de spécifier le dossier contenant le dataset musan (http://www.openslr.org/17/)

Exemple:

```
./data-augmentation.sh --nj 8 --data-in data/train --data-out data/train-aug-no-sil --features-out features-aug/ --kaldi-root /home/me/kaldi --rirs-root RIRS_NOISES --musan-root musan
```
