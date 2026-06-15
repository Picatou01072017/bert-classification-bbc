"""
model.py
--------
Chargement du modèle BERT pré-entraîné et définition de la tête de classification.
On utilise BertForSequenceClassification de Hugging Face qui ajoute automatiquement
une couche linéaire (dropout + linear) au-dessus du token [CLS].
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from pathlib import Path
