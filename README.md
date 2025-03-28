# language_translation_project
Projet : Aligner GPT-2 pour en faire un assistant (Instruction Tuning)
🎯 Objectif
L’objectif de ce projet est de transformer un modèle de langage standard (GPT-2) en un assistant capable de suivre des instructions. Les étudiants commenceront par utiliser GPT-2 dans sa forme brute, puis le fine-tuneront à l’aide d’un petit dataset d’instructions (comme OpenAssistant) et d’un apprentissage efficace via LoRA (Low-Rank Adaptation).

🧑‍🤝‍🧑 **Travail en groupe**
Groupes de 2 à 4 étudiants
Durée : 2 semaines
Liberté totale sur le choix du modèle/dataset (tant que la méthodologie est respectée)

🧩**Étapes du projet**
1. Chargement et test du modèle de base
Charger et utiliser GPT-2 (ou un modèle similaire, ex : DistilGPT2).
Générer du texte avec des prompts simples.

Constater que le modèle :
Ne suit pas les instructions (ex : "traduire", "résumer", "donner un conseil").
Se contente de compléter une phrase.

3. Choix d’un dataset d’instructions
Télécharger un petit dataset d'instructions. Exemples :
OpenAssistant (OASST1)
Alpaca
FLAN
Possibilité de créer un dataset personnalisé (500 à 2 000 exemples suffisent).

5. Fine-tuning avec LoRA (PEFT)
Utiliser les outils Hugging Face : transformers, peft, datasets, etc.
Geler le modèle de base et entraîner uniquement les couches LoRA.
Adapter les séquences (max 128 ou 256 tokens) pour entraîner rapidement sur GPU (Colab ou autre).
6. Évaluation et comparaison
Faire une conversation avant/après fine-tuning sur plusieurs types d'instructions.
Comparer qualitativement les réponses : pertinence, clarté, utilité.
(Bonus) Ajouter une interface simple en ligne de commande ou notebook pour tester l'assistant.

**📦 Livrables attendus**
📁 Code du projet (scripts, notebooks, requirements)
🧠 Modèle entraîné (uploadé sur Hugging Face ou sauvegardé localement)
📄 Exemples de prompts/réponses avant et après fine-tuning
(Optionnel) Un court rapport expliquant :
Le choix du dataset
La méthode de fine-tuning
Les limites observées

**🛠️ Ressources utiles**

Hugging Face Transformers

PEFT / LoRA (HF)

Tutoriel de fine-tuning avec LoRA

Colab avec GPU (si besoin)

**💬 Questions types pour tester le modèle**
Traduis : "Je vais bien, merci."
Résume ce texte : ...
Donne-moi une idée de projet.
Que puis-je faire aujourd’hui ?
Quels sont les avantages du sport ?

**🤖 Objectif final**
À la fin du projet, vous aurez :

Expérimenté avec un vrai modèle de langage non aligné.
Appris à le fine-tuner avec LoRA de façon efficace.
Créé une version “assistante” capable de suivre des instructions !
