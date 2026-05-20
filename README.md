# 🧠 Aletheia SNN Engine : Dynamic Spiking Neural Network Core

> **L'évolution neuronale à l'état pur. Zero-Allocation. Zero-Lock. Zero-Compromise.**

**Aletheia SNN Engine** est un moteur d'inférence et de mutation pour Réseaux de Neurones à Impulsions (SNN - *Spiking Neural Networks*). Conçu pour des environnements d'apprentissage par renforcement temps réel (Gymnasium, SC2) et le live-streaming interactif, ce moteur abandonne les paradigmes d'objets traditionnels pour embrasser le **Data-Oriented Design**.

L'objectif ? Simuler une plasticité synaptique et une évolution topologique en temps réel (NEAT/Apprentissage Hebbien) sans **aucune** allocation dynamique dans la boucle de rendu critique, tout en garantissant une télémétrie asynchrone à haute fréquence pour le monitoring WebGL.

---

## 🚀 Vision & Philosophie Architecturale

Les frameworks d'IA traditionnels (PyTorch, TensorFlow) sont optimisés pour des architectures statiques et des multiplications de matrices denses. Ils sont inadaptés pour des réseaux fortement creux (*sparse*) qui mutent en direct.

Pour répondre à ce défi, **Aletheia SNN Engine** est fondé sur trois piliers intransigeants :

1. **Zéro-Allocation (Hot Path) :** Le graphe neuronal est aplati dans une Memory Arena pré-allouée au format CSR (*Compressed Sparse Row*). Créer ou détruire une synapse n'est qu'une mutation d'index. Le Garbage Collector n'existe pas ici.
2. **Concurrence Asymétrique :** L'inférence (réflexe) et la mutation (évolution) vivent dans des threads étanches. La mise à jour de la topologie se fait via des pointeurs atomiques (`ArcSwap`). Le moteur de jeu n'attend jamais l'IA.
3. **Télémétrie Lock-Free :** L'extraction de l'état neuronal (spikes) vers l'interface visuelle s'effectue via un Ring Buffer atomique. L'UI peut s'effondrer, le moteur d'inférence ne perdra pas un seul cycle d'horloge.

---

## 🏗️ Topologie du Monstre (Workspace)

Le projet est divisé en trois sous-systèmes hautement découplés.

```text
aletheia-snn-engine/
├── core_engine/        # [Rust] Le Cœur du Réacteur
├── python_bridge/      # [Rust/PyO3] L'Interface FFI
└── twitch_dashboard/   # [TypeScript/WebGL] L'Œil Télémétrique

```

### 1. `core_engine` (Le Système Nerveux Central)

C'est ici que la magie noire opère. Entièrement écrit en Rust "bare-metal", ce module garantit la *Memory Safety* tout en forçant la localisation des données pour optimiser le Cache L1/L2 du CPU.

* **`arena.rs` :** Gestionnaire de *Memory Pools* contigus.
* **`brain.rs` :** Moteur d'inférence LIF (*Leaky Integrate-and-Fire*) basé sur des vecteurs plats. Exécution massivement parallélisable via SIMD (AVX-512).
* **`mutation.rs` :** Algorithmes génétiques asynchrones. Évalue la *fitness*, élague les synapses mortes, fait bourgeonner les nouvelles connexions.
* **`telemetry.rs` :** Serveur WebSocket embarqué écoutant un Ring Buffer SPSC (*Single-Producer Single-Consumer*).

### 2. `python_bridge` (Le Connecteur Frontalier)

Un wrapper *Zero-Cost* généré via `PyO3`. Il expose les `Traits` Rust abstraits de l'environnement sous forme d'une API Python standard.

* Permet au moteur d'ingérer n'importe quel environnement *Gymnasium* (ex: `BipedalWalker-v3`) de manière agnostique.
* Convertit les appels Python en structures mémoires Rust sans copie (ou avec un overhead minime).

### 3. `twitch_dashboard` (La Rétine)

Le client de visualisation haute performance. Conçu pour OBS et pour les interactions de la VTubeuse Aletheia.

* Écoute le flux binaire ultra-compressé du serveur WebSocket.
* Rendu WebGL (shaders) pour afficher des dizaines de milliers de neurones et de synapses en temps réel (144+ FPS) sans surcharger le thread principal.

---

## ⚙️ Cycle de Vie d'une Frame

La puissance de l'architecture se révèle dans la gestion d'une "Tick" (itération d'environnement) :

1. **Input :** Le `python_bridge` reçoit l'état des capteurs (ex: 24 floats pour BipedalWalker) et les injecte dans les neurones d'entrée du `FastBrain`.
2. **Propagation :** Le CPU parcourt les matrices CSR, calcule la fuite de potentiel (*Leaky*), intègre les courants, et déclenche les *spikes* (Impulsions).
3. **Output :** Les neurones de sortie dictent les actions du moteur physique.
4. **Télémétrie Asynchrone :** Un *bitmask* des neurones ayant spiké est poussé dans le Ring Buffer. Le thread WebSocket le dépile et l'envoie au GPU pour le rendu.
5. **Évolution Subconsciente :** En tâche de fond, le thread de mutation évalue les performances, recompile un nouveau graphe SNN, et effectue un *swap atomique* du cerveau principal. À la prochaine frame, le robot utilise une nouvelle topologie synaptique.

---

## 🛠️ Pré-requis & Compilation

Étant donné la nature agressive des optimisations, ce moteur exige un environnement de compilation moderne.

* **Rust :** Édition 2021 (Nightly recommandée pour certaines intrinsèques SIMD).
* **Python :** 3.10+ avec `venv` pour l'isolation.
* **Node.js :** 18+ (pour builder le dashboard Vite/WebGL).
* *(Optionnel mais recommandé) :* Architecture CPU supportant AVX2/AVX-512 pour une inférence maximale.

### Déploiement Rapide

```bash
# 1. Compiler le moteur core et le bridge Python en mode Release (crucial pour les performances)
cd python_bridge
maturin develop --release

# 2. Lancer le client WebGL de télémétrie
cd ../twitch_dashboard
npm install && npm run dev

# 3. Lancer l'entraînement (Exemple avec BipedalWalker)
cd ../scripts
python train_bipedal.py

```

---

## 🧬 Roadmap de l'Évolution

* [ ] Architecture CSR des matrices synaptiques.
* [ ] Moteur d'inférence LIF (Leaky Integrate-and-Fire) CPU.
* [ ] Pont FFI PyO3 pour compatibilité OpenAI Gym.
* [ ] Télémétrie Lock-Free via Ring Buffer & WebSocket.
* [ ] Shaders WebGL pour rendu SNN de masse (Twitch Dashboard).
* [ ] Offload de l'évaluation des mutations (Fitness) sur GPU via CUDA Pinned Memory.

> *"Perfection is achieved, not when there is nothing more to add, but when there are no more allocations left to remove."* — L'Architecte.