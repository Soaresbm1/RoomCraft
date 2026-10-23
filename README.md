# RoomCraft

Site Web compilé : scan caméra, reconstruction 3D locale sur le téléphone et atelier 2D/3D.

https://soaresbm1.github.io/RoomCraft/

Scanner la pièce, terminer le scan puis laisser le téléphone calculer. Aucun PC, ZIP ou serveur photo requis. Les relevés 3D et les prochaines séries de photos sont enregistrés sur cet appareil lorsque le stockage local est disponible. Les anciennes séries peuvent être reprises sans ZIP ; une erreur de stockage conserve les nouvelles vues en mémoire et doit être résolue avant de fermer la page. Un diagnostic du calcul peut être copié sans photos. Une hauteur sol-plafond mesurée est nécessaire pour des dimensions en mètres ; la précision réelle reste à vérifier. Les surfaces non observées ne sont pas inventées. Mode expérimental, sans garantie de reconstruction pour des murs sans texture ou réfléchissants.

GitHub Pages : branche `main`, dossier `/docs`. Aucune image ni aucun projet utilisateur ne figure dans ce dépôt. La reconstruction Python reste une option locale séparée.

Moteur expérimental non officiel colmap-wasm 0.1.0 (BSD-3-Clause), COLMAP 4.2.1 et composants tiers : notices dans `docs/THIRD_PARTY_NOTICES.txt`, provenance dans `docs/COLMAP_WASM_PROVENANCE.json`. Le portage n’est pas approuvé par les auteurs de COLMAP. Eigen 3.4.0 non modifié (MPL-2.0) : sources https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
