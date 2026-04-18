"""French disease labels and UI-ready agronomic information for Streamlit."""

from __future__ import annotations

SPECIES_LABELS = {
    "tomato": "Tomate",
    "apple": "Pommier",
    "grape": "Vigne",
    "corn": "Maïs",
    "potato": "Pomme de terre",
    "pepper": "Poivron",
    "strawberry": "Fraisier",
}

DISEASE_LABELS = {
    "Apple_Scab": "Tavelure du pommier",
    "Black_Rot": "Pourriture noire",
    "Cedar_Apple_Rust": "Rouille grillagée du pommier",
    "Cercospora_Leaf_Spot": "Tache grise / cercosporiose du maïs",
    "Common_Rust": "Rouille commune du maïs",
    "Northern_Leaf_Blight": "Brûlure septentrionale des feuilles",
    "Esca_Black_Measles": "Esca / black measles",
    "Leaf_Blight": "Brûlure des feuilles / Isariopsis",
    "Bacterial_Spot": "Tache bactérienne",
    "Early_Blight": "Mildiou précoce / alternariose",
    "Late_Blight": "Mildiou tardif",
    "Leaf_Scorch": "Brûlure des feuilles",
    "Leaf_Mold": "Moisissure des feuilles",
    "Mosaic_Virus": "Virus de la mosaïque de la tomate",
    "Septoria_Leaf_Spot": "Septoriose",
    "Spider_Mites": "Acariens tétranyques",
    "Target_Spot": "Tache cible",
    "Yellow_Leaf_Curl_Virus": "Virus de l'enroulement jaune des feuilles",
    "Healthy": "Feuille saine",
}

DISEASE_INFO = {
    ("apple", "Apple_Scab"): {
        "title": "Tavelure du pommier",
        "english": "Apple scab",
        "url": "https://fr.wikipedia.org/wiki/Tavelure_du_pommier",
        "agent": (
            "Champignon Venturia inaequalis. Il hiverne dans les feuilles mortes "
            "et infecte surtout les jeunes feuilles et fruits au printemps."
        ),
        "curative": (
            "Fongicide appliqué très tôt après risque d'infection ; efficacité "
            "limitée si les symptômes sont déjà bien installés."
        ),
        "preventive": (
            "Ramasser les feuilles mortes, tailler pour aérer, choisir des variétés "
            "résistantes et protéger préventivement en période à risque."
        ),
        "season_gravity": (
            "Surtout printemps, puis contaminations secondaires possibles en saison. "
            "Gravité élevée : forte défoliation et baisse de qualité/rendement."
        ),
    },
    ("apple", "Black_Rot"): {
        "title": "Pourriture noire du pommier",
        "english": "Apple black rot",
        "url": "https://fr.wikipedia.org/wiki/Botryosphaeria_obtusa",
        "agent": (
            "Champignon Botryosphaeria obtusa. Il survit dans les chancres, le bois "
            "mort et les fruits momifiés."
        ),
        "curative": (
            "Couper et éliminer les rameaux ou organes atteints ; traitement fongicide "
            "précoce possible pour freiner l'extension."
        ),
        "preventive": (
            "Retirer fruits momifiés et bois mort, désinfecter les outils, limiter "
            "les blessures et maintenir l'arbre vigoureux."
        ),
        "season_gravity": (
            "Printemps à été, favorisée par chaleur et humidité. Gravité modérée à "
            "élevée : peut affecter feuilles, fruits et rameaux."
        ),
    },
    ("apple", "Cedar_Apple_Rust"): {
        "title": "Rouille grillagée du pommier",
        "english": "Cedar apple rust",
        "url": "https://fr.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae",
        "agent": (
            "Champignon Gymnosporangium juniperi-virginianae, avec cycle entre "
            "pommier et genévrier."
        ),
        "curative": (
            "Pas de vrai curatif sur feuille déjà atteinte ; l'objectif est surtout "
            "de bloquer les nouvelles infections."
        ),
        "preventive": (
            "Éloigner ou surveiller les genévriers hôtes, couper les galles, choisir "
            "des variétés tolérantes et protéger au printemps."
        ),
        "season_gravity": (
            "Printemps, lors des pluies douces au moment de la floraison. Gravité "
            "modérée, surtout foliaire."
        ),
    },
    ("corn", "Cercospora_Leaf_Spot"): {
        "title": "Tache grise / cercosporiose du maïs",
        "english": "Gray leaf spot",
        "url": "https://fr.wikipedia.org/wiki/Cercosporiose_du_ma%C3%AFs",
        "agent": (
            "Champignons Cercospora zeae-maydis et parfois C. zeina. Ils survivent "
            "dans les résidus de culture."
        ),
        "curative": (
            "Fongicide foliaire en végétation si pression forte, surtout pour protéger "
            "les feuilles du haut avant ou après floraison."
        ),
        "preventive": "Rotation, gestion des résidus, hybrides résistants, densité et aération adaptées.",
        "season_gravity": (
            "Été, favorisée par chaleur et forte humidité. Gravité élevée : baisse "
            "notable du remplissage du grain si l'attaque monte tôt."
        ),
    },
    ("corn", "Common_Rust"): {
        "title": "Rouille commune du maïs",
        "english": "Common rust",
        "url": "https://fr.wikipedia.org/wiki/Puccinia_sorghi",
        "agent": "Champignon Puccinia sorghi, disséminé par spores aériennes.",
        "curative": (
            "Pas de guérison des pustules existantes ; traitement foliaire possible "
            "pour ralentir la progression si attaque précoce."
        ),
        "preventive": "Hybrides résistants, surveillance régulière et gestion adaptée selon pression locale.",
        "season_gravity": (
            "Été, surtout conditions fraîches à modérées et humides. Gravité faible "
            "à modérée, plus gênante si l'attaque est précoce."
        ),
    },
    ("corn", "Northern_Leaf_Blight"): {
        "title": "Brûlure septentrionale des feuilles",
        "english": "Northern corn leaf blight",
        "url": "https://fr.wikipedia.org/wiki/Helminthosporiose_du_ma%C3%AFs",
        "agent": "Champignon Exserohilum turcicum. Il hiverne dans les résidus de maïs.",
        "curative": (
            "Fongicide foliaire en végétation quand les premières lésions progressent "
            "et que la météo reste favorable."
        ),
        "preventive": (
            "Hybrides résistants, rotation, destruction ou enfouissement des résidus "
            "et bonne gestion parcellaire."
        ),
        "season_gravity": (
            "Été, favorisée par humidité prolongée et températures modérées. Gravité "
            "élevée si installée avant ou autour de la floraison."
        ),
    },
    ("grape", "Black_Rot"): {
        "title": "Pourriture noire de la vigne",
        "english": "Black rot",
        "url": "https://fr.wikipedia.org/wiki/Pourriture_noire_de_la_vigne",
        "agent": (
            "Champignon Guignardia bidwellii. Il hiverne surtout dans les baies "
            "momifiées et tissus infectés."
        ),
        "curative": "Intervention fongicide précoce possible, mais l'assainissement reste essentiel.",
        "preventive": (
            "Retirer les baies momifiées, aérer la vigne, réduire l'humidité et "
            "protéger au printemps-début été."
        ),
        "season_gravity": (
            "Printemps à été, avec temps chaud et humide. Gravité très élevée : "
            "peut compromettre fortement la récolte."
        ),
    },
    ("grape", "Esca_Black_Measles"): {
        "title": "Esca / black measles",
        "english": "Esca / black measles",
        "url": "https://fr.wikipedia.org/wiki/Esca_(maladie)",
        "agent": (
            "Complexe de champignons du bois, notamment Phaeomoniella chlamydospora, "
            "Phaeoacremonium spp. et Fomitiporia mediterranea."
        ),
        "curative": (
            "Pas de vrai traitement curatif fiable ; on retire souvent les parties "
            "très atteintes, voire le cep."
        ),
        "preventive": "Protéger les plaies de taille, utiliser du matériel sain, limiter les contaminations du bois.",
        "season_gravity": (
            "Symptômes foliaires surtout en été ; maladie chronique sur plusieurs "
            "années. Gravité élevée."
        ),
    },
    ("grape", "Leaf_Blight"): {
        "title": "Brûlure des feuilles / Isariopsis leaf spot",
        "english": "Isariopsis leaf spot",
        "url": "https://fr.wikipedia.org/wiki/Pseudocercospora_vitis",
        "agent": "Champignon Pseudocercospora vitis.",
        "curative": "Fongicide foliaire possible dès les premiers symptômes si pression réelle.",
        "preventive": "Assainissement, aération de la canopée et gestion de l'humidité.",
        "season_gravity": (
            "Fin d'été à automne, surtout par temps chaud et humide. Gravité faible "
            "à modérée."
        ),
    },
    ("pepper", "Bacterial_Spot"): {
        "title": "Tache bactérienne du poivron",
        "english": "Bacterial spot",
        "url": "https://fr.wikipedia.org/wiki/Xanthomonas_euvesicatoria",
        "agent": "Bactéries du genre Xanthomonas, plusieurs espèces ou pathovars selon les cas.",
        "curative": "Pas de vrai curatif ; les produits cupriques peuvent seulement freiner l'évolution.",
        "preventive": (
            "Semences ou plants sains, désinfection, rotation, irrigation évitant de "
            "mouiller le feuillage, variétés résistantes quand disponibles."
        ),
        "season_gravity": "Été ou serre chaude/humide. Gravité élevée : dommageable sur feuillage et fruits.",
    },
    ("potato", "Early_Blight"): {
        "title": "Mildiou précoce / alternariose",
        "english": "Early blight",
        "url": "https://fr.wikipedia.org/wiki/Alternariose_de_la_pomme_de_terre",
        "agent": "Champignon Alternaria solani, favorisé par le stress des plantes et les résidus contaminés.",
        "curative": "Fongicide dès les premières lésions pour limiter la défoliation.",
        "preventive": "Rotation, destruction des résidus, nutrition et irrigation équilibrées, variétés tolérantes.",
        "season_gravity": (
            "Été, plutôt en deuxième partie de cycle. Gravité modérée à élevée si la "
            "défoliation devient importante."
        ),
    },
    ("potato", "Late_Blight"): {
        "title": "Mildiou tardif",
        "english": "Late blight",
        "url": "https://fr.wikipedia.org/wiki/Mildiou_de_la_pomme_de_terre",
        "agent": (
            "Oomycète Phytophthora infestans. Il se conserve surtout dans les "
            "tubercules infectés et se disperse rapidement par sporanges."
        ),
        "curative": (
            "Réaction très rapide nécessaire ; produits anti-mildiou pour freiner "
            "les nouvelles contaminations, avec élimination des foyers si besoin."
        ),
        "preventive": (
            "Plants sains, surveillance météo, protection préventive, destruction "
            "des repousses et déchets de pommes de terre."
        ),
        "season_gravity": "Été, surtout temps frais à doux et très humide. Gravité très élevée.",
    },
    ("strawberry", "Leaf_Scorch"): {
        "title": "Brûlure des feuilles",
        "english": "Leaf scorch",
        "url": "https://fr.wikipedia.org/wiki/Diplocarpon_earlianum",
        "agent": "Champignon Diplocarpon earlianum, conservé dans les débris infectés.",
        "curative": "Fongicide possible dès les premiers symptômes, surtout si les conditions restent humides.",
        "preventive": "Plants sains, rotation, paillage, irrigation sans mouiller le feuillage et nettoyage des débris.",
        "season_gravity": "Printemps à été. Gravité modérée : affaiblit les plants et peut réduire la production.",
    },
    ("tomato", "Bacterial_Spot"): {
        "title": "Tache bactérienne",
        "english": "Bacterial spot",
        "url": "https://fr.wikipedia.org/wiki/Xanthomonas_vesicatoria",
        "agent": "Bactéries du genre Xanthomonas, favorisées par éclaboussures, manipulation et humidité.",
        "curative": "Pas de curatif réel ; les produits à base de cuivre aident surtout à limiter la propagation.",
        "preventive": "Semences ou plants sains, rotation, hygiène stricte, irrigation localisée et destruction des résidus.",
        "season_gravity": "Été chaud et humide. Gravité élevée : forte défoliation et pertes sur fruits.",
    },
    ("tomato", "Early_Blight"): {
        "title": "Mildiou précoce / alternariose",
        "english": "Early blight",
        "url": "https://fr.wikipedia.org/wiki/Alternaria_solani",
        "agent": (
            "Champignon Alternaria solani. Il attaque surtout les feuilles basses "
            "et forme des lésions concentriques typiques."
        ),
        "curative": "Fongicide précoce et suppression des feuilles très atteintes.",
        "preventive": "Rotation, paillage, tuteurage, aération, élimination des résidus, fertilisation équilibrée.",
        "season_gravity": "Été. Gravité modérée à élevée : défoliation progressive et brûlure solaire des fruits.",
    },
    ("tomato", "Late_Blight"): {
        "title": "Mildiou tardif",
        "english": "Late blight",
        "url": "https://fr.wikipedia.org/wiki/Phytophthora_infestans",
        "agent": "Oomycète Phytophthora infestans.",
        "curative": (
            "Traitement immédiat anti-mildiou pour contenir la flambée ; efficacité "
            "limitée si l'attaque est déjà avancée."
        ),
        "preventive": "Plants sains, aération, surveillance météo, protection préventive en période fraîche et humide.",
        "season_gravity": "Été, surtout temps frais à doux et humide. Gravité très élevée : progression très rapide.",
    },
    ("tomato", "Leaf_Mold"): {
        "title": "Moisissure des feuilles",
        "english": "Leaf mold",
        "url": "https://fr.wikipedia.org/wiki/Cladosporiose_de_la_tomate",
        "agent": "Champignon Passalora fulva, anciennement Cladosporium fulvum, très fréquent sous serre.",
        "curative": "Fongicide possible au début, avec retrait des feuilles très touchées.",
        "preventive": "Ventilation, baisse de l'humidité, irrigation au sol, désinfection de serre et variétés résistantes.",
        "season_gravity": "Surtout en serre lorsque l'humidité reste élevée. Gravité modérée.",
    },
    ("tomato", "Mosaic_Virus"): {
        "title": "Virus de la mosaïque de la tomate",
        "english": "Tomato mosaic virus",
        "url": "https://fr.wikipedia.org/wiki/Virus_de_la_mosa%C3%AFque_de_la_tomate",
        "agent": "Tomato mosaic virus, virus très stable transmis surtout mécaniquement.",
        "curative": "Aucun ; arracher et détruire les plants atteints.",
        "preventive": "Semences saines, hygiène stricte, désinfection des mains/outils et variétés résistantes.",
        "season_gravity": "Toute la saison. Gravité modérée à élevée : baisse de rendement et fruits déformés.",
    },
    ("tomato", "Septoria_Leaf_Spot"): {
        "title": "Septoriose",
        "english": "Septoria leaf spot",
        "url": "https://fr.wikipedia.org/wiki/Septoria_lycopersici",
        "agent": "Champignon Septoria lycopersici, souvent parti du bas du feuillage.",
        "curative": "Fongicide dès les premiers points nécrotiques et retrait des feuilles les plus atteintes.",
        "preventive": "Rotation, paillage, arrosage au pied, tuteurage, suppression des débris et adventices hôtes.",
        "season_gravity": "Été humide. Gravité élevée : peut défolier fortement la plante.",
    },
    ("tomato", "Spider_Mites"): {
        "title": "Acariens tétranyques",
        "english": "Two-spotted spider mite",
        "url": "https://fr.wikipedia.org/wiki/Tetranychus_urticae",
        "agent": "Acarien Tetranychus urticae, qui pique les cellules foliaires et tisse de fines toiles.",
        "curative": "Acaricide ciblé, savon/huile selon contexte, ou lutte biologique par acariens prédateurs.",
        "preventive": "Surveillance sous les feuilles, culture moins stressée et préservation des auxiliaires.",
        "season_gravity": "Été chaud et sec, ou toute l'année sous serre. Gravité modérée à élevée.",
    },
    ("tomato", "Target_Spot"): {
        "title": "Tache cible",
        "english": "Target spot",
        "url": "https://fr.wikipedia.org/wiki/Corynespora_cassiicola",
        "agent": "Champignon Corynespora cassiicola.",
        "curative": "Fongicide dès les premières lésions, surtout si climat durablement humide.",
        "preventive": "Rotation, aération, plants sains et destruction des résidus.",
        "season_gravity": "Été chaud et humide. Gravité modérée à élevée : peut toucher feuilles et fruits.",
    },
    ("tomato", "Yellow_Leaf_Curl_Virus"): {
        "title": "Virus de l'enroulement jaune des feuilles",
        "english": "Tomato yellow leaf curl virus",
        "url": "https://fr.wikipedia.org/wiki/Virus_de_l%27enroulement_jaune_en_cuill%C3%A8re_de_la_tomate",
        "agent": "Bégomovirus TYLCV, transmis par l'aleurode Bemisia tabaci.",
        "curative": "Aucun ; arracher rapidement les plants infectés.",
        "preventive": "Variétés résistantes, contrôle des aleurodes, filets, gestion des adventices hôtes.",
        "season_gravity": "Toute la saison en zones chaudes, surtout été en zones tempérées. Gravité très élevée.",
    },
}
