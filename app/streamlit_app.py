"""Streamlit frontend for the plant disease diagnosis API."""

from __future__ import annotations

import base64
import html
from typing import Any

import streamlit as st
from dotenv import load_dotenv

try:
    from app.api_client import (
        cached_get_monitoring_events,
        cached_get_monitoring_summary,
        call_predict_api,
        get_api_health,
        get_api_url,
        get_models_info,
        get_monitoring_events,
        get_monitoring_summary,
        submit_feedback,
    )
except ModuleNotFoundError:
    from api_client import (
        cached_get_monitoring_events,
        cached_get_monitoring_summary,
        call_predict_api,
        get_api_health,
        get_api_url,
        get_models_info,
        get_monitoring_events,
        get_monitoring_summary,
        submit_feedback,
    )

try:
    from app.disease_info import DISEASE_INFO, DISEASE_LABELS, SPECIES_LABELS
except ModuleNotFoundError:
    from disease_info import DISEASE_INFO, DISEASE_LABELS, SPECIES_LABELS

SPECIES_OPTIONS = SPECIES_LABELS


def main() -> None:
    """Render the Streamlit application."""

    load_dotenv()
    configure_page()

    api_url = get_api_url()
    render_header()
    page = render_sidebar(api_url)

    if page == "Monitoring":
        render_monitoring_page(api_url)
        return

    progress_placeholder = st.empty()
    uploaded_file, selected_species, analyze_clicked = render_input_panel()

    if analyze_clicked and uploaded_file is not None:
        run_prediction(
            api_url=api_url,
            uploaded_file=uploaded_file,
            species=selected_species,
            progress_placeholder=progress_placeholder,
        )

    render_last_result(api_url)


def configure_page() -> None:
    """Configure page metadata and styling."""

    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1120px;
            padding-top: 2rem;
        }
        .diagnosis-badge {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            min-height: 130px;
        }
        .diagnosis-label {
            color: #465264;
            font-size: 0.86rem;
            margin-bottom: 0.25rem;
        }
        .diagnosis-value {
            color: #18212f;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .confidence-bar {
            height: 4px;
            background: #e2e8f0;
            border-radius: 2px;
            margin-top: 0.55rem;
            margin-bottom: 0.35rem;
        }
        .confidence-bar.compact {
            height: 5px;
            margin-top: 0.25rem;
            margin-bottom: 0.6rem;
        }
        .confidence-bar-fill {
            height: 100%;
            background: #16735f;
            border-radius: 2px;
        }
        .diagnosis-confidence {
            color: #16735f;
            font-size: 0.86rem;
        }
        .responsive-image {
            width: min(100%, 360px);
            max-height: 360px;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .responsive-image.upload-preview {
            width: min(100%, 420px);
            max-height: 320px;
        }
        .image-caption {
            color: #6b7280;
            font-size: 0.84rem;
            text-align: center;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .info-panel {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #f8fafc;
            margin-top: 0.8rem;
        }
        .info-panel h4 {
            margin-top: 0;
            margin-bottom: 0.75rem;
            color: #18212f;
        }
        .info-panel p {
            margin: 0.45rem 0;
            line-height: 1.45;
        }
        .info-row {
            display: grid;
            grid-template-columns: 1.45rem minmax(0, 1fr);
            column-gap: 0.35rem;
            align-items: start;
            margin: 0.6rem 0;
            line-height: 1.45;
        }
        .info-icon {
            text-align: center;
            line-height: 1.45;
        }
        .ranked-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-top: 0.4rem;
        }
        .ranked-label {
            font-weight: 650;
            color: #18212f;
        }
        .ranked-confidence {
            color: #16735f;
            font-weight: 700;
        }
        .monitoring-card {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            background: #ffffff;
            min-height: 136px;
        }
        .monitoring-card.good {
            border-left: 5px solid #16735f;
        }
        .monitoring-card.watch {
            border-left: 5px solid #b7791f;
        }
        .monitoring-card.critical {
            border-left: 5px solid #b42318;
        }
        .monitoring-card.neutral {
            border-left: 5px solid #2f6db3;
        }
        .monitoring-label {
            color: #5b6678;
            font-size: 0.82rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }
        .monitoring-value {
            color: #18212f;
            font-size: 1.4rem;
            font-weight: 750;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        .monitoring-help {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.35;
            margin-top: 0.55rem;
        }
        .monitoring-note {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #f8fafc;
            color: #344054;
            line-height: 1.45;
            margin: 0.5rem 0 1rem;
        }
        .chart-panel {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            background: #ffffff;
        }
        .chart-title {
            color: #18212f;
            font-weight: 750;
            margin-bottom: 0.85rem;
        }
        .bar-row {
            display: grid;
            grid-template-columns: minmax(7rem, 11rem) minmax(0, 1fr) 3.5rem;
            gap: 0.75rem;
            align-items: center;
            margin: 0.58rem 0;
        }
        .bar-label {
            color: #344054;
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }
        .bar-track {
            height: 0.72rem;
            border-radius: 6px;
            background: #e7edf4;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            background: #16735f;
        }
        .bar-value {
            color: #18212f;
            font-size: 0.88rem;
            font-weight: 700;
            text-align: right;
        }
        .chart-hint {
            color: #64748b;
            font-size: 0.8rem;
            min-height: 1rem;
            margin-top: 0.55rem;
        }
        @media (prefers-color-scheme: dark) {
            .diagnosis-badge {
                background: #1a2332;
                border-color: #2d4060;
            }
            .diagnosis-label { color: #8ba4c0; }
            .diagnosis-value { color: #e2eaf4; }
            .confidence-bar { background: #2d4060; }
            .confidence-bar-fill { background: #4dba99; }
            .diagnosis-confidence { color: #4dba99; }
            .info-panel {
                background: #111827;
                border-color: #2d4060;
            }
            .info-panel h4,
            .ranked-label { color: #e2eaf4; }
            .ranked-confidence { color: #4dba99; }
            .image-caption { color: #9ca3af; }
            .monitoring-card {
                background: #111827;
                border-color: #2d4060;
            }
            .monitoring-label,
            .monitoring-help { color: #9ca3af; }
            .monitoring-value { color: #e2eaf4; }
            .monitoring-note {
                background: #111827;
                border-color: #2d4060;
                color: #d5dde8;
            }
            .chart-panel {
                background: #111827;
                border-color: #2d4060;
            }
            .chart-title,
            .bar-value { color: #e2eaf4; }
            .bar-label,
            .chart-hint { color: #9ca3af; }
            .bar-track { background: #2d4060; }
            .bar-fill { background: #4dba99; }
        }
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .responsive-image,
            .responsive-image.upload-preview {
                width: min(100%, 320px);
                max-height: 280px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the page title and short description."""

    st.title("🌿 Diagnostic foliaire par image")
    st.caption(
        "Chargez une photo de feuille pour identifier l'espèce et détecter une maladie éventuelle."
    )


def render_sidebar(api_url: str) -> str:
    """Render service status and supported species."""

    st.sidebar.title("Plant Disease Detection")
    st.sidebar.caption("Diagnostic de maladies foliaires par IA")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=["Diagnostic", "Monitoring"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    health = get_api_health(api_url)
    models_info = get_models_info(api_url)

    st.sidebar.subheader("État du service")
    if health.status_code == 200:
        st.sidebar.success("Service en ligne")
    else:
        st.sidebar.error("Service hors ligne")

    if models_info.status_code == 200 and models_info.payload.get("config_available"):
        st.sidebar.success("Modèles chargés")
    elif models_info.status_code == 200:
        st.sidebar.warning("Modèles en attente")

    st.sidebar.divider()
    st.sidebar.subheader("Espèces supportées")
    for label in SPECIES_OPTIONS.values():
        st.sidebar.markdown(f"• {label}")

    return page


def render_input_panel() -> tuple[Any | None, str | None, bool]:
    """Render upload, analysis mode controls and submit button."""

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Photo de feuille",
            type=["jpg", "jpeg", "png", "webp"],
            help="Utilisez une image nette, idéalement centrée sur une feuille.",
        )
        if uploaded_file is not None:
            render_responsive_image(
                uploaded_file.getvalue(),
                caption="Image chargée",
                css_class="upload-preview",
            )
        else:
            st.info("Ajoutez une image pour lancer le diagnostic.")

    with right:
        st.subheader("Mode d'analyse")
        mode = st.radio(
            "Choisir le mode",
            ["Automatique", "Manuel"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if mode == "Automatique":
            st.write("L'espèce est détectée automatiquement avant le diagnostic maladie.")
        else:
            st.write("Indiquez l'espèce pour un diagnostic direct, sans détection préalable.")

        selected_species = None
        if mode == "Manuel":
            selected_label = st.selectbox(
                "Espèce connue",
                options=list(SPECIES_OPTIONS.values()),
                index=0,
            )
            selected_species = label_to_species(selected_label)

        st.write("")
        analyze_clicked = st.button(
            "Analyser l'image",
            type="primary",
            disabled=uploaded_file is None,
            width="stretch",
        )

    return uploaded_file, selected_species, analyze_clicked


def run_prediction(
    *,
    api_url: str,
    uploaded_file: Any,
    species: str | None,
    progress_placeholder: Any,
) -> None:
    """Call the prediction API and store the latest result in session state."""

    image_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name or "leaf.png"

    progress_placeholder.progress(
        10,
        text="Préparation de l'image. Le premier appel peut prendre un peu de temps.",
    )
    progress_placeholder.progress(35, text="Envoi à l'API et chargement des modèles.")
    response = call_predict_api(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
        species=species,
    )
    progress_placeholder.progress(100, text="Analyse terminée.")
    progress_placeholder.empty()

    st.session_state["last_response"] = response.payload
    st.session_state["last_status_code"] = response.status_code
    st.session_state["last_image_bytes"] = image_bytes
    st.session_state["last_image_name"] = filename
    st.session_state["last_feedback_sent"] = False


def render_monitoring_page(api_url: str) -> None:
    """Render a visual monitoring dashboard for API predictions."""

    st.subheader("Monitoring du service IA")
    st.markdown(
        """
        <div class="monitoring-note">
            Lecture rapide : le dashboard suit la disponibilité de l'API, la confiance du
            modèle, les signaux de drift et les retours utilisateur. Les images ne sont pas
            stockées ; seuls des événements JSONL et des métriques dérivées sont conservés
            dans le bucket Hugging Face monté sur /data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Rafraîchir", width="content"):
        cached_get_monitoring_summary.clear()
        cached_get_monitoring_events.clear()

    response = get_monitoring_summary(api_url)
    if response.status_code != 200:
        st.error(response.payload.get("detail", "Monitoring indisponible."))
        return
    events_response = get_monitoring_events(api_url, limit=100)

    payload = response.payload
    render_monitoring_overview(payload)

    render_alerts(payload.get("alerts", []))

    overview_tab, drift_tab, events_tab = st.tabs(
        ["Flux observé", "Drift et feedback", "Journal technique"]
    )
    with overview_tab:
        render_distribution_section(payload)
    with drift_tab:
        render_drift_section(
            payload.get("domain_shift", {}),
            payload.get("model_quality_shift", {}),
        )
        st.divider()
        render_feedback_summary(payload.get("feedback", {}))
    with events_tab:
        if events_response.status_code == 200:
            render_recent_events(events_response.payload.get("events", []))
        else:
            st.warning(events_response.payload.get("detail", "Événements indisponibles."))

    last_event_at = payload.get("last_event_at")
    if last_event_at:
        st.caption(f"Dernier événement : {last_event_at}")
    else:
        st.info("Aucune prédiction enregistrée depuis le dernier démarrage de l'API.")

    with st.expander("Réponse brute de l'API"):
        st.json(payload)


def render_monitoring_overview(payload: dict[str, Any]) -> None:
    """Render the main monitoring status cards."""

    total_predictions = int(payload.get("total_predictions") or 0)
    ok_rate = float(payload.get("ok_rate") or 0.0)
    error_rate = float(payload.get("error_rate") or 0.0)
    uncertain_rate = float(payload.get("uncertain_rate") or 0.0)
    low_confidence_rate = float(payload.get("low_confidence_rate") or 0.0)
    domain_shift = payload.get("domain_shift", {})
    feedback_summary = payload.get("feedback", {})

    service_tone = "good"
    service_label = "Service stable"
    if error_rate > 0.05:
        service_tone = "critical"
        service_label = "Erreurs à surveiller"
    elif uncertain_rate > 0.25:
        service_tone = "watch"
        service_label = "Beaucoup d'incertitudes"

    domain_status = "insufficient_data"
    domain_tone = "neutral"
    if isinstance(domain_shift, dict):
        domain_status = str(domain_shift.get("status", "insufficient_data"))
        domain_tone = risk_to_tone(str(domain_shift.get("risk_level", "none")))

    high_confidence_disagreement_count = 0
    high_confidence_disagreement_rate = 0.0
    high_confidence_threshold = 0.9
    if isinstance(feedback_summary, dict):
        high_confidence_disagreement_count = int(
            feedback_summary.get("high_confidence_disagreement_count") or 0
        )
        high_confidence_disagreement_rate = float(
            feedback_summary.get("high_confidence_disagreement_rate") or 0.0
        )
        high_confidence_threshold = float(
            feedback_summary.get("high_confidence_threshold") or 0.9
        )

    if total_predictions and total_predictions < 10:
        st.info(
            "Échantillon encore faible : les taux et la latence P95 sont indicatifs "
            "tant que peu de prédictions ont été enregistrées."
        )

    first_row = st.columns(3, gap="medium")
    with first_row[0]:
        render_monitoring_card(
            "Volume observé",
            str(total_predictions),
            "Nombre de prédictions enregistrées dans le JSONL de monitoring.",
            tone="neutral",
        )
    with first_row[1]:
        render_monitoring_card(
            "Santé API",
            service_label,
            f"{format_optional_percent(ok_rate)} OK · "
            f"{format_optional_percent(error_rate)} erreurs · "
            f"{format_optional_percent(uncertain_rate)} incertitudes.",
            tone=service_tone,
        )
    with first_row[2]:
        render_monitoring_card(
            "Latence P95",
            format_ms(payload.get("p95_latency_ms")),
            "95 % des requêtes sont plus rapides que cette valeur ; les cold starts peuvent la tirer vers le haut.",
            tone="watch" if _numeric(payload.get("p95_latency_ms")) > 5000 else "good",
        )

    second_row = st.columns(3, gap="medium")
    with second_row[0]:
        render_monitoring_card(
            "Scores faibles",
            format_optional_percent(low_confidence_rate),
            "Part de prédictions où au moins une confiance passe sous le seuil ; le modèle sait qu'il hésite.",
            tone="watch" if low_confidence_rate > 0.25 else "good",
        )
    with second_row[1]:
        render_monitoring_card(
            "Domaine image",
            format_domain_status(domain_status),
            "Similarité des dernières images avec PlantVillage et PlantDoc.",
            tone=domain_tone,
        )
    with second_row[2]:
        render_monitoring_card(
            "Forte confiance contestée",
            f"{high_confidence_disagreement_count} cas",
            f"Feedback incorrect avec confiance >= {format_optional_percent(high_confidence_threshold)} "
            f"({format_optional_percent(high_confidence_disagreement_rate)} des retours).",
            tone="watch" if high_confidence_disagreement_count else "good",
        )


def render_monitoring_card(label: str, value: str, help_text: str, *, tone: str) -> None:
    """Render a compact explanatory monitoring card."""

    safe_tone = tone if tone in {"good", "watch", "critical", "neutral"} else "neutral"
    st.markdown(
        f"""
        <div class="monitoring-card {safe_tone}">
            <div class="monitoring-label">{html.escape(label)}</div>
            <div class="monitoring-value">{html.escape(value)}</div>
            <div class="monitoring-help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alerts(alerts: Any) -> None:
    """Render active monitoring alerts."""

    st.markdown("#### Alertes actives")
    if not alerts:
        st.success("Aucune alerte active : les seuils de service, confiance, drift et feedback restent sous contrôle.")
        return
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        message = alert.get("message", "Alerte monitoring")
        value = alert.get("value")
        threshold = alert.get("threshold")
        st.warning(f"{message} Valeur observée : {value} ; seuil configuré : {threshold}.")


def render_distribution_section(payload: dict[str, Any]) -> None:
    """Render the production flow distribution."""

    st.markdown("#### Flux de prédictions")
    st.caption(
        "Objectif : vérifier si l'usage réel se concentre sur quelques espèces. "
        "Un flux très déséquilibré peut expliquer des alertes ou orienter la collecte future."
    )
    left, right = st.columns([1.7, 1], gap="medium")
    with left:
        render_horizontal_bars(
            "Top espèces observées",
            payload.get("species_distribution", {}),
            max_items=6,
        )
    with right:
        healthy_ratio = payload.get("healthy_ratio")
        render_monitoring_card(
            "Feuilles saines",
            format_optional_percent(healthy_ratio),
            "Part des diagnostics maladie classés Healthy. Calculé seulement quand une maladie est prédite.",
            tone="neutral",
        )
        st.caption(
            "Les distributions détaillées des maladies restent disponibles dans la réponse brute "
            "pour audit, mais ne sont pas affichées ici pour garder la lecture centrée sur les signaux."
        )


def render_drift_section(domain_shift: Any, model_quality_shift: Any) -> None:
    """Render the domain-shift summary."""

    st.markdown("#### Similarité au domaine d'entraînement")
    st.caption(
        "Objectif : savoir si les dernières images ressemblent au domaine attendu "
        "PlantVillage ou plutôt à un domaine OOD connu comme PlantDoc. L'image n'est "
        "pas stockée : seules des métriques dérivées sont comparées."
    )
    st.info(
        "Méthode : l'API calcule la moyenne récente de quelques descripteurs "
        "simples (luminosité, contraste, netteté, saturation, ratio vert/brun, "
        "confiances), puis mesure leur distance normalisée à chaque référence. "
        "Plus la distance est basse, plus le flux ressemble à la référence."
    )
    if not isinstance(domain_shift, dict) or not domain_shift.get("reference_available"):
        st.info("Référence de drift indisponible. Le monitoring reste limité aux métriques service.")
        return

    status = str(domain_shift.get("status", "insufficient_data"))
    risk_level = str(domain_shift.get("risk_level", "none"))
    closest_reference = domain_shift.get("closest_reference") or "non déterminée"
    status_label = format_domain_status(status)

    cols = st.columns(4, gap="medium")
    with cols[0]:
        render_monitoring_card(
            "Conclusion",
            status_label,
            "Résumé de similarité du flux récent, pas une preuve de performance.",
            tone=risk_to_tone(risk_level),
        )
    with cols[1]:
        render_monitoring_card(
            "Alerte domaine",
            format_risk_level(risk_level),
            "Aucun si le flux reste proche du domaine attendu ou d'une référence connue.",
            tone=risk_to_tone(risk_level),
        )
    with cols[2]:
        render_monitoring_card(
            "Référence proche",
            format_reference_label(str(closest_reference)),
            "Référence dont les métriques dérivées ressemblent le plus au flux récent.",
            tone="neutral",
        )
    with cols[3]:
        render_monitoring_card(
            "Fenêtre récente",
            str(int(domain_shift.get("window_size") or 0)),
            f"Minimum requis : {int(domain_shift.get('minimum_window_size') or 0)} événements.",
            tone="neutral",
        )

    distances = domain_shift.get("distances", {})
    render_horizontal_bars(
        "Similarité aux références",
        format_reference_mapping(distances),
        max_items=3,
        lower_is_better=True,
    )
    st.caption(
        "Ce sont des distances normalisées, pas des pourcentages : 0 signifie très proche ; "
        "au-dessus des seuils, le flux est considéré comme décalé."
    )

    prediction_drift = domain_shift.get("prediction_drift", {})
    if isinstance(prediction_drift, dict):
        left, right = st.columns(2, gap="medium")
        with left:
            render_monitoring_card(
                "Écart espèces",
                format_distance(prediction_drift.get("species_distance")),
                "Différence entre les espèces prédites récemment et la référence la plus proche.",
                tone="neutral",
            )
        with right:
            render_monitoring_card(
                "Écart maladies",
                format_distance(prediction_drift.get("disease_distance")),
                "Différence entre les maladies prédites récemment et la référence la plus proche.",
                tone="neutral",
            )

    signals = domain_shift.get("signals", [])
    render_drift_signals(signals)

    if isinstance(model_quality_shift, dict):
        st.markdown("#### Signal feedback")
        st.caption(
            "Le feedback ne détecte pas le data drift à lui seul. Il ajoute une indication "
            "de qualité perçue par l'utilisateur, utile pour prioriser les futures validations."
        )
        feedback_cols = st.columns(4, gap="medium")
        with feedback_cols[0]:
            render_monitoring_card(
                "État qualité",
                format_quality_status(str(model_quality_shift.get("status", "n/a"))),
                "Synthèse du signal issu des retours utilisateur.",
                tone=risk_to_tone(str(model_quality_shift.get("risk_level", "none"))),
            )
        with feedback_cols[1]:
            render_monitoring_card(
                "Risque qualité",
                format_risk_level(str(model_quality_shift.get("risk_level", "none"))),
                "Le risque monte si les désaccords deviennent fréquents.",
                tone=risk_to_tone(str(model_quality_shift.get("risk_level", "none"))),
            )
        with feedback_cols[2]:
            render_monitoring_card(
                "Désaccord",
                format_optional_percent(model_quality_shift.get("disagreement_rate")),
                "Part des feedbacks marqués comme incorrects.",
                tone="watch"
                if float(model_quality_shift.get("disagreement_rate") or 0.0) > 0.3
                else "good",
            )
        with feedback_cols[3]:
            render_monitoring_card(
                "Seuil feedback",
                str(int(model_quality_shift.get("minimum_feedback") or 0)),
                "Nombre minimal de retours avant d'interpréter le signal qualité.",
                tone="neutral",
            )


def render_feedback_summary(feedback: Any) -> None:
    """Render aggregate user feedback."""

    st.markdown("#### Retours utilisateur collectés")
    st.caption(
        "Ces retours sont stockés sans image et servent à identifier les classes contestées, "
        "pas à réentraîner automatiquement le modèle."
    )
    if not isinstance(feedback, dict):
        st.info("Aucun retour utilisateur disponible.")
        return
    high_confidence_disagreement_count = int(
        feedback.get("high_confidence_disagreement_count") or 0
    )
    high_confidence_disagreement_rate = float(
        feedback.get("high_confidence_disagreement_rate") or 0.0
    )
    high_confidence_threshold = float(feedback.get("high_confidence_threshold") or 0.9)

    cols = st.columns(4, gap="medium")
    with cols[0]:
        render_monitoring_card(
            "Retours",
            str(int(feedback.get("total_feedback") or 0)),
            "Nombre de feedbacks explicitement envoyés après une prédiction.",
            tone="neutral",
        )
    with cols[1]:
        render_monitoring_card(
            "Désaccord",
            format_optional_percent(feedback.get("disagreement_rate")),
            "Part des retours où l'utilisateur signale une prédiction incorrecte.",
            tone="watch" if float(feedback.get("disagreement_rate") or 0.0) > 0.3 else "good",
        )
    with cols[2]:
        render_monitoring_card(
            "Forte confiance contestée",
            str(high_confidence_disagreement_count),
            f"Feedback incorrect alors qu'une confiance concernée dépasse "
            f"{format_optional_percent(high_confidence_threshold)}.",
            tone="watch" if high_confidence_disagreement_count else "good",
        )
    with cols[3]:
        render_monitoring_card(
            "Dernier retour",
            short_timestamp(feedback.get("last_feedback_at")),
            f"Dernier feedback reçu. Taux forte confiance : "
            f"{format_optional_percent(high_confidence_disagreement_rate)}.",
            tone="neutral",
        )
    st.divider()
    left, right = st.columns(2, gap="medium")
    with left:
        render_ranked_counts(
            "Maladies contestées",
            feedback.get("disputed_disease_distribution", {}),
            "Classes maladie le plus souvent associées à un feedback utilisateur.",
        )
    with right:
        render_ranked_counts(
            "Corrections proposées",
            feedback.get("corrected_disease_distribution", {}),
            "Maladies indiquées par l'utilisateur quand il conteste le diagnostic.",
        )


def render_recent_events(events: Any) -> None:
    """Render recent prediction events and latency trend."""

    st.markdown("#### Derniers événements enregistrés")
    st.caption(
        "Cette section garde la trace technique : statut, classes prédites, confiances et latence. "
        "Elle permet de relier les agrégats du dashboard à des appels API réels."
    )
    if not isinstance(events, list) or not events:
        st.info("Aucun événement récent à afficher.")
        return

    visible_events = [
        {
            "timestamp": short_timestamp(event.get("timestamp")),
            "status": event.get("status"),
            "species": event.get("species"),
            "disease": event.get("disease"),
            "species_confidence": format_optional_percent(event.get("species_confidence")),
            "disease_confidence": format_optional_percent(event.get("disease_confidence")),
            "latency_ms": format_ms(event.get("latency_ms")),
        }
        for event in events
        if isinstance(event, dict)
    ]
    st.dataframe(visible_events, width="stretch", hide_index=True)


def render_horizontal_bars(
    title: str,
    values: Any,
    *,
    max_items: int = 6,
    lower_is_better: bool = False,
) -> None:
    """Render compact horizontal bars from a mapping."""

    if not isinstance(values, dict) or not values:
        st.write(title)
        st.caption("Aucune donnée.")
        return
    numeric_items = [
        (str(label), float(value))
        for label, value in values.items()
        if isinstance(value, int | float)
    ]
    numeric_items.sort(key=lambda item: item[1], reverse=not lower_is_better)
    selected_items = numeric_items[:max_items]
    max_value = max((value for _, value in selected_items), default=0.0)

    rows = []
    for label, value in selected_items:
        width = 0 if max_value <= 0 else int((value / max_value) * 100)
        rows.append(
            f"""
            <div class="bar-row">
                <div class="bar-label">{html.escape(label)}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{width}%"></div>
                </div>
                <div class="bar-value">{format_chart_value(value, lower_is_better=lower_is_better)}</div>
            </div>
            """
        )

    hint = "Plus bas = plus proche." if lower_is_better else ""
    st.html(
        f"""
        <div class="chart-panel">
            <div class="chart-title">{html.escape(title)}</div>
            {''.join(rows)}
            <div class="chart-hint">{html.escape(hint)}</div>
        </div>
        """,
    )


def render_ranked_counts(title: str, values: Any, help_text: str) -> None:
    """Render ranked count values without adding another chart."""

    st.write(title)
    st.caption(help_text)
    if not isinstance(values, dict) or not values:
        st.caption("Aucune donnée.")
        return
    rows = sorted(
        values.items(),
        key=lambda item: int(item[1]) if isinstance(item[1], int | float) else 0,
        reverse=True,
    )[:5]
    for label, value in rows:
        st.markdown(f"- **{html.escape(str(label))}** : {int(value)}")


def render_drift_signals(signals: Any) -> None:
    """Render only the most actionable drift signals."""

    if not isinstance(signals, list) or not signals:
        st.caption("Aucun signal fort sur les métriques dérivées de l'image.")
        return

    st.write("Métriques image les plus éloignées")
    st.caption(
        "Ces lignes expliquent pourquoi le flux peut s'éloigner de la référence, "
        "sans afficher toutes les métriques techniques."
    )
    rows = []
    for signal in signals[:3]:
        if not isinstance(signal, dict):
            continue
        rows.append(
            {
                "métrique": signal.get("metric"),
                "niveau": signal.get("level"),
                "direction": signal.get("direction"),
                "z_score": signal.get("z_score"),
            }
        )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def format_distance(value: Any) -> str:
    """Format a drift distance value."""

    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.2f}"


def format_chart_value(value: float, *, lower_is_better: bool = False) -> str:
    """Format compact chart values for counts, ratios and distances."""

    if value.is_integer():
        return str(int(value))
    if 0 <= value <= 1 and not lower_is_better:
        return f"{value:.0%}"
    return f"{value:.2f}"


def format_domain_status(status: str) -> str:
    """Return a readable domain-shift status."""

    return {
        "in_domain": "Proche du domaine attendu",
        "ood_like": "Proche OOD connu",
        "reference_shift": "Décalage modéré",
        "unknown_shift": "Décalage inconnu",
        "insufficient_data": "Données insuffisantes",
    }.get(status, status.replace("_", " "))


def format_reference_label(reference: str) -> str:
    """Return a readable monitoring reference label."""

    return {
        "plantvillage_in_domain": "PlantVillage attendu",
        "plantdoc_ood": "PlantDoc OOD connu",
    }.get(reference, reference.replace("_", " "))


def format_reference_mapping(values: Any) -> dict[str, float]:
    """Format reference keys while preserving their numeric distances."""

    if not isinstance(values, dict):
        return {}
    return {
        format_reference_label(str(label)): value
        for label, value in values.items()
        if isinstance(value, int | float)
    }


def format_quality_status(status: str) -> str:
    """Return a readable model-quality status."""

    return {
        "insufficient_feedback": "Feedback insuffisant",
        "feedback_stable": "Feedback stable",
        "quality_drift_suspected": "Qualité à surveiller",
        "feedback_confirms_domain_risk": "Risque confirmé",
    }.get(status, status.replace("_", " "))


def format_risk_level(risk_level: str) -> str:
    """Return a readable risk label."""

    return {
        "none": "Aucun",
        "watch": "Surveillance",
        "warning": "Avertissement",
        "critical": "Critique",
    }.get(risk_level, risk_level.replace("_", " "))


def risk_to_tone(risk_level: str) -> str:
    """Map API risk levels to CSS card tones."""

    if risk_level == "critical":
        return "critical"
    if risk_level in {"warning", "watch"}:
        return "watch"
    if risk_level == "none":
        return "good"
    return "neutral"


def short_timestamp(value: Any) -> str:
    """Shorten an ISO timestamp for compact dashboard tables."""

    if not value:
        return "Aucun"
    text = str(value)
    if "T" not in text:
        return text
    date_part, time_part = text.split("T", maxsplit=1)
    return f"{date_part} {time_part[:8]}"


def _numeric(value: Any) -> float:
    """Return a numeric value with a stable fallback for monitoring thresholds."""

    if isinstance(value, int | float):
        return float(value)
    return 0.0


def render_last_result(api_url: str) -> None:
    """Render the last API response, if any."""

    payload = st.session_state.get("last_response")
    status_code = st.session_state.get("last_status_code")
    if payload is None:
        return

    st.divider()
    st.subheader("Résultat")

    if status_code != 200:
        detail = payload.get("detail", "Erreur inconnue.")
        if status_code == 503:
            st.warning(detail)
            st.info("Les modèles ne sont pas encore disponibles. Réessayez dans quelques instants.")
        elif status_code == 400:
            st.error(detail)
        else:
            st.error(f"Erreur {status_code} : {detail}")
        return

    if payload.get("status") == "uncertain_species":
        render_uncertain_species(payload)
        render_feedback_form(api_url, payload)
        return

    render_successful_prediction(payload)
    render_feedback_form(api_url, payload)


def render_uncertain_species(payload: dict[str, Any]) -> None:
    """Render the case where the API asks for species confirmation."""

    species = payload.get("species", {})
    st.warning(payload.get("action_required", "Espèce incertaine."))
    render_prediction_overview(
        species=species,
        disease=None,
        include_disease_info=False,
    )
    render_ranked_predictions(
        "Top prédictions espèce",
        species.get("top_predictions", []),
        label_formatter=display_species,
    )
    st.info("Passez en mode **Manuel** avec l'espèce correcte pour lancer le diagnostic maladie.")


def render_successful_prediction(payload: dict[str, Any]) -> None:
    """Render a successful diagnosis."""

    species = payload.get("species", {})
    disease = payload.get("disease") or {}
    render_prediction_overview(
        species=species,
        disease=disease,
        include_disease_info=True,
    )

    render_ranked_predictions(
        "Top prédictions espèce",
        species.get("top_predictions", []),
        label_formatter=display_species,
    )
    render_ranked_predictions(
        "Top prédictions maladie",
        disease.get("top_predictions", []),
        label_formatter=lambda label: display_disease(label, species.get("species")),
    )

    with st.expander("Détails techniques"):
        st.json(payload)


def render_feedback_form(api_url: str, payload: dict[str, Any]) -> None:
    """Render a compact user feedback form for iterative monitoring."""

    st.divider()
    st.subheader("Retour sur la prédiction")
    st.caption(
        "Le retour est enregistré pour le monitoring. L'image envoyée n'est pas conservée."
    )

    if st.session_state.get("last_feedback_sent"):
        st.success("Retour déjà enregistré pour cette prédiction.")
        return

    species = payload.get("species", {})
    disease = payload.get("disease") or {}
    verdict_labels = {
        "Correcte": "correct",
        "Incorrecte": "incorrect",
        "Je ne sais pas": "unsure",
    }

    verdict_label = st.radio(
        "Cette prédiction vous semble-t-elle correcte ?",
        options=list(verdict_labels),
        horizontal=True,
    )
    corrected_species = None
    corrected_disease = None
    if verdict_labels[verdict_label] == "incorrect":
        species_options = ["Non renseigné", *SPECIES_OPTIONS.values()]
        predicted_species_label = display_species(species.get("species"))
        species_index = (
            species_options.index(predicted_species_label)
            if predicted_species_label in species_options
            else 0
        )
        selected_species = st.selectbox(
            "Espèce correcte",
            options=species_options,
            index=species_index,
            help=(
                "Si l'espèce prédite était correcte, gardez-la sélectionnée : "
                "la liste des maladies sera filtrée pour cette espèce."
            ),
        )
        corrected_species = (
            label_to_species(selected_species)
            if selected_species != "Non renseigné"
            else None
        )

        disease_options = feedback_disease_options(corrected_species)
        selected_disease = st.selectbox(
            "Maladie correcte",
            options=["Non renseignée", *disease_options],
            disabled=corrected_species is None,
            help="La liste est limitée aux maladies connues pour l'espèce correcte.",
        )
        corrected_disease = (
            feedback_disease_label_to_key(selected_disease, corrected_species)
            if corrected_species is not None and selected_disease != "Non renseignée"
            else None
        )
    comment = st.text_area("Commentaire optionnel", max_chars=500)
    submitted = st.button("Envoyer le retour", type="primary", width="content")

    if not submitted:
        return

    feedback_payload = {
        "verdict": verdict_labels[verdict_label],
        "prediction_status": payload.get("status"),
        "predicted_species": species.get("species"),
        "predicted_disease": disease.get("disease"),
        "predicted_species_confidence": species.get("confidence"),
        "predicted_disease_confidence": disease.get("confidence"),
        "corrected_species": corrected_species,
        "corrected_disease": corrected_disease,
        "comment": comment.strip() or None,
    }
    response = submit_feedback(api_url, feedback_payload)
    if response.status_code == 200 and response.payload.get("stored"):
        st.session_state["last_feedback_sent"] = True
        cached_get_monitoring_summary.clear()
        st.success(response.payload.get("message", "Retour enregistré."))
    else:
        st.error(response.payload.get("detail", "Impossible d'enregistrer le retour."))


def render_prediction_overview(
    *,
    species: dict[str, Any],
    disease: dict[str, Any] | None,
    include_disease_info: bool,
) -> None:
    """Render the main result with the uploaded image and disease information."""

    left, right = st.columns([0.8, 1.6], gap="large")

    with left:
        image_bytes = st.session_state.get("last_image_bytes")
        image_name = st.session_state.get("last_image_name", "Image envoyée")
        if image_bytes:
            render_responsive_image(image_bytes, caption=image_name)

    with right:
        disease_label = disease.get("disease") if disease else None
        species_label = species.get("species")
        render_badges(
            [
                ("Espèce", display_species(species_label), species.get("confidence")),
                (
                    "Maladie",
                    display_disease(disease_label, species_label)
                    if disease_label
                    else "En attente de confirmation",
                    disease.get("confidence") if disease else None,
                ),
            ]
        )
        if include_disease_info:
            render_disease_information(species_label, disease_label)


def render_disease_information(species: Any, disease: Any) -> None:
    """Render the agronomic information card for the predicted disease."""

    if not disease or disease == "Healthy":
        st.success("Aucune maladie foliaire détectée parmi les classes entraînées.")
        return

    info = DISEASE_INFO.get((str(species), str(disease)))
    if info is None:
        st.info("Informations détaillées indisponibles pour cette classe.")
        return

    title = html.escape(str(info["title"]))
    english = html.escape(str(info["english"]))
    url = html.escape(str(info["url"]))

    st.markdown(f"### À propos - [{title}]({url}) (*{english}*)")
    st.markdown(
        f"""
        <div class="info-panel">
            <h4>Plus d'informations</h4>
            <div class="info-row">
                <span class="info-icon">🌿</span>
                <span><strong>Agent causatif :</strong> {html.escape(str(info["agent"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">🩺</span>
                <span><strong>Traitement curatif :</strong> {html.escape(str(info["curative"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">💊</span>
                <span><strong>Traitement préventif :</strong> {html.escape(str(info["preventive"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">🛡️</span>
                <span><strong>Saison / gravité :</strong> {html.escape(str(info["season_gravity"]))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_responsive_image(
    image_bytes: bytes,
    *,
    caption: str,
    css_class: str = "",
) -> None:
    """Render an image with CSS constraints instead of a fixed pixel width."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_caption = html.escape(caption)
    extra_class = f" {css_class}" if css_class else ""
    st.markdown(
        f"""
        <img class="responsive-image{extra_class}" src="data:image/jpeg;base64,{encoded}" alt="{safe_caption}">
        <div class="image-caption">{safe_caption}</div>
        """,
        unsafe_allow_html=True,
    )


def render_ranked_predictions(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    label_formatter: Any,
) -> None:
    """Render the top prediction candidates returned by the API."""

    if not candidates:
        return

    st.markdown(f"#### {title}")
    for rank, candidate in enumerate(candidates[:3], start=1):
        label = label_formatter(candidate.get("label"))
        confidence = float(candidate.get("confidence") or 0.0)
        st.markdown(
            f"""
            <div class="ranked-row">
                <div class="ranked-label">{rank}. {html.escape(label)}</div>
                <div class="ranked-confidence">{confidence:.0%}</div>
            </div>
            <div class="confidence-bar compact">
                <div class="confidence-bar-fill" style="width:{int(confidence * 100)}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_badges(items: list[tuple[str, str, float | None]]) -> None:
    """Render compact result badges with a confidence progress bar."""

    columns = st.columns(len(items), gap="medium")
    for column, (label, value, confidence) in zip(columns, items):
        pct = int(float(confidence) * 100) if confidence is not None else 0
        safe_label = html.escape(label)
        safe_value = html.escape(value)
        safe_confidence = html.escape(format_confidence(confidence))
        bar_html = (
            f'<div class="confidence-bar">'
            f'<div class="confidence-bar-fill" style="width:{pct}%"></div>'
            f"</div>"
            if confidence is not None
            else ""
        )
        with column:
            st.markdown(
                f"""
                <div class="diagnosis-badge">
                    <div class="diagnosis-label">{safe_label}</div>
                    <div class="diagnosis-value">{safe_value}</div>
                    {bar_html}
                    <div class="diagnosis-confidence">{safe_confidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def label_to_species(label: str) -> str:
    """Return the API species key matching a French display label."""

    reverse_mapping = {display: key for key, display in SPECIES_OPTIONS.items()}
    return reverse_mapping[label]


def label_to_disease(label: str) -> str:
    """Return the API disease key matching a French display label."""

    reverse_mapping = {display: key for key, display in DISEASE_LABELS.items()}
    return reverse_mapping[label]


def feedback_disease_options(species: str | None) -> list[str]:
    """Return disease feedback labels filtered by corrected species."""

    if not species:
        return []

    disease_keys = {
        disease_key
        for species_key, disease_key in DISEASE_INFO
        if species_key == species
    }
    disease_keys.add("Healthy")
    return [
        feedback_disease_label(disease_key, species)
        for disease_key in sorted(disease_keys, key=lambda key: feedback_disease_label(key, species))
    ]


def feedback_disease_label_to_key(label: str, species: str | None) -> str:
    """Return the disease key matching a species-aware feedback label."""

    if not species:
        return label_to_disease(label)
    mapping = {
        feedback_disease_label(disease_key, species): disease_key
        for disease_key in {
            disease_key
            for species_key, disease_key in DISEASE_INFO
            if species_key == species
        }
        | {"Healthy"}
    }
    return mapping[label]


def feedback_disease_label(disease_key: str, species: str) -> str:
    """Return a French disease label with the English class name in parentheses."""

    if disease_key == "Healthy":
        return "Feuille saine (Healthy)"

    info = DISEASE_INFO.get((species, disease_key))
    if info is not None:
        return f"{info['title']} ({info['english']})"

    french_label = DISEASE_LABELS.get(disease_key, disease_key.replace("_", " "))
    english_label = disease_key.replace("_", " ")
    return f"{french_label} ({english_label})"


def display_species(value: Any) -> str:
    """Format a species key for display."""

    if not value:
        return "Non déterminée"
    return SPECIES_OPTIONS.get(str(value), str(value))


def display_disease(value: Any, species: Any | None = None) -> str:
    """Format a disease label for display."""

    if not value:
        return "Non déterminée"
    if species and (str(species), str(value)) in DISEASE_INFO:
        return str(DISEASE_INFO[(str(species), str(value))]["title"])
    return DISEASE_LABELS.get(str(value), str(value).replace("_", " "))


def format_confidence(value: float | None) -> str:
    """Format confidence as a percentage."""

    if value is None:
        return "Confiance non disponible"
    return f"Confiance : {float(value):.0%}"


def format_optional_percent(value: Any) -> str:
    """Format a nullable API ratio as a compact percentage."""

    if value is None:
        return "N/A"
    return f"{float(value):.0%}"


def format_ms(value: Any) -> str:
    """Format a nullable latency value."""

    if value is None:
        return "N/A"
    return f"{float(value):.0f} ms"


if __name__ == "__main__":
    main()
